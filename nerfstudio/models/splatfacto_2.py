# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from pathlib import Path

import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from nerfstudio.utils.writer import GLOBAL_BUFFER

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from pytorch_msssim import SSIM
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.gauss_utils import transform_gaussians
from nerfstudio.utils.obj_3d_seg import Object3DSeg
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SplatfactoFinetuneModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoFinetuneModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    strategy: Literal["default", "mcmc"] = "default"
    """The default strategy will be used if strategy is not specified. Other strategies, e.g. mcmc, can be used."""
    max_gs_num: int = 1_000_000
    """Maximum number of GSs. Default to 1_000_000."""
    noise_lr: float = 5e5
    """MCMC samping noise learning rate. Default to 5e5."""
    mcmc_opacity_reg: float = 0.01
    """Regularization term for opacity in MCMC strategy. Only enabled when using MCMC strategy"""
    mcmc_scale_reg: float = 0.01
    """Regularization term for scale in MCMC strategy. Only enabled when using MCMC strategy"""

    ## Finetuning parameters
    obj_mask_file: Optional[Path] = None
   

class SplatfactoFinetuneModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoFinetuneModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        print("Populating Splatfacto model modules...")
        print(f"=== DEBUG: Training Setup ===")
        print(f"Number of training data: {self.num_train_data}")
        print(f"=== END DEBUG ===")

        if "step" not in GLOBAL_BUFFER:
            GLOBAL_BUFFER["step"] = self.step = 0
        means = torch.zeros((1, 3)).float().cuda()  
        scales = torch.zeros((1, 3)).float().cuda()
        quats = torch.zeros((1, 4)).float().cuda()
        dim_sh = num_sh_bases(self.config.sh_degree)
        features_dc = torch.zeros((1, 3)).float().cuda()
        features_rest = torch.zeros((1, dim_sh-1, 3)).float().cuda()
        opacities = torch.zeros((1, 1)).float().cuda()
        
        # Convert to parameters
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(scales),
                "quats": torch.nn.Parameter(quats),
                "features_dc": torch.nn.Parameter(features_dc),
                "features_rest": torch.nn.Parameter(features_rest),
                "opacities": torch.nn.Parameter(opacities),
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        if self.config.strategy == "default":
            # Strategy for GS densification
            self.strategy = DefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(f"""Splatfacto does not support strategy {self.config.strategy} 
                             Currently, the supported strategies include default and mcmc.""")

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):
        print(f"!!! Loading state_dict, training={self.training}")
        
        # Always check if we need to do object-specific loading
        needs_object_filtering = (
            self.config.obj_mask_file is not None and 
            "gauss_params.means" in dict and 
            dict["gauss_params.means"].shape[0] > 1  # More than our placeholder size
        )
        
        if needs_object_filtering:
            print("!!! Detected object-filtered checkpoint, handling special loading...")
            
            # Load object mask (needed for both training and inference)
            if isinstance(self.config.obj_mask_file, Path):
                self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
                print(f"Loaded object mask from {self.config.obj_mask_file}")
            else:
                raise ValueError(f"Unknown type of obj_mask_file {self.config.obj_mask_file}")
            
            # Handle backwards compatibility
            if "means" in dict:
                for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    dict[f"gauss_params.{p}"] = dict[p]
            
            if self.training:
                # TRAINING MODE: Filter and transform gaussians
                print("!!! Training mode: filtering and transforming gaussians")
                
                self.obj_mask = self.obj_3d_seg.query(dict["gauss_params.means"].cuda()).cpu()
                print(f"Gaussians inside object mask: {self.obj_mask.sum().item()}")
                
                if self.obj_mask.sum() == 0:
                    CONSOLE.print("[red]No gaussians inside the object mask![/red]")
                    return
                
                # Transform gaussians
                pose = self.obj_3d_seg.pose_change
                dict["gauss_params.means"][self.obj_mask], dict["gauss_params.quats"][self.obj_mask] = \
                    transform_gaussians(pose.cpu(), dict["gauss_params.means"][self.obj_mask], dict["gauss_params.quats"][self.obj_mask])

                # Filter to only object gaussians
                filtered_data = {}
                for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    filtered_data[name] = dict[f"gauss_params.{name}"][self.obj_mask]
                
                # Resize existing parameters IN-PLACE
                for name, new_data in filtered_data.items():
                    existing_param = self.gauss_params[name]
                    existing_param.data = new_data.to(existing_param.device).detach()
                    print(f"Resized {name}: {existing_param.shape}")
                
                # Store fixed gaussians
                self.gauss_params_fixed = {}
                non_obj_mask = ~self.obj_mask
                for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    self.gauss_params_fixed[name] = dict[f"gauss_params.{name}"][non_obj_mask].to(self.device)
                
                print(f"[INFO] Loaded {filtered_data['means'].shape[0]} Gaussians for fine-tuning.")
                
            else:
                # INFERENCE MODE: Load all gaussians directly
                print("!!! Inference mode: loading all gaussians directly")
                
                # Resize model to match checkpoint size
                checkpoint_size = dict["gauss_params.means"].shape[0]
                print(f"Resizing model from {self.means.shape[0]} to {checkpoint_size} gaussians")
                
                for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    checkpoint_data = dict[f"gauss_params.{name}"]
                    existing_param = self.gauss_params[name]
                    existing_param.data = checkpoint_data.to(existing_param.device).detach()
                    print(f"Loaded {name}: {existing_param.shape}")
            
            self.step = 0
            
        else:
            # Normal loading (original checkpoint without object filtering)
            print("!!! Normal checkpoint loading")
            super().load_state_dict(dict, **kwargs)
            self.step = 0

        # Debug info
        print(f"=== FINAL STATE ===")
        print(f"Gaussians loaded: {self.means.shape[0]}")
        print(f"Fixed gaussians: {getattr(self, 'gauss_params_fixed', {}).get('means', torch.tensor([])).shape[0] if hasattr(self, 'gauss_params_fixed') else 0}")
        print(f"Training mode: {self.training}")
        print(f"=== END ===")

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],  # the learning rate for the "means" attribute of the GS
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # print("!!!Creating training callbacks for Splatfacto model")
        trainer = training_callback_attributes.trainer
        # print(f"!!! Trainer start_step: {getattr(trainer, '_start_step', 'not found')}")
        # print(f"!!! Trainer max_iterations: {getattr(trainer.config, 'max_num_iterations', 'not found')}")
        # print(f"!!! Training will run: {getattr(trainer.config, 'max_num_iterations', 0) - getattr(trainer, '_start_step', 0)} iterations")
        
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs
    
    def clear_optimizer_state(self, optimizers):
        """Clear optimizer state after parameter resizing"""
        # print("!!! Clearing optimizer state after parameter resizing...")
        
        for name, optimizer in optimizers.optimizers.items():
            if name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                # Clear the optimizer state for this parameter group
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param in optimizer.state:
                            print(f"Clearing state for {name}")
                            optimizer.state[param].clear()
                            # Re-initialize the state
                            optimizer.state[param] = {}

    def step_cb(self, optimizers: Optimizers, step):
        # print(f"!!!Step callback: {step}")  
        if step == 20000:  
            self.clear_optimizer_state(optimizers)
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers"""
        
        # # Check if we have actual parameters loaded
        # if self.means.shape[0] == 0:
        #     print("!!! WARNING: get_param_groups called with empty parameters!")
        #     print("!!! This should happen after load_state_dict")
        #     # Return minimal groups that will be updated later
        #     return {"empty": []}
        
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        
        # # Debug the actual parameter counts
        # print(f"=== DEBUG: Parameter Groups (After Loading) ===")
        # for group_name, params in gps.items():
        #     if params:
        #         param_count = sum(p.numel() for p in params)
        #         requires_grad = all(p.requires_grad for p in params)
        #         print(f"Group '{group_name}': {len(params)} tensors, {param_count} params, requires_grad={requires_grad}")
        #     else:
        #         print(f"Group '{group_name}': EMPTY")
        # print(f"=== END DEBUG ===")
        
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        # make xy grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        # print(f"!!! get_outputs called at step {getattr(self, 'step', 'unknown')}")

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

            if hasattr(self, "gauss_params_fixed") and self.training:
                # print(f"!!! Combining {self.means.shape[0]} trainable + {self.gauss_params_fixed['means'].shape[0]} fixed gaussians")
                
                opacities_crop = torch.cat([opacities_crop, self.gauss_params_fixed["opacities"]], dim=0)
                means_crop = torch.cat([means_crop, self.gauss_params_fixed["means"]], dim=0)
                features_dc_crop = torch.cat([features_dc_crop, self.gauss_params_fixed["features_dc"]], dim=0)
                features_rest_crop = torch.cat([features_rest_crop, self.gauss_params_fixed["features_rest"]], dim=0)
                scales_crop = torch.cat([scales_crop, self.gauss_params_fixed["scales"]], dim=0)
                quats_crop = torch.cat([quats_crop, self.gauss_params_fixed["quats"]], dim=0)
            elif hasattr(self, "gauss_params_fixed"):
                # During evaluation, also include fixed gaussians
                # print(f"!!! (Eval) Combining {self.means.shape[0]} trainable + {self.gauss_params_fixed['means'].shape[0]} fixed gaussians")
                
                opacities_crop = torch.cat([opacities_crop, self.gauss_params_fixed["opacities"]], dim=0)
                means_crop = torch.cat([means_crop, self.gauss_params_fixed["means"]], dim=0)
                features_dc_crop = torch.cat([features_dc_crop, self.gauss_params_fixed["features_dc"]], dim=0)
                features_rest_crop = torch.cat([features_rest_crop, self.gauss_params_fixed["features_rest"]], dim=0)
                scales_crop = torch.cat([scales_crop, self.gauss_params_fixed["scales"]], dim=0)
                quats_crop = torch.cat([quats_crop, self.gauss_params_fixed["quats"]], dim=0)


        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # print(f"!!! get_loss_dict called at step {getattr(self, 'step', 'unknown')}")
        # print("#######Computing loss dict for Splatfacto model...")
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
    
    # def after_train(
    #     self, training_callback_attributes: TrainingCallbackAttributes, optimizers: Optimizers, step: int
    # ):
    #     """Called after each training iteration."""
    #     print(f"After train step {step}")
    #     if isinstance(self.strategy, DefaultStrategy):
    #         self.strategy.after_train(
    #             params=self.gauss_params,
    #             optimizers=optimizers,
    #             state=self.strategy_state,
    #             step=self.step,
    #             info=self.info,
    #         )
    #     elif isinstance(self.strategy, MCMCStrategy):
    #         self.strategy.after_train(
    #             params=self.gauss_params,
    #             optimizers=optimizers,
    #             state=self.strategy_state,
    #             step=self.step,
    #         )
    #     else:
    #         raise ValueError(f"Unknown strategy {self.strategy}")


    def after_train(self, optimizers, *, step: int):
        """Custom fine-tuning strategy focused on quality and stability"""
        
        if self.step % 200 == 0:
            scale_exp = torch.exp(self.scales)
            max_scale = scale_exp.max(dim=-1).values
            min_scale = scale_exp.min(dim=-1).values
            aspect_ratios = max_scale / min_scale
            print(f"Step {self.step} | Gaussians: {self.num_points} | "
                f"Opacity: {torch.sigmoid(self.opacities).mean().item():.3f} | "
                f"Max aspect ratio: {aspect_ratios.max().item():.1f} | "
                f"Mean aspect ratio: {aspect_ratios.mean().item():.1f}")

        assert step == self.step

        # Apply our custom fine-tuning strategy every 100 steps
        if self.step % 100 == 0 and self.step > 0:
            with torch.no_grad():
                self._apply_finetuning_constraints(optimizers)
        
        # Reset opacities less frequently to maintain stability
        # if self.step % 3000 == 0 and self.step > 0:
        #     self._reset_opacities_gentle(optimizers)

    def _apply_finetuning_constraints(self, optimizers):
        """Apply constraints to prevent spikey gaussians and maintain quality"""
        
        scale_exp = torch.exp(self.scales)
        num_before = self.num_points
        
        # 1. CULL EXTREMELY ELONGATED GAUSSIANS
        max_scale = scale_exp.max(dim=-1).values
        min_scale = scale_exp.min(dim=-1).values
        aspect_ratios = max_scale / min_scale
        
        # Remove gaussians with extreme aspect ratios (these cause the spikes)
        elongated_mask = aspect_ratios > 10.0
        
        # 2. CULL VERY LARGE GAUSSIANS
        large_mask = max_scale > 0.1  # Adjust based on your scene scale
        
        # 3. CULL LOW OPACITY GAUSSIANS
        low_opacity_mask = torch.sigmoid(self.opacities).squeeze() < 0.005
        
        # Skip object region check for now to avoid device issues
        # outside_mask = torch.zeros_like(elongated_mask)
        
        # Combine culling criteria (without outside mask for now)
        cull_mask = elongated_mask | large_mask | low_opacity_mask
        
        if cull_mask.sum() > 0:
            print(f"    Culling {cull_mask.sum().item()} gaussians: "
                f"{elongated_mask.sum().item()} elongated, "
                f"{large_mask.sum().item()} large, "
                f"{low_opacity_mask.sum().item()} low opacity")
            
            # Remove culled gaussians
            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(param[~cull_mask])
            
            # Update optimizers
            self.remove_from_all_optim(optimizers, cull_mask)
        
        # 5. REGULARIZE REMAINING GAUSSIAN SCALES
        self._regularize_gaussian_scales()
        
        print(f"    Gaussians: {num_before} -> {self.num_points}")

    def _regularize_gaussian_scales(self):
        """Regularize gaussian scales to prevent extreme elongation"""
        with torch.no_grad():
            scale_exp = torch.exp(self.scales)
            
            # Clamp scales to reasonable range
            min_scale = 0.001  # Minimum scale
            max_scale = 0.05   # Maximum scale (adjust for your scene)
            
            scale_exp = torch.clamp(scale_exp, min_scale, max_scale)
            
            # Limit aspect ratios by bringing extreme scales closer together
            max_scale_per_gaussian = scale_exp.max(dim=-1, keepdim=True).values
            min_scale_per_gaussian = scale_exp.min(dim=-1, keepdim=True).values
            aspect_ratio = max_scale_per_gaussian / min_scale_per_gaussian
            
            # If aspect ratio is too high, reduce the max scales
            max_allowed_ratio = 5.0
            too_elongated = aspect_ratio > max_allowed_ratio
            
            if too_elongated.any():
                # Reduce the largest scale to maintain max ratio
                target_max_scale = min_scale_per_gaussian * max_allowed_ratio
                scale_exp = torch.where(
                    too_elongated.expand_as(scale_exp),
                    torch.minimum(scale_exp, target_max_scale.expand_as(scale_exp)),
                    scale_exp
                )
            
            # Update the parameters
            self.scales.data = torch.log(scale_exp)

    def _reset_opacities_gentle(self, optimizers):
        """Gently reset opacities without being too aggressive"""
        print(f"    Gentle opacity reset at step {self.step}")
        
        with torch.no_grad():
            # Don't reset all opacities, just clamp very high ones
            current_opacities = torch.sigmoid(self.opacities)
            
            # Reset only very high opacities (> 0.95) to avoid oversaturation
            high_opacity_mask = current_opacities > 0.95
            if high_opacity_mask.any():
                # Reset high opacities to a more reasonable value
                target_opacity = 0.8
                self.opacities.data[high_opacity_mask] = torch.logit(
                    torch.tensor(target_opacity, device=self.device)
                )
                print(f"    Reset {high_opacity_mask.sum().item()} high opacities")
        
        # Reset optimizer state for opacities
        if "opacities" in optimizers.optimizers:
            optim = optimizers.optimizers["opacities"]
            if optim.param_groups and optim.param_groups[0]["params"]:
                param = optim.param_groups[0]["params"][0]
                if param in optim.state and "exp_avg" in optim.state[param]:
                    optim.state[param]["exp_avg"] = torch.zeros_like(optim.state[param]["exp_avg"])
                    optim.state[param]["exp_avg_sq"] = torch.zeros_like(optim.state[param]["exp_avg_sq"])

    # You'll also need these helper methods from your original code:
    def remove_from_all_optim(self, optimizers, deleted_mask):
        """Remove deleted gaussians from all optimizers"""
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            if group in optimizers.optimizers:
                self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """Remove deleted gaussians from a specific optimizer"""
        if not new_params:
            return
            
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state.get(param, {})
        
        # Update state tensors
        for state_key in ["exp_avg", "exp_avg_sq"]:
            if state_key in param_state:
                param_state[state_key] = param_state[state_key][~deleted_mask]
        
        # Update parameter groups
        optimizer.param_groups[0]["params"] = new_params
        
        # Update state dict
        if param in optimizer.state:
            del optimizer.state[param]
        if new_params:
            optimizer.state[new_params[0]] = param_state