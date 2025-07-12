import itertools
import torch
import matplotlib.pyplot as plt
import numpy as np
# import pycolmap

# from lightglue.utils import rbd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from nerfstudio.utils.img_utils import points2D_to_point_masks
# from nerfstudio.utils.proj_utils import project_points
from pyquaternion import Quaternion
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.measure import marching_cubes

import open3d as o3d
from skimage.measure import marching_cubes
import os

class Object3DSeg:
    """
    Object 3D segmentation represented as binary voxel grid
    """
    def __init__(
        self, bbox_min, bbox_max, voxel, pose_change,
        tight_bbox=None, mask_dilate_uniform=0, mask_dilate_top=5
    ):
        """
        Args:
            bbox_min (3-tuple): Min corner of the object bounding box
            bbox_max (3-tuple): Max corner of the object bounding box
            voxel (N, M, K tensor): Binary voxel grid representing obj 3D seg
            pose_change (4x4 tensor): 6DoF object pose change
            tight_bbox (3-tuple-tuple): Tight bounding box for the object
            mask_dilate_uniform (int): Uniform dilation for the object mask
            mask_dilate_top (int): Non-uniform dilation for the object mask
        """
        assert voxel.dtype == torch.bool
        assert all([min_ < max_ for min_, max_ in zip(bbox_min, bbox_max)])
        assert pose_change == None or pose_change.shape == (4, 4)
        self.bbox_min = torch.tensor(bbox_min).to(voxel.device).float()
        self.bbox_max = torch.tensor(bbox_max).to(voxel.device).float()
        self.dims = self.bbox_max - self.bbox_min
        self.voxel = voxel
        self.voxel_original = voxel.clone()
        self.tight_bbox = tight_bbox
        self.pose_change = pose_change
        # NOTE: We must dilate the object 3D segmentation to account for
        #       densities exceeding the object boundaries
        self.mask_dilate_uniform = mask_dilate_uniform
        self.mask_dilate_top = mask_dilate_top
        # self.visualize("/home/ziqi/Desktop/test")
        if mask_dilate_uniform > 0:
            self.voxel = self.dilate_uniform(mask_dilate_uniform)
        if mask_dilate_top > 0:
            self.voxel = self.dilate_top(mask_dilate_top)

    def get_bbox(self):
        """
        Get the **loose** 3D bbox for the object
        """
        return self.bbox_min, self.bbox_max
    
    def get_tight_bbox(self):
        """
        Get the **tight** 3D bbox that perfectly enclose the object
        """
        if self.tight_bbox is not None:
            return self.tight_bbox
        else:
            raise NotImplementedError("Need tight bbox computation")
        
    def get_pose_change(self):
        """
        Get the 6DoF pose change of the object
        """
        if self.pose_change is None:
            return torch.eye(4, device=self.voxel.device)
        return self.pose_change
    
    def set_pose_change(self, pose_change):
        """
        Set the 6DoF pose change of the object

        Args:
            pose_change (4x4 tensor): 6DoF pose change
        """
        if self.pose_change is None:
            self.pose_change = None
        else:
            self.pose_change = pose_change

    def get_all_corners(self):
        """
        Get all corner of the loose bbox

        Returns:
            corners (8x3 tensor): 8 corners of the bbox
        """
        combined = torch.stack([self.bbox_min, self.bbox_max], dim=1)
        corners = torch.tensor(list(itertools.product(*combined)))
        return corners.to(self.voxel.device)
    
    def get_obj_coords(self):
        """
        Get the coordinates of the obj occupied voxels

        Returns:
            coords (N, 3 tensor): Occupied voxel coordinates
        """
        x_min, y_min, z_min = self.bbox_min.cpu().numpy().ravel()
        x_max, y_max, z_max = self.bbox_max.cpu().numpy().ravel()
        x_step = (x_max - x_min) / (self.voxel.shape[0] - 1)
        y_step = (y_max - y_min) / (self.voxel.shape[1] - 1)
        z_step = (z_max - z_min) / (self.voxel.shape[2] - 1)
        obj_coords = torch.nonzero(self.voxel)
        x_coords = x_min + obj_coords[:, 0] * x_step
        y_coords = y_min + obj_coords[:, 1] * y_step
        z_coords = z_min + obj_coords[:, 2] * z_step
        coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        return coords

    def dilate_uniform(self, kernel_size=1):
        """
        Uniformly dilate the binary voxel grid

        Args:
            kernel_size (int): Size of the dilation kernel
        
        Returns:
            dilated_voxel (N, M, K tensor): Dilated voxel grid
        """
        voxel = self.voxel.cpu().numpy()
        voxel = binary_dilation(
            voxel, iterations=kernel_size,
            structure=generate_binary_structure(3, 3)
        )
        voxel = torch.from_numpy(voxel).to(self.voxel.device)
        return voxel

    def dilate_top(self, kernel_size=3):
        """
        Non-uniformly dilate the binary voxel grid, more on top, none at bottom

        Args:
            kernel_size (int): Size of the dilation kernel
        
        Returns:
            dilated_voxel (N, M, K tensor): Dilated voxel grid
        """
        voxel = self.voxel.cpu().numpy()
        struct = generate_binary_structure(3, 1)
        struct[..., 0:-1] = False
        struct[..., -1] = True
        struct[1, 1, 1] = True
        voxel = binary_dilation(
            voxel, iterations=kernel_size, structure=struct
        )
        voxel = torch.from_numpy(voxel).to(self.voxel.device)
        return voxel
    
    def dilate_dir(self, dir, kernel_size=1):
        """
        Dilate the binary voxel grid in a specific direction
        We find the axis direction (+/-x, +/-y, +/-z) that is closest to the
        input direction and dilate the mask in that direction

        Args:
            dir (3-tuple): Direction to dilate the mask
            kernel_size (int): Size of the dilation kernel

        Returns:
            dilated_voxel (N, M, K tensor): Dilated voxel grid
        """
        dir = np.array(dir).ravel()
        assert len(dir) == 3, "Direction must have 3 components"
        # identify the axis with the closest axis direction
        axis = np.argmax(np.abs(dir))
        direction = np.sign(dir[axis])
        # construct the structuring element
        se = np.zeros((3, 3, 3), dtype=bool)
        if direction > 0:
            if axis == 0:  # x-axis
                se[1:, 1, 1] = True
            elif axis == 1:  # y-axis
                se[1, 1:, 1] = True
            elif axis == 2:  # z-axis
                se[1, 1, 1:] = True
        else:
            if axis == 0:  # x-axis
                se[:-1, 1, 1] = True
            elif axis == 1:  # y-axis
                se[1, :-1, 1] = True
            elif axis == 2:  # z-axis
                se[1, 1, :-1] = True
        # Dilate along the closest axis direction
        voxel = self.voxel.cpu().numpy()
        voxel = binary_dilation(voxel, structure=se, iterations=kernel_size)
        voxel = torch.from_numpy(voxel).to(self.voxel.device)
        return voxel

    def fill_under(self):
        """
        Dilate the binary voxel grid till the bottom of the object bbox
        
        Returns:
            dilated_voxel (N, M, K tensor): Dilated voxel grid
        """
        voxel = self.voxel
        flip_voxel = torch.flip(voxel, dims=(2,))
        under_object_mask = torch.cumsum(flip_voxel, dim=2) > 0
        under_object_mask = torch.flip(under_object_mask, dims=(2, ))
        voxel[under_object_mask] = True
        return voxel

    def query(self, points, dilate=None, voxel=None):
        """
        Query the object 3D seg at points to determine whether they are inside

        Args:
            points (..., 3): 3D query points
            dilate (bool): Whether to use dilated voxel grid for query
            voxel (N, M, K tensor or None): another voxel grid to query

        Returns:
            inside (...,): Whether the points are inside the object
        """
        assert points.shape[-1] == 3
        in_bbox = (
            (points >= self.bbox_min) & (points <= self.bbox_max)
        ).all(dim=-1)
        points_in_bbox = points[in_bbox]
        if points_in_bbox.numel() == 0:
            return torch.zeros_like(in_bbox).bool()
        # Normalize the points to the bounding box
        points_in_bbox = (points_in_bbox - self.bbox_min) / self.dims * 2 - 1
        # Convert to voxel indices
        grid = points_in_bbox.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        grid = torch.flip(grid, dims=(-1,)) # grid_sample 3D uses zyx order
        # Trilinear interpolation to extract occupancy values
        voxel = self.voxel if voxel is None else voxel
        occupancy = torch.nn.functional.grid_sample(
            voxel.float().unsqueeze(0).unsqueeze(0), grid.float(),
            align_corners=True
        ).squeeze()
        # Occupancy check
        inside = torch.zeros_like(in_bbox).to(points.device)
        inside[in_bbox] = occupancy > 0.1
        return inside
    
    def sample_random_points(self, num_points):
        """
        Sample random 3D points within the object's occupied voxels.

        Args:
            num_points (int): Number of random points to sample

        Returns:
            points (num_points x 3 tensor): Sampled points in world coordinates
        """
        # Get all occupied voxel indices
        occupied = torch.nonzero(self.voxel, as_tuple=False)

        if occupied.shape[0] == 0:
            raise ValueError("No occupied voxels found to sample points from.")

        # If requesting more points than available voxels, sample with replacement
        replace = occupied.shape[0] < num_points

        indices = torch.randint(
            0, occupied.shape[0],
            (num_points,), device=occupied.device,
            dtype=torch.long
        )

        voxel_samples = occupied[indices]

        # Convert voxel indices to real world coordinates
        x_min, y_min, z_min = self.bbox_min.cpu().numpy().ravel()
        x_max, y_max, z_max = self.bbox_max.cpu().numpy().ravel()

        x_step = (x_max - x_min) / (self.voxel.shape[0] - 1)
        y_step = (y_max - y_min) / (self.voxel.shape[1] - 1)
        z_step = (z_max - z_min) / (self.voxel.shape[2] - 1)

        # Add random offset within each voxel cell for uniform sampling
        offsets = torch.rand_like(voxel_samples, dtype=torch.float)

        x_coords = x_min + (voxel_samples[:, 0].float() + offsets[:, 0]) * x_step
        y_coords = y_min + (voxel_samples[:, 1].float() + offsets[:, 1]) * y_step
        z_coords = z_min + (voxel_samples[:, 2].float() + offsets[:, 2]) * z_step

        points = torch.stack([x_coords, y_coords, z_coords], dim=-1).to(self.voxel.device)

        return points

    
    def query_new(self, points, voxel=None):
        """
        Query the binary voxel w/ points to see if they are in obj new 3D mask

        Args:
            points (..., 3): 3D query points
            dilate (bool): Whether to use dilated voxel grid for query

        Returns:
            inside (...): Whether the points are inside the object's new mask
        """
        assert points.shape[-1] == 3
        # Transform points back to object's old 3D mask
        if self.pose_change is None:
            pose_change_inv = torch.eye(4, device=points.device)
        else:
            pose_change_inv = torch.inverse(self.pose_change)
        points_trans = torch.einsum(
            "ij,...j->...i", pose_change_inv[:3, :3], points
        ) + pose_change_inv[:3, 3]
        in_bbox = (
            (points_trans >= self.bbox_min) & (points_trans <= self.bbox_max)
        ).all(dim=-1)
        points_in_bbox = points_trans[in_bbox]
        if points_in_bbox.numel() == 0:
            return torch.zeros_like(in_bbox).bool()
        # Normalize the points to the bounding box
        points_in_bbox = (points_in_bbox - self.bbox_min) / self.dims * 2 - 1
        # Convert to voxel indices
        grid = points_in_bbox.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        grid = torch.flip(grid, dims=(-1,))
        # Trilinear interpolation to extract occupancy values
        voxel = self.voxel if voxel is None else voxel
        occupancy = torch.nn.functional.grid_sample(
            voxel.float().unsqueeze(0).unsqueeze(0), grid.float(),
            align_corners=True
        ).squeeze()
        # Occupancy check
        inside = torch.zeros_like(in_bbox).to(points.device).bool()
        inside[in_bbox] = occupancy > 0.5
        return inside

    # def project(self, poses, Ks, dist_coeffs, H, W, kernel_size=0):
    #     """
    #     Project the object points to camera views

    #     Args:
    #         poses (Nx4x4 tensor): Camera poses
    #         Ks (Nx3x3 tensor): Camera intrinsics
    #         dist_coeffs (Nx4 tensor): Camera distortion coefficients
    #         H (int): Image height
    #         W (int): Image width
    #         kernel_size (int): Kernel size for closing the mask
        
    #     Returns:
    #         masks (Nx1xHxW tensor): Projected masks
    #     """
    #     obj_pts = self.get_obj_coords()
    #     verts_proj, proj_valid = project_points(
    #         obj_pts, poses, Ks, dist_coeffs, H, W
    #     )
    #     masks = points2D_to_point_masks(
    #         verts_proj, proj_valid, H, W, kernel_size
    #     )
    #     return masks

    # def project_new(self, poses, Ks, dist_coeffs, H, W, kernel_size=0):
    #     """
    #     Project reconfigured objects' points to camera views

    #     Args:
    #         poses (Nx4x4 tensor): Camera poses
    #         Ks (Nx3x3 tensor): Camera intrinsics
    #         dist_coeffs (Nx4 tensor): Camera distortion coefficients
    #         H (int): Image height
    #         W (int): Image width
    #         kernel_size (int): Kernel size for closing the mask
        
    #     Returns:
    #         masks (Nx1xHxW tensor): Projected 2D masks
    #     """
    #     obj_pts = self.get_obj_coords()
    #     if self.pose_change is None:
    #         pose_change = torch.eye(4, device=poses.device)
    #     else:
    #         pose_change = self.pose_change
    #     obj_pts_moved = (
    #         pose_change[:3, :3] @ obj_pts.T + pose_change[:3, 3:]
    #     ).T.reshape(-1, 3)
    #     obj_proj, in_img = project_points(
    #         obj_pts_moved, poses, Ks, dist_coeffs, H, W
    #     )
    #     masks = points2D_to_point_masks(obj_proj, in_img, H, W, kernel_size)
    #     return masks

    def save(self, output):
        """
        Save the input of object 3D segmentation

        Args:
            output (str): Path to save the 3D object mask
        """
        if self.pose_change is not None:
            pose_change = self.pose_change.cpu()
        else:
            pose_change = None
        data_to_save = {
            'bbox_min': self.bbox_min.cpu(), 'bbox_max': self.bbox_max.cpu(),
            'voxel': self.voxel_original.cpu(),
            'pose_change': pose_change,
            'mask_dilate_uniform': self.mask_dilate_uniform,
            'mask_dilate_top': self.mask_dilate_top
        }
        if self.tight_bbox is not None:
            data_to_save['tight_bbox'] = self.tight_bbox
        else:
            data_to_save['tight_bbox'] = None
        torch.save(data_to_save, f"{output}")

    @classmethod
    def read_from_file(cls, file_path, device='cuda'):
        """
        Create an Object3DSeg instance from a file

        Args:
            file_path (str): Path to mask_3D.pt file
        
        Returns:
            Object3DSeg: An instance of the Object3DSeg class
        """
        # Load the data from the file
        data = torch.load(file_path)
        # Create a new Object3DSeg instance with the loaded data
        if data['tight_bbox'] is not None:
            tight_bbox = data['tight_bbox']
        else:
            tight_bbox = None
        obj = cls(
            bbox_min=data['bbox_min'].tolist(),
            bbox_max=data['bbox_max'].tolist(),
            voxel=data['voxel'].to(device),
            tight_bbox=tight_bbox,
            pose_change=data['pose_change'].to(device),
            mask_dilate_uniform=data['mask_dilate_uniform'],
            mask_dilate_top=data['mask_dilate_top']
        )
        print(f"Succesfully loaded the scene change estimates")
        print(f"Pose change:\n {obj.pose_change}")
        return obj
    
    def visualize(self, output_dir):
        voxel = self.voxel.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xmin, ymin, zmin = self.bbox_min.cpu().numpy()
        xmax, ymax, zmax = self.bbox_max.cpu().numpy()
        # Define the 12 edges of the bounding box
        edges = [
            [(xmin, ymin, zmin), (xmax, ymin, zmin)],
            [(xmax, ymin, zmin), (xmax, ymax, zmin)],
            [(xmax, ymax, zmin), (xmin, ymax, zmin)],
            [(xmin, ymax, zmin), (xmin, ymin, zmin)],
            [(xmin, ymin, zmax), (xmax, ymin, zmax)],
            [(xmax, ymin, zmax), (xmax, ymax, zmax)],
            [(xmax, ymax, zmax), (xmin, ymax, zmax)],
            [(xmin, ymax, zmax), (xmin, ymin, zmax)],
            [(xmin, ymin, zmin), (xmin, ymin, zmax)],
            [(xmax, ymin, zmin), (xmax, ymin, zmax)],
            [(xmax, ymax, zmin), (xmax, ymax, zmax)],
            [(xmin, ymax, zmin), (xmin, ymax, zmax)]
        ]
        for edge in edges:
            ax.plot3D(*zip(*edge), color='r')
        # Create a mesh grid for the voxel positions
        x = np.linspace(xmin, xmax, voxel.shape[0]+1)
        y = np.linspace(ymin, ymax, voxel.shape[1]+1)
        z = np.linspace(zmin, zmax, voxel.shape[2]+1)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        verts, faces, _, _ = marching_cubes(voxel, 0.5)
        x_scale = (xmax - xmin) / voxel.shape[0]
        y_scale = (ymax - ymin) / voxel.shape[1]
        z_scale = (zmax - zmin) / voxel.shape[2]
        verts[:, 0] = verts[:, 0] * x_scale + xmin
        verts[:, 1] = verts[:, 1] * y_scale + ymin
        verts[:, 2] = verts[:, 2] * z_scale + zmin
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('b')
        ax.add_collection3d(mesh)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #plt.savefig(f"{output_dir}/occ_grid.png")
        ax.set_box_aspect([xmax - xmin, ymax - ymin, zmax - zmin])

        plt.show()



    def visualize_ply(self, output_dir):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes
        import os
        import numpy as np
        import open3d as o3d

        voxel = self.occ_grid.cpu().numpy().astype(np.float32)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xmin, ymin, zmin = self.bbox_min.cpu().numpy()
        xmax, ymax, zmax = self.bbox_max.cpu().numpy()

        # Bounding box edges
        edges = [...]
        for edge in edges:
            ax.plot3D(*zip(*edge), color='r')

        # Marching cubes
        verts, faces, _, _ = marching_cubes(voxel, 0.5)

        x_scale = (xmax - xmin) / voxel.shape[0]
        y_scale = (ymax - ymin) / voxel.shape[1]
        z_scale = (zmax - zmin) / voxel.shape[2]
        verts[:, 0] = verts[:, 0] * x_scale + xmin
        verts[:, 1] = verts[:, 1] * y_scale + ymin
        verts[:, 2] = verts[:, 2] * z_scale + zmin

        # Plot with matplotlib
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_edgecolor('b')
        ax.add_collection3d(mesh)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([xmax - xmin, ymax - ymin, zmax - zmin])

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/occ_grid.png", dpi=300)
        plt.show()

        # Also save as .ply
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.compute_vertex_normals()
        o3d.io.write_triangle_mesh(f"{output_dir}/occ_mesh_debug.ply", mesh_o3d)





# class Obj3DFeats:
#     """
#     Object's multiview SuperPoint features
#     """
#     def __init__(self, feats=[], pts3D=[]):
#         """
#         Args:
#             feats (dict): SuperPoint feature dict
#         """
#         self.feats = feats
#         self.pts3D = pts3D
#         assert len(feats) == len(pts3D)
#         assert all(
#             [feats['keypoints'].shape[1] == pts.shape[0] 
#             for feats, pts in zip(feats, pts3D)]
#         )
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )
    
#     def add_feats(self, feats, pts3D):
#         """
#         Add features to the object

#         Args:
#             feats (dict): SuperPoint feature dict from a view
#             pts3D (Nx3 tensor): 3D points corresponding to the features
#         """
#         assert feats['keypoints'].shape[1] == pts3D.shape[0]
#         self.feats.append(feats)
#         self.pts3D.append(pts3D)

#     def match(self, feats, matcher=None):
#         """
#         Match features with the object features

#         Args:
#             feats (dict): SuperPoint feature dict for 2D image points
#             matcher (LightGlue.Matcher): Feature matcher
        
#         Returns:
#             matched3D (Nx3): Matched 3D points
#             matched2D (Nx2): Matched 2D points
#         """
#         assert len(self.feats) > 0, "No features to match"
#         # TODO: have to take matcher as input
#         if matcher is None:
#             from lightglue import LightGlue            
#             matcher = LightGlue(features='superpoint').eval().to(self.device)
#         matched_pts3D, matches = [], []
#         for obj_feats, obj_pts3D in zip(self.feats, self.pts3D): 
#             mm = matcher({'image0': obj_feats, 'image1': feats})
#             _, _, mm = [rbd(x) for x in [obj_feats, feats, mm]]
#             # Uncomment to filter low-confidence matches
#             # mm["matches"] = mm["matches"][mm["scores"] > 0.4, :]
#             matched3D = obj_pts3D[mm["matches"][:, 0]]
#             matches.append(mm["matches"][:, 1])
#             matched_pts3D.append(matched3D)
#         matches = torch.cat(matches, dim=0)
#         matched_pts3D = torch.cat(matched_pts3D, dim=0)
#         # TODO: handle duplicate matches by comparing 
#         matched_pts2D = feats["keypoints"][0][matches]
#         return matched_pts2D, matched_pts3D
    
#     def PnP(self, feats, K, H, W, matcher=None, verbose=False):
#         """
#         Solve PnP problem to estimate cam-to-obj pose

#         Args:
#             feats (dict): SuperPoint feature dict
#             K (3x3): Camera intrinsics
#             matcher (LightGlue.Matcher): Feature matcher
        
#         Returns:
#             pose (4x4 tensor): Camera-to-obj pose
#             num_inliers (int): Number of inliers
#             num_matches (int): Number of matches
#         """

#         assert K.shape == (3, 3)
#         matched_pts2D, matched_pts3D = self.match(feats, matcher)
#         matched_pts2D = matched_pts2D.cpu().numpy()
#         matched_pts3D = matched_pts3D.cpu().numpy()
#         if matched_pts3D.shape[0] < 4:
#             print("Warn: Not enough points for PnP")
#             return None, 0.0, 0
#         pycolmap_cam = pycolmap.Camera(
#             model='OPENCV', width=W, height=H,
#             params=[K[0, 0], K[1, 1], K[0, -1], K[1, -1], 0, 0, 0, 0]
#         )
#         ret = pycolmap.absolute_pose_estimation(
#             matched_pts2D, matched_pts3D, pycolmap_cam,
#             estimation_options={'ransac': {'max_error': 12.0}}, 
#             refinement_options={'print_summary': verbose}
#         )
#         if not ret['success']:
#             print("Warn: PnP failed!!")
#             return None, 0.0, 0
#         R_mat = torch.tensor(Quaternion(*ret['qvec']).rotation_matrix)
#         tvec = torch.tensor(ret['tvec'])
#         pose = torch.eye(4, device=K.device)
#         pose[:3, :3], pose[:3, 3] = R_mat, tvec
#         pose = pose.inverse()
#         if verbose:
#             print(
#                 f"Number of inliers: {ret['num_inliers']}/{len(matched_pts2D)}"
#             )
#             print(f"pose est:\n {pose}")
#         return pose, ret["num_inliers"], len(matched_pts2D)