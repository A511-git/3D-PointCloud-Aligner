"""
Walls Aligner (Manhattan Yaw Alignment).
Rotates indoor point clouds around the Z-axis so dominant architectural walls
run parallel to the X and Y coordinate axes.
Uses 4-fold periodic voting on horizontal surface normals,
natively supporting standard rectangular rooms as well as L-shaped and multi-corner layouts.
"""

from typing import Tuple
import numpy as np
import open3d as o3d


class WallsAligner:
    """
    Estimates and applies optimal yaw rotation around the +Z axis to align walls to the X/Y axes.
    """

    @staticmethod
    def find_manhattan_yaw_angle(pcd: o3d.geometry.PointCloud) -> Tuple[float, float]:
        """
        Finds dominant horizontal wall direction using 4-fold periodic voting on 2D wall normals.
        Assumes pcd is already oriented with +Z as Up.

        Returns:
            (best_alpha_radians, score)
        """
        sample = pcd.voxel_down_sample(0.04)
        if len(sample.points) < 500:
            sample = pcd

        sample.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
        normals = np.asarray(sample.normals)
        pts = np.asarray(sample.points)

        z_min, z_max = np.percentile(pts[:, 2], 5), np.percentile(pts[:, 2], 95)
        # Select vertical wall points (normals almost perpendicular to +Z)
        wall_mask = (np.abs(normals[:, 2]) < 0.25) & (pts[:, 2] > z_min + 0.2) & (pts[:, 2] < z_max - 0.2)
        wall_normals = normals[wall_mask]

        if len(wall_normals) < 50:
            # Fallback to 2D covariance of middle points
            wall_pts = pts[(pts[:, 2] > z_min + 0.2) & (pts[:, 2] < z_max - 0.2)][:, :2]
            if len(wall_pts) < 10:
                return 0.0, 0.0
            cov = np.cov(wall_pts, rowvar=False)
            _, vecs = np.linalg.eigh(cov)
            angle = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
            return angle, 1.0

        thetas = np.arctan2(wall_normals[:, 1], wall_normals[:, 0])

        # 4-Fold Periodic Voting:
        # Walls at 0°, 90°, 180°, and 270° all align when multiplied by 4:
        # cos(4*(theta - alpha)) reaches maximum when walls match alpha
        candidate_alphas = np.linspace(0, np.pi / 2, num=180, endpoint=False)
        diffs = 4.0 * (thetas[:, None] - candidate_alphas[None, :])
        scores = np.sum(np.cos(diffs), axis=0)

        best_idx = np.argmax(scores)
        best_alpha = float(candidate_alphas[best_idx])
        return best_alpha, float(scores[best_idx])

    @classmethod
    def align_walls(cls, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, float, np.ndarray]:
        """
        Aligns dominant walls of the point cloud with the X and Y axes.

        Returns:
            (aligned_pcd, yaw_angle_rad, R_matrix)
        """
        yaw_angle, score = cls.find_manhattan_yaw_angle(pcd)
        R_yaw = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, -yaw_angle))
        pcd.rotate(R_yaw, center=(0, 0, 0))
        return pcd, yaw_angle, R_yaw
