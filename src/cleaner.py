"""
Cleaner and Denoising Filter.
Filters stray sensor noise, reflections, and floating points via statistical outlier removal,
and unifies spatial density via voxel grid downsampling.
"""

from typing import Tuple, Optional
import numpy as np
import open3d as o3d


class Cleaner:
    """
    Cleans and standardizes point cloud density.
    """

    @staticmethod
    def clean(
        pcd: o3d.geometry.PointCloud,
        remove_outliers: bool = True,
        nb_neighbors: int = 25,
        std_ratio: float = 2.0,
        voxel_size: float = 0.03
    ) -> Tuple[o3d.geometry.PointCloud, int, int]:
        """
        Filters outliers and downsamples the point cloud.

        Args:
            pcd: Open3D PointCloud.
            remove_outliers: If True, applies statistical outlier removal.
            nb_neighbors: Number of neighbors for distance calculation.
            std_ratio: Standard deviation threshold multiplier.
            voxel_size: Voxel cube edge size in meters (0 to disable).

        Returns:
            (cleaned_pcd, initial_point_count, final_point_count)
        """
        initial_count = len(pcd.points)

        # Statistical outlier removal
        if remove_outliers and initial_count > nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )

        # Voxel downsampling
        if voxel_size and voxel_size > 0.0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        final_count = len(pcd.points)
        return pcd, initial_count, final_count
