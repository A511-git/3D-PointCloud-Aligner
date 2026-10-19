"""
Scale Calibrator.
Detects vertical room clearance (floor-to-ceiling) and calibrates point clouds to true metric scale (1 unit = 1 meter).
Protects already metric scans from accidental distortion while safely recovering normalized unit boxes (<0.8m)
or millimeter scans (>50m).
"""

from typing import Tuple
import numpy as np
import open3d as o3d


class ScaleCalibrator:
    """
    Measures floor-to-ceiling height and rescales the scene to realistic metric units.
    """

    @staticmethod
    def detect_room_height(pcd: o3d.geometry.PointCloud) -> Tuple[float, float, float]:
        """
        Computes floor level (2nd percentile), ceiling level (98th percentile), and vertical clearance.
        Percentiles are used instead of min/max to prevent stray floating points from distorting measurements.

        Returns:
            (height, z_floor, z_ceiling)
        """
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return 0.0, 0.0, 0.0
        z_floor = float(np.percentile(pts[:, 2], 2))
        z_ceil = float(np.percentile(pts[:, 2], 98))
        height = max(0.0, z_ceil - z_floor)
        return height, z_floor, z_ceil

    @classmethod
    def calibrate_scale(
        cls,
        pcd: o3d.geometry.PointCloud,
        target_height: float = 2.65,
        force_scale: bool = False,
        min_metric_height: float = 0.8,
        max_metric_height: float = 50.0
    ) -> Tuple[o3d.geometry.PointCloud, float, float]:
        """
        Calibrates the scale of the point cloud.

        Args:
            pcd: Point cloud oriented with +Z as Up.
            target_height: Standard room height in meters (typically 2.5m - 2.8m).
            force_scale: If True, always forces the detected height to target_height.
            min_metric_height: Heights below this are diagnosed as normalized unit boxes.
            max_metric_height: Heights above this are diagnosed as millimeter/centimeter scale.

        Returns:
            (scaled_pcd, scale_factor, detected_height)
        """
        detected_height, z_floor, z_ceil = cls.detect_room_height(pcd)
        if detected_height <= 0:
            return pcd, 1.0, 0.0

        should_rescale = False
        if force_scale and target_height > 0:
            should_rescale = True
        elif target_height > 0:
            if detected_height < min_metric_height:
                # Normalized unit bounding box (e.g., from monocular SLAM or normalized datasets)
                should_rescale = True
            elif detected_height > max_metric_height:
                # Millimeter or centimeter scale
                should_rescale = True

        if should_rescale:
            scale_factor = target_height / detected_height
            pcd.scale(scale_factor, center=(0, 0, 0))
            return pcd, scale_factor, detected_height
        else:
            return pcd, 1.0, detected_height
