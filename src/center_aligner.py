"""
Center and Ground Aligner.
Translates point clouds so that:
1. Floor level is grounded precisely at Z = 0.0.
2. Horizontal bounds (X and Y) are centered around origin (0.0, 0.0).
"""

from typing import Tuple
import numpy as np
import open3d as o3d


class CenterAligner:
    """
    Applies translation vectors to center the room horizontally and ground the floor at Z = 0.
    """

    @staticmethod
    def center_and_ground(
        pcd: o3d.geometry.PointCloud,
        center_xy: bool = True,
        ground_z: bool = True,
        use_median: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Translates the point cloud to the origin coordinate frame.

        Args:
            pcd: Point cloud oriented with +Z as Up.
            center_xy: Center horizontal X and Y at origin.
            ground_z: Shift floor (2nd percentile Z) to Z = 0.
            use_median: Use median X, Y (more robust to outlier bias than center of bounding box).

        Returns:
            (centered_pcd, translation_vector)
        """
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return pcd, np.zeros(3)

        tx = 0.0
        ty = 0.0
        tz = 0.0

        if center_xy:
            if use_median:
                tx = -float(np.median(pts[:, 0]))
                ty = -float(np.median(pts[:, 1]))
            else:
                tx = -float((pts[:, 0].max() + pts[:, 0].min()) / 2.0)
                ty = -float((pts[:, 1].max() + pts[:, 1].min()) / 2.0)

        if ground_z:
            floor_z = float(np.percentile(pts[:, 2], 2))
            tz = -floor_z

        trans = np.array([tx, ty, tz], dtype=np.float64)
        pcd.translate(trans)
        return pcd, trans
