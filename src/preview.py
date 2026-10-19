"""
Diagnostic Preview Generator.
Renders a 3-panel orthogonal projection (Top X-Y, Front X-Z, Side Y-Z) PNG plot
for instant visual inspection of room alignment, wall squareness, and floor grounding.
"""

import os
from typing import Optional
import numpy as np
import open3d as o3d


class DiagnosticPreview:
    """
    Creates high-resolution 2D orthogonal diagnostic plots for 3D point clouds.
    """

    @staticmethod
    def generate_preview(
        pcd: o3d.geometry.PointCloud,
        output_png_path: str,
        max_points_display: int = 50000,
        dpi: int = 180
    ) -> bool:
        """
        Renders and saves a 3-view orthogonal diagnostic PNG.

        Returns:
            True if preview was saved successfully, False otherwise.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                return False

            if len(pts) > max_points_display:
                indices = np.random.choice(len(pts), max_points_display, replace=False)
                sample_pts = pts[indices]
            else:
                sample_pts = pts

            fig, (ax_top, ax_front, ax_side) = plt.subplots(1, 3, figsize=(15, 5))

            # Top View (X-Y)
            ax_top.scatter(sample_pts[:, 0], sample_pts[:, 1], s=0.3, c="#1d2939", alpha=0.4)
            ax_top.set_title(
                f"Top View (X-Y Plane)\nSpan: {pts[:, 0].ptp():.2f}m x {pts[:, 1].ptp():.2f}m",
                fontsize=10, weight="bold"
            )
            ax_top.set_xlabel("X (m)")
            ax_top.set_ylabel("Y (m)")
            ax_top.set_aspect("equal")
            ax_top.grid(True, linestyle="--", alpha=0.3)

            # Front View (X-Z)
            ax_front.scatter(sample_pts[:, 0], sample_pts[:, 2], s=0.3, c="#0ba5ec", alpha=0.4)
            ax_front.set_title(
                f"Front View (X-Z Plane)\nHeight: {pts[:, 2].ptp():.2f}m",
                fontsize=10, weight="bold"
            )
            ax_front.set_xlabel("X (m)")
            ax_front.set_ylabel("Z (m) [Up]")
            ax_front.set_aspect("equal")
            ax_front.grid(True, linestyle="--", alpha=0.3)

            # Side View (Y-Z)
            floor_z = float(np.percentile(pts[:, 2], 2))
            ax_side.scatter(sample_pts[:, 1], sample_pts[:, 2], s=0.3, c="#7a5af8", alpha=0.4)
            ax_side.set_title(
                f"Side View (Y-Z Plane)\nFloor at Z={floor_z:.2f}m",
                fontsize=10, weight="bold"
            )
            ax_side.set_xlabel("Y (m)")
            ax_side.set_ylabel("Z (m) [Up]")
            ax_side.set_aspect("equal")
            ax_side.grid(True, linestyle="--", alpha=0.3)

            fig.suptitle("Point Cloud Alignment Diagnostic Preview", fontsize=13, weight="bold")
            plt.tight_layout()
            os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
            plt.savefig(output_png_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            return True
        except Exception as err:
            print(f"[DiagnosticPreview Warning] Could not render preview plot: {err}")
            return False
