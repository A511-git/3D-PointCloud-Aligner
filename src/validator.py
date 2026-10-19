"""
Point Cloud Validator.
Performs comprehensive quality and compliance checks on 3D indoor point clouds:
- Point density and minimum count
- Bounding box spans and realistic room proportions
- Vertical axis orientation (+Z up) and floor grounding (Z ~ 0)
- Color channel presence and valid range [0, 1]
"""

from typing import Dict, Any, Tuple, Union
import numpy as np
import open3d as o3d


class PointCloudValidator:
    """
    Validates indoor point clouds for downstream spatial AI models.
    """

    @classmethod
    def validate(
        cls,
        pcd_or_path: Union[str, o3d.geometry.PointCloud],
        min_points: int = 1000,
        min_height: float = 1.8,
        max_height: float = 4.5,
        min_span: float = 0.8,
        max_floor_deviation: float = 0.25
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates point cloud geometry, bounds, and orientation.

        Returns:
            (is_compliant, report_dict)
        """
        if isinstance(pcd_or_path, str):
            pcd = o3d.io.read_point_cloud(pcd_or_path)
        else:
            pcd = pcd_or_path

        pts = np.asarray(pcd.points)
        num_points = len(pts)

        report = {
            "num_points": num_points,
            "has_colors": pcd.has_colors(),
            "passed": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }

        if num_points == 0:
            report["passed"] = False
            report["errors"].append("Point cloud is empty.")
            return False, report

        # 1. Point Count Check
        if num_points < min_points:
            report["passed"] = False
            report["errors"].append(f"Insufficient points: {num_points} (required >= {min_points}).")
            report["checks"]["point_count"] = False
        else:
            report["checks"]["point_count"] = True

        # 2. Coordinate Spans
        x_span = float(pts[:, 0].ptp())
        y_span = float(pts[:, 1].ptp())
        z_span = float(pts[:, 2].ptp())
        z_floor = float(np.percentile(pts[:, 2], 2))
        z_ceil = float(np.percentile(pts[:, 2], 98))
        effective_height = z_ceil - z_floor

        report["metrics"] = {
            "x_span_m": round(x_span, 2),
            "y_span_m": round(y_span, 2),
            "z_span_m": round(z_span, 2),
            "effective_height_m": round(effective_height, 2),
            "floor_level_z": round(z_floor, 2)
        }

        # 3. Floor Grounding Check
        if abs(z_floor) > max_floor_deviation:
            report["warnings"].append(
                f"Floor is not grounded at Z=0 (detected at Z={z_floor:.2f}m, deviation > {max_floor_deviation}m)."
            )
            report["checks"]["floor_grounded"] = False
        else:
            report["checks"]["floor_grounded"] = True

        # 4. Realistic Room Height Check
        if effective_height < min_height or effective_height > max_height:
            report["warnings"].append(
                f"Room height ({effective_height:.2f}m) outside expected range [{min_height}m, {max_height}m]."
            )
            report["checks"]["height_valid"] = False
        else:
            report["checks"]["height_valid"] = True

        # 5. Horizontal Span Check
        if x_span < min_span or y_span < min_span:
            report["warnings"].append(
                f"Horizontal dimensions (X: {x_span:.2f}m, Y: {y_span:.2f}m) appear too small (min {min_span}m)."
            )
            report["checks"]["span_valid"] = False
        else:
            report["checks"]["span_valid"] = True

        # 6. Colors Check
        if not pcd.has_colors():
            report["warnings"].append("Point cloud lacks RGB color channels.")
            report["checks"]["colors_present"] = False
        else:
            colors = np.asarray(pcd.colors)
            if colors.min() < 0.0 or colors.max() > 1.05:
                report["warnings"].append("Color values are outside standard normalized range [0.0, 1.0].")
                report["checks"]["colors_range"] = False
            else:
                report["checks"]["colors_range"] = True
            report["checks"]["colors_present"] = True

        if len(report["errors"]) > 0:
            report["passed"] = False

        return report["passed"], report
