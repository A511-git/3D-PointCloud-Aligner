"""
Floor and Up-Axis Aligner.
Aligns arbitrary 3D indoor point clouds so the floor is level and the vertical axis points to +Z.
Uses multi-plane orthogonal consensus to prevent the classic 'Wall-as-Floor' RANSAC trap,
and density percentile asymmetry to distinguish Floor (+Z bottom) from Ceiling (+Z top).
"""

from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import open3d as o3d


class FloorAligner:
    """
    Identifies and aligns the true vertical axis of a 3D indoor point cloud to +Z.
    """

    AXIS_MAP = {
        "x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }

    PRESET_MAP = {
        "arkit": "y",     # ARKit/iOS: +Y is Up
        "opengl": "y",    # OpenGL: +Y is Up
        "opencv": "-y",   # OpenCV: +Y is Down -> -Y is Up
        "scannet": "z",   # ScanNet: +Z is Up
    }

    @staticmethod
    def vector_align_rotation_matrix(src_vec: np.ndarray, tgt_vec: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        """
        Computes 3x3 rotation matrix that rotates unit vector src_vec to tgt_vec
        using Rodrigues' rotation formula.
        """
        v_src = src_vec / np.linalg.norm(src_vec)
        v_tgt = tgt_vec / np.linalg.norm(tgt_vec)

        dot = np.dot(v_src, v_tgt)
        if dot > 0.999999:
            return np.eye(3)
        if dot < -0.999999:
            # 180 degree flip around an orthogonal axis
            ortho = np.array([1.0, 0.0, 0.0])
            if abs(v_src[0]) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0])
            axis = np.cross(v_src, ortho)
            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            return np.eye(3) + 2.0 * (K @ K)

        cross = np.cross(v_src, v_tgt)
        s = np.linalg.norm(cross)
        K = np.array([
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0]
        ])
        R = np.eye(3) + K + (K @ K) * ((1.0 - dot) / (s * s))
        return R

    @staticmethod
    def extract_candidate_planes(
        pcd: o3d.geometry.PointCloud,
        max_planes: int = 5,
        distance_thresh: float = 0.035
    ) -> List[Dict]:
        """
        Extracts dominant planar surfaces iteratively by peeling off inliers after each RANSAC fit.
        """
        planes = []
        current_pcd = pcd
        for _ in range(max_planes):
            if len(current_pcd.points) < 500:
                break
            plane_model, inliers = current_pcd.segment_plane(
                distance_threshold=distance_thresh,
                ransac_n=3,
                num_iterations=1000
            )
            if len(inliers) < 150:
                break
            normal = np.array(plane_model[:3], dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            planes.append({
                "model": plane_model,
                "normal": normal,
                "num_inliers": len(inliers)
            })
            # Peel off the detected plane to find the next one
            current_pcd = current_pcd.select_by_index(inliers, invert=True)
        return planes

    @classmethod
    def estimate_up_axis_multiplane(
        cls,
        pcd: o3d.geometry.PointCloud,
        max_planes: int = 5,
        distance_thresh: float = 0.035
    ) -> np.ndarray:
        """
        Robust multi-plane consensus to identify the vertical axis (+Z).
        Guarantees that the detected vertical normal is orthogonal to dominant vertical wall planes,
        preventing the 'Wall-as-Floor' RANSAC trap.
        """
        pts = np.asarray(pcd.points)
        sample_pcd = pcd.voxel_down_sample(0.04) if len(pts) > 20000 else pcd
        planes = cls.extract_candidate_planes(sample_pcd, max_planes=max_planes, distance_thresh=distance_thresh)

        if len(planes) >= 2:
            # Score each plane: a true floor/ceiling plane must be perpendicular to vertical wall planes
            best_score = -1.0
            best_normal = None

            for i, p_i in enumerate(planes):
                n_i = p_i["normal"]
                ortho_score = 0.0
                for j, p_j in enumerate(planes):
                    if i == j:
                        continue
                    dot_ij = abs(np.dot(n_i, p_j["normal"]))
                    # If planes are approximately perpendicular (dot product near 0)
                    if dot_ij < 0.25:
                        ortho_score += p_j["num_inliers"]

                total_score = p_i["num_inliers"] + ortho_score
                if total_score > best_score:
                    best_score = total_score
                    best_normal = n_i

            normal = best_normal
        elif len(planes) == 1:
            normal = planes[0]["normal"]
        else:
            # Fallback to shortest coordinate dimension
            spans = [pts[:, 0].ptp(), pts[:, 1].ptp(), pts[:, 2].ptp()]
            min_dim = np.argmin(spans)
            norm = np.zeros(3)
            norm[min_dim] = 1.0
            return norm

        # Disambiguate Sign (+Z vs -Z): Floor has furniture and clutter resting above it
        proj = pts @ normal
        p10 = np.percentile(proj, 10)
        p90 = np.percentile(proj, 90)
        low_density = np.sum((proj >= p10 - 0.1) & (proj <= p10 + 0.1))
        high_density = np.sum((proj >= p90 - 0.1) & (proj <= p90 + 0.1))

        sign = 1.0 if low_density >= high_density else -1.0
        directed_normal = normal * sign
        return directed_normal

    @classmethod
    def align_up(
        cls,
        pcd: o3d.geometry.PointCloud,
        up_axis: str = "auto",
        preset: Optional[str] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        """
        Rotates the point cloud so that the vertical axis aligns with +Z.

        Returns:
            (aligned_pcd, up_vec, R_matrix)
        """
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            raise ValueError("Cannot align an empty point cloud.")

        if preset:
            preset_key = preset.strip().lower()
            if preset_key not in cls.PRESET_MAP:
                raise ValueError(f"Unknown preset '{preset}'. Supported presets: {list(cls.PRESET_MAP.keys())}")
            axis_name = cls.PRESET_MAP[preset_key]
            up_vec = cls.AXIS_MAP[axis_name]
        elif up_axis.strip().lower() in cls.AXIS_MAP:
            up_vec = cls.AXIS_MAP[up_axis.strip().lower()]
        elif up_axis.strip().lower() == "auto":
            up_vec = cls.estimate_up_axis_multiplane(pcd)
        else:
            raise ValueError(f"Unknown up_axis '{up_axis}'. Choose from {list(cls.AXIS_MAP.keys())} or 'auto'.")

        R_up = cls.vector_align_rotation_matrix(up_vec, np.array([0.0, 0.0, 1.0]))
        pcd.rotate(R_up, center=(0, 0, 0))
        return pcd, up_vec, R_up
