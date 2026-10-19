"""
Point Cloud Pipeline Coordinator.
Orchestrates modular alignment, scaling, centering, denoising, validation, and preview stages.
Supports both in-memory Python operations and batch directory processing preserving subfolder structures.
"""

import os
import copy
from typing import Optional, Dict, Any, Tuple, Union, List
import open3d as o3d
import numpy as np

from .floor_aligner import FloorAligner
from .walls_aligner import WallsAligner
from .scale_calibrator import ScaleCalibrator
from .center_aligner import CenterAligner
from .cleaner import Cleaner
from .preview import DiagnosticPreview
from .validator import PointCloudValidator


class PointCloudPipeline:
    """
    Unified coordinator for point cloud alignment.
    """

    def __init__(
        self,
        do_floor_align: bool = True,
        do_scale: bool = True,
        do_walls_align: bool = True,
        do_center: bool = True,
        do_clean: bool = True,
        up_axis: str = "auto",
        preset: Optional[str] = None,
        target_height: float = 2.65,
        force_scale: bool = False,
        voxel_size: float = 0.03,
        remove_outliers: bool = True,
        nb_neighbors: int = 25,
        std_ratio: float = 2.0,
        do_preview: bool = False,
        do_validate: bool = False
    ):
        self.do_floor_align = do_floor_align
        self.do_scale = do_scale
        self.do_walls_align = do_walls_align
        self.do_center = do_center
        self.do_clean = do_clean
        self.up_axis = up_axis
        self.preset = preset
        self.target_height = target_height
        self.force_scale = force_scale
        self.voxel_size = voxel_size
        self.remove_outliers = remove_outliers
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.do_preview = do_preview
        self.do_validate = do_validate

    @classmethod
    def from_preset(cls, preset_name: str, target_height: float = 2.65, **kwargs) -> "PointCloudPipeline":
        """
        Creates a pre-configured pipeline tailored to standard camera/sensor conventions.
        Supported presets: 'arkit', 'opencv', 'opengl', 'scannet'.
        """
        return cls(preset=preset_name, target_height=target_height, **kwargs)

    def process(
        self,
        pcd_or_path: Union[str, o3d.geometry.PointCloud],
        output_path: Optional[str] = None,
        preview_path: Optional[str] = None,
        copy_input: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
        """
        Executes the alignment pipeline on a single point cloud.

        Args:
            pcd_or_path: Filepath (str) or in-memory Open3D PointCloud.
            output_path: Optional filepath to save the aligned point cloud.
            preview_path: Optional filepath to save the diagnostic PNG preview.
            copy_input: If True and input is an in-memory PointCloud, makes a copy to avoid mutating the original.

        Returns:
            (aligned_pcd, metadata_dict)
        """
        if isinstance(pcd_or_path, str):
            if not os.path.exists(pcd_or_path):
                raise FileNotFoundError(f"Input point cloud not found: {pcd_or_path}")
            pcd = o3d.io.read_point_cloud(pcd_or_path)
            input_name = os.path.abspath(pcd_or_path)
        else:
            pcd = copy.deepcopy(pcd_or_path) if copy_input else pcd_or_path
            input_name = "in_memory_pcd"

        # Check and ensure RGB channels exist
        if not pcd.has_colors():
            pts_len = len(pcd.points)
            pcd.colors = o3d.utility.Vector3dVector(np.full((pts_len, 3), 0.65, dtype=np.float64))

        meta = {
            "source": input_name,
            "steps_applied": [],
            "transforms": {}
        }

        # Step 1: Up-Axis Alignment
        if self.do_floor_align:
            pcd, up_vec, R_up = FloorAligner.align_up(pcd, up_axis=self.up_axis, preset=self.preset)
            meta["steps_applied"].append("floor_align")
            meta["transforms"]["R_up"] = R_up.tolist()
            meta["transforms"]["up_vector"] = up_vec.tolist()

        # Step 2: Scale Calibration
        if self.do_scale:
            pcd, scale_factor, raw_height = ScaleCalibrator.calibrate_scale(
                pcd,
                target_height=self.target_height,
                force_scale=self.force_scale
            )
            meta["steps_applied"].append("scale_calibration")
            meta["transforms"]["scale_factor"] = scale_factor
            meta["transforms"]["detected_height_m"] = raw_height

        # Step 3: Walls / Manhattan Yaw Alignment
        if self.do_walls_align:
            pcd, yaw_angle, R_yaw = WallsAligner.align_walls(pcd)
            meta["steps_applied"].append("walls_align")
            meta["transforms"]["yaw_angle_rad"] = yaw_angle
            meta["transforms"]["R_yaw"] = R_yaw.tolist()

        # Step 4: Ground Floor and Center (X, Y)
        if self.do_center:
            pcd, trans_vec = CenterAligner.center_and_ground(pcd)
            meta["steps_applied"].append("center_and_ground")
            meta["transforms"]["translation"] = trans_vec.tolist()

        # Step 5: Denoising & Downsampling
        if self.do_clean:
            pcd, pts_before, pts_after = Cleaner.clean(
                pcd,
                remove_outliers=self.remove_outliers,
                nb_neighbors=self.nb_neighbors,
                std_ratio=self.std_ratio,
                voxel_size=self.voxel_size
            )
            meta["steps_applied"].append("clean")
            meta["transforms"]["points_before_clean"] = pts_before
            meta["transforms"]["points_after_clean"] = pts_after

        # Step 6: Diagnostic Preview
        if self.do_preview or preview_path:
            if not preview_path and output_path:
                preview_path = os.path.splitext(output_path)[0] + "_preview.png"
            if preview_path:
                DiagnosticPreview.generate_preview(pcd, preview_path)
                meta["preview_path"] = preview_path

        # Step 7: Validation
        if self.do_validate:
            is_valid, report = PointCloudValidator.validate(pcd)
            meta["validation"] = report
            meta["is_valid"] = is_valid

        # Save to disk if requested
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            o3d.io.write_point_cloud(output_path, pcd)
            meta["output_path"] = os.path.abspath(output_path)

        return pcd, meta

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = (".ply", ".pcd"),
        generate_previews: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Recursively processes all point clouds in input_dir, writing aligned clouds
        to output_dir while preserving relative folder hierarchy and file names.
        """
        results = []
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)

        for root, _, files in os.walk(input_dir):
            for file in sorted(files):
                if any(file.lower().endswith(ext) for ext in extensions):
                    in_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(in_file_path, input_dir)
                    out_file_path = os.path.join(output_dir, rel_path)

                    prev_path = None
                    if generate_previews or self.do_preview:
                        prev_path = os.path.splitext(out_file_path)[0] + "_preview.png"

                    print(f"Processing: {rel_path} -> {os.path.relpath(out_file_path, output_dir)}")
                    _, meta = self.process(
                        in_file_path,
                        output_path=out_file_path,
                        preview_path=prev_path
                    )
                    results.append(meta)

        return results
