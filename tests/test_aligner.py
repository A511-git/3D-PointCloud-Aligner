#!/usr/bin/env python3
"""
Self-contained unit and pipeline integration tests for the Aligner project.
Tests individual modules (FloorAligner, WallsAligner, ScaleCalibrator, CenterAligner, Cleaner, Validator)
and the unified PointCloudPipeline.
"""

import os
import sys
import unittest
import numpy as np
import open3d as o3d

# Ensure Aligner package root is on path
aligner_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if aligner_root not in sys.path:
    sys.path.insert(0, aligner_root)

from src.floor_aligner import FloorAligner
from src.walls_aligner import WallsAligner
from src.scale_calibrator import ScaleCalibrator
from src.center_aligner import CenterAligner
from src.cleaner import Cleaner
from src.preview import DiagnosticPreview
from src.validator import PointCloudValidator
from src.pipeline import PointCloudPipeline


def make_synthetic_room(length=5.0, width=4.0, height=2.8, density=500):
    """Generates an axis-aligned room with metric dimensions."""
    pts = []
    colors = []

    # Floor (z=0)
    fx = np.random.uniform(0, length, int(length * width * density))
    fy = np.random.uniform(0, width, len(fx))
    fz = np.zeros_like(fx)
    pts.append(np.stack([fx, fy, fz], axis=1))
    colors.append(np.full((len(fx), 3), [0.8, 0.75, 0.7]))

    # Ceiling (z=height)
    cx = np.random.uniform(0, length, int(length * width * density * 0.7))
    cy = np.random.uniform(0, width, len(cx))
    cz = np.full_like(cx, height)
    pts.append(np.stack([cx, cy, cz], axis=1))
    colors.append(np.full((len(cx), 3), [0.95, 0.95, 0.95]))

    # Walls
    w1_y = np.random.uniform(0, width, int(width * height * density))
    w1_z = np.random.uniform(0, height, len(w1_y))
    pts.append(np.stack([np.zeros_like(w1_y), w1_y, w1_z], axis=1))
    colors.append(np.full((len(w1_y), 3), [0.85, 0.85, 0.9]))

    w2_y = np.random.uniform(0, width, int(width * height * density))
    w2_z = np.random.uniform(0, height, len(w2_y))
    pts.append(np.stack([np.full_like(w2_y, length), w2_y, w2_z], axis=1))
    colors.append(np.full((len(w2_y), 3), [0.85, 0.85, 0.9]))

    w3_x = np.random.uniform(0, length, int(length * height * density))
    w3_z = np.random.uniform(0, height, len(w3_x))
    pts.append(np.stack([w3_x, np.zeros_like(w3_x), w3_z], axis=1))
    colors.append(np.full((len(w3_x), 3), [0.88, 0.88, 0.88]))

    w4_x = np.random.uniform(0, length, int(length * height * density))
    w4_z = np.random.uniform(0, height, len(w4_x))
    pts.append(np.stack([w4_x, np.full_like(w4_x, width), w4_z], axis=1))
    colors.append(np.full((len(w4_x), 3), [0.88, 0.88, 0.88]))

    # Bed box
    bx = np.random.uniform(1.0, 3.0, 1000)
    by = np.random.uniform(1.0, 2.5, 1000)
    bz = np.random.uniform(0.0, 0.6, 1000)
    pts.append(np.stack([bx, by, bz], axis=1))
    colors.append(np.full((len(bx), 3), [0.2, 0.3, 0.8]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    return pcd


class TestAlignerModules(unittest.TestCase):

    def setUp(self):
        self.pcd_gt = make_synthetic_room(length=5.0, width=4.0, height=2.8)
        self.test_data_dir = os.path.join(aligner_root, "tests", "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)

    def test_floor_aligner_auto(self):
        # Rotate room to Y-up
        R_tilt = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)
        pcd_tilted = o3d.geometry.PointCloud(self.pcd_gt)
        pcd_tilted.rotate(R_tilt, center=(0, 0, 0))

        aligned_pcd, up_vec, R_matrix = FloorAligner.align_up(pcd_tilted, up_axis="auto")
        pts = np.asarray(aligned_pcd.points)
        z_span = pts[:, 2].ptp()
        # Effective height should be back around ~2.8m in Z
        self.assertTrue(2.6 <= z_span <= 3.0, f"Z span {z_span} outside expected range")

    def test_walls_aligner(self):
        # Apply 30 degree yaw
        theta = np.deg2rad(30.0)
        R_yaw = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        pcd_yaw = o3d.geometry.PointCloud(self.pcd_gt)
        pcd_yaw.rotate(R_yaw, center=(0, 0, 0))

        aligned_pcd, detected_angle, R_mat = WallsAligner.align_walls(pcd_yaw)
        self.assertAlmostEqual(np.rad2deg(detected_angle) % 90, 30.0, delta=5.0)

    def test_scale_calibrator(self):
        # Scale to 0.1x (simulate unscaled unit box)
        pcd_small = o3d.geometry.PointCloud(self.pcd_gt)
        pcd_small.scale(0.1, center=(0, 0, 0))

        scaled_pcd, factor, orig_h = ScaleCalibrator.calibrate_scale(pcd_small, target_height=2.8)
        self.assertTrue(factor > 5.0, f"Expected large scaling factor, got {factor}")
        h_rec, _, _ = ScaleCalibrator.detect_room_height(scaled_pcd)
        self.assertTrue(2.7 <= h_rec <= 2.9)

    def test_center_and_cleaner(self):
        pcd_offset = o3d.geometry.PointCloud(self.pcd_gt)
        pcd_offset.translate([50.0, -30.0, 15.0])

        centered_pcd, trans = CenterAligner.center_and_ground(pcd_offset)
        pts = np.asarray(centered_pcd.points)
        floor_z = np.percentile(pts[:, 2], 2)
        self.assertAlmostEqual(floor_z, 0.0, delta=0.05)
        self.assertAlmostEqual(np.median(pts[:, 0]), 0.0, delta=0.1)

        # Cleaner
        cleaned_pcd, count_before, count_after = Cleaner.clean(centered_pcd, voxel_size=0.04)
        self.assertTrue(count_after < count_before)

    def test_end_to_end_pipeline(self):
        # Create heavily perturbed scan
        pcd_noisy = o3d.geometry.PointCloud(self.pcd_gt)
        # 25 degree yaw + ARKit Y-up rotation
        theta = np.deg2rad(25.0)
        R_yaw = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        R_arkit = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)
        pcd_noisy.rotate(R_yaw, center=(0, 0, 0))
        pcd_noisy.rotate(R_arkit, center=(0, 0, 0))

        # Add stray far-field outliers
        pts = np.asarray(pcd_noisy.points)
        outliers = np.array([[20.0, 15.0, 10.0], [-15.0, -18.0, 12.0]])
        all_pts = np.vstack([pts, outliers])
        all_colors = np.vstack([np.asarray(pcd_noisy.colors), np.full((2, 3), 0.5)])
        pcd_noisy.points = o3d.utility.Vector3dVector(all_pts)
        pcd_noisy.colors = o3d.utility.Vector3dVector(all_colors)

        in_ply = os.path.join(self.test_data_dir, "test_input.ply")
        out_ply = os.path.join(self.test_data_dir, "test_output.ply")
        o3d.io.write_point_cloud(in_ply, pcd_noisy)

        pipeline = PointCloudPipeline(
            up_axis="auto",
            target_height=2.8,
            do_preview=True,
            do_validate=True
        )

        aligned_pcd, meta = pipeline.process(in_ply, output_path=out_ply)
        self.assertTrue(meta.get("is_valid"), f"Pipeline validation failed: {meta.get('validation')}")
        self.assertTrue(os.path.exists(out_ply))

        # Check preview PNG
        preview_png = os.path.splitext(out_ply)[0] + "_preview.png"
        self.assertTrue(os.path.exists(preview_png), "Preview PNG should have been generated")


if __name__ == "__main__":
    unittest.main()
