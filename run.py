#!/usr/bin/env python3
"""
CLI Entrypoint for the Aligner Sub-Project.
Enables full-pipeline or individual stage execution for point cloud alignment,
with recursive folder structure preservation for batch jobs.
"""

import os
import sys
import json
import argparse

# Enable relative imports when executed directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.pipeline import PointCloudPipeline
from src.validator import PointCloudValidator


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modular Manhattan-World Point Cloud Aligner for Indoor 3D Scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input .ply/.pcd point cloud file OR directory of point clouds."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to output .ply file OR directory to store aligned point clouds."
    )

    # Stage Toggles (Modular flags)
    stages = parser.add_argument_group("Stage Toggles (Default: run all if none specified)")
    stages.add_argument(
        "--floor-align", "--up-align", action="store_true",
        help="Run vertical (+Z up) alignment via multi-plane consensus or preset."
    )
    stages.add_argument(
        "--scale", action="store_true",
        help="Run metric scale calibration (floor-to-ceiling clearance check)."
    )
    stages.add_argument(
        "--walls-align", action="store_true",
        help="Run 4-fold periodic Manhattan yaw alignment (align walls to X/Y axes)."
    )
    stages.add_argument(
        "--center-align", action="store_true",
        help="Center horizontal median to (0, 0) and ground floor to Z = 0."
    )
    stages.add_argument(
        "--clean", action="store_true",
        help="Run statistical outlier removal and voxel grid downsampling."
    )
    stages.add_argument(
        "--preview", action="store_true",
        help="Generate 3-panel orthogonal projection (Top, Front, Side) diagnostic PNG."
    )
    stages.add_argument(
        "--validate", action="store_true",
        help="Run compliance validation assertions against SpatialLM requirements."
    )

    # Tuning & Parameters
    params = parser.add_argument_group("Tuning & Camera Parameters")
    params.add_argument(
        "--preset", type=str, default=None,
        choices=["arkit", "opencv", "opengl", "scannet"],
        help="Camera sensor preset convention (e.g. arkit/opengl = +Y up, opencv = -Y up)."
    )
    params.add_argument(
        "--up-axis", type=str, default="auto",
        choices=["auto", "x", "-x", "y", "-y", "z", "-z"],
        help="Vertical axis override (or 'auto' for multi-plane consensus)."
    )
    params.add_argument(
        "--target-height", type=float, default=2.65,
        help="Target floor-to-ceiling clearance in meters."
    )
    params.add_argument(
        "--force-scale", action="store_true",
        help="Force rescaling to target-height even if detected height is within metric range."
    )
    params.add_argument(
        "--voxel-size", type=float, default=0.03,
        help="Voxel downsampling cube size in meters (0 to disable)."
    )
    params.add_argument(
        "--nb-neighbors", type=int, default=25,
        help="Number of nearest neighbors for statistical outlier filtering."
    )
    params.add_argument(
        "--std-ratio", type=float, default=2.0,
        help="Standard deviation multiplier for outlier removal."
    )
    params.add_argument(
        "--save-meta", action="store_true",
        help="Save transformation matrices and pipeline logs to a .json file."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine if any specific stage was requested
    stage_flags = [
        args.floor_align, args.scale, args.walls_align,
        args.center_align, args.clean, args.preview, args.validate
    ]
    # If no stage flag is explicitly supplied, run the full pipeline
    run_all = not any(stage_flags)

    do_floor = run_all or args.floor_align
    do_scale = run_all or args.scale
    do_walls = run_all or args.walls_align
    do_center = run_all or args.center_align
    do_clean = run_all or args.clean
    do_prev = args.preview or run_all
    do_val = args.validate or run_all

    pipeline = PointCloudPipeline(
        do_floor_align=do_floor,
        do_scale=do_scale,
        do_walls_align=do_walls,
        do_center=do_center,
        do_clean=do_clean,
        up_axis=args.up_axis,
        preset=args.preset,
        target_height=args.target_height,
        force_scale=args.force_scale,
        voxel_size=args.voxel_size,
        remove_outliers=do_clean,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        do_preview=do_prev,
        do_validate=do_val
    )

    print("\n" + "=" * 60)
    print("  Aligner: Modular Manhattan Point Cloud Alignment")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Active stages: floor={do_floor}, scale={do_scale}, walls={do_walls}, center={do_center}, clean={do_clean}")
    print("=" * 60 + "\n")

    if os.path.isdir(args.input):
        print(f"Directory detected. Recursively processing point clouds preserving folder tree...")
        results = pipeline.process_directory(args.input, args.output, generate_previews=do_prev)
        print(f"\nBatch processing finished: {len(results)} clouds processed.")
        if args.save_meta:
            meta_out = os.path.join(args.output, "alignment_manifest.json")
            with open(meta_out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved batch metadata manifest: {meta_out}")
    else:
        preview_file = (os.path.splitext(args.output)[0] + "_preview.png") if do_prev else None
        pcd, meta = pipeline.process(
            args.input,
            output_path=args.output,
            preview_path=preview_file
        )

        print("\nAlignment Complete!")
        if "validation" in meta:
            val = meta["validation"]
            status = "PASSED" if meta.get("is_valid") else "FLAGGED"
            print(f"Validation Status: {status}")
            if val.get("warnings"):
                print("Warnings:")
                for w in val["warnings"]:
                    print(f"  - {w}")

        if args.save_meta:
            meta_out = os.path.splitext(args.output)[0] + "_meta.json"
            with open(meta_out, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved alignment metadata: {meta_out}")


if __name__ == "__main__":
    main()
