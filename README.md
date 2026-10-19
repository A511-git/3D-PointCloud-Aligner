# 3D PointCloud Aligner for Indoor 3D Scenes

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.17+-orange.svg)](http://www.open3d.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](README.md)

**A modular, sensor-agnostic point cloud alignment library and CLI for indoor 3D reconstructions.**  
*Engineered for SpatialLM, SLAM (SLAM3R, MASt3R-SLAM, ORB-SLAM), LiDAR scanners, and 3D computer vision pipelines.*

</div>

---

## 📌 Overview

Raw 3D point clouds from SLAM systems, smartphone LiDAR, or monocular depth estimators often arrive at arbitrary orientations, uncalibrated scale, and with stray sensor reflections.

**`3D PointCloud Aligner`** automatically transforms arbitrary point clouds into the standard **ScanNet orientation convention**:
1. **Vertical Axis Aligned to $+Z$ Up**: Uses multi-plane peeling RANSAC with orthogonality consensus (preventing the "Wall-as-Floor" trap) and density percentile asymmetry to distinguish Floor from Ceiling.
2. **Walls Aligned to $X$ and $Y$ Planes (Manhattan World)**: Computes 4-fold periodic circular voting on horizontal surface normals ($\sum \cos(4(\theta - \alpha))$), seamlessly supporting rectangular, multi-corner, and L-shaped rooms.
3. **Metric Scale Calibration ($1\text{ unit} = 1\text{ meter}$)**: Detects floor-to-ceiling clearance and safely scales normalized unit boxes ($<0.8\text{m}$) or millimeter scans ($>50\text{m}$) while preserving calibrated metric scans.
4. **Grounded & Centered**: Shifts the floor level to $Z = 0.0$ and centers the room median at $(X=0.0, Y=0.0)$.
5. **Denoising & Downsampling**: Statistical outlier removal purges ghost points outside windows, followed by voxel downsampling.

---

## 🏷️ GitHub Topics & Keywords

When publishing to GitHub, set these repository topics in the **About** section for optimal search ranking:

`point-cloud` · `point-cloud-alignment` · `scannet` · `slam` · `manhattan-world` · `open3d` · `3d-reconstruction` · `lidar` · `arkit` · `spatiallm` · `computer-vision` · `depth-estimation`

---

## 🚀 Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install as an Editable Package (Recommended for Integration)
```bash
pip install -e .
```

---

## 💻 CLI Usage

### Basic Usage (Single File)
```bash
python run.py -i raw_scan.ply -o aligned_scan.ply
```

### Batch Processing (Preserves Directory Structure)
Recursively processes every `.ply` and `.pcd` file in a directory tree, preserving relative folder paths and filenames:
```bash
python run.py -i ./raw_scans/ -o ./aligned_scans/ --preview --save-meta
```

### Modular Stage Flags
Run only the specific steps you need:
```bash
# Only align floor and clean outliers
python run.py -i scan.ply -o floor_cleaned.ply --floor-align --clean

# Only square the walls to X/Y
python run.py -i scan.ply -o walls_aligned.ply --walls-align

# Rescale and ground an existing scan
python run.py -i scan.ply -o scaled_grounded.ply --scale --center-align --target-height 2.8 --force-scale
```

### Camera Presets (Skip Auto-Detection)
If your scanner or SLAM system uses a known coordinate frame:
* `--preset arkit` / `--preset opengl`: Rotates $+Y \to +Z$ Up (Apple iPhone/iPad LiDAR, Polycam)
* `--preset opencv`: Rotates $-Y \to +Z$ Up (OpenCV, COLMAP camera frames)
* `--preset scannet`: Retains $+Z$ Up (ScanNet, Matterport, ROS)

---

## 🐍 Python API (Embedding into Custom Repos)

You can embed `Aligner` directly into your SLAM or 3D vision pipeline without any disk read/write overhead:

```python
import open3d as o3d
from src.pipeline import PointCloudPipeline

# 1. Initialize pipeline with desired settings or presets
pipeline = PointCloudPipeline.from_preset(
    preset_name="arkit",
    target_height=2.8,
    voxel_size=0.03,
    do_preview=True,
    do_validate=True
)

# 2. Process an in-memory Open3D point cloud
raw_pcd = o3d.io.read_point_cloud("room_raw.ply")
aligned_pcd, metadata = pipeline.process(raw_pcd, output_path="room_aligned.ply")

print(f"Validation Status: {metadata['is_valid']}")
print(f"Detected Height: {metadata['transforms'].get('detected_height_m', 0.0):.2f}m")
```

---

## 📂 Project Architecture

```
3D-PointCloud-Aligner/
├── pyproject.toml              # Build & dependency packaging (pip install -e .)
├── requirements.txt            # Python dependencies (open3d, numpy, matplotlib)
├── LICENSE                     # MIT License (Mohammad Atif)
├── CITATION.cff                # Academic citation metadata
├── README.md                   # Project overview & CLI guide
├── TUNING_GUIDE.md             # In-depth sensor & SLAM parameter tuning guide
├── run.py                      # Multi-feature CLI entrypoint
├── src/                        # Modular library components
│   ├── __init__.py             # Public package exports
│   ├── floor_aligner.py        # Multi-plane RANSAC + consensus (+Z up)
│   ├── walls_aligner.py        # 4-fold periodic circular voting on wall yaw
│   ├── scale_calibrator.py     # Floor-to-ceiling metric height detector
│   ├── center_aligner.py       # Ground floor to Z=0 & center (X,Y)
│   ├── cleaner.py              # Statistical outlier filter & voxel downsampling
│   ├── preview.py              # 3-view orthogonal diagnostic PNG generator
│   ├── validator.py            # Quality assertions & geometry compliance
│   └── pipeline.py             # Orchestrator for in-memory & batch directory jobs
└── tests/
    ├── __init__.py
    └── test_aligner.py         # Self-contained synthetic test suite
```

---

## 📖 Tuning Guide

For detailed recommendations on handling monocular SLAM scale drift, partial scans without ceilings, and noisy depth sensors, see **[TUNING_GUIDE.md](TUNING_GUIDE.md)**.

---

## 📜 Citation

If you use this work in academic research, please cite:
```bibtex
@software{atif2026pointcloudaligner,
  author = {Mohammad Atif},
  title = {3D PointCloud Aligner: Modular Manhattan-World Alignment & Scale Calibration for Indoor 3D Scenes},
  year = {2026},
  url = {https://github.com/A511-git/3D-PointCloud-Aligner}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - free for personal, academic, and commercial use.
