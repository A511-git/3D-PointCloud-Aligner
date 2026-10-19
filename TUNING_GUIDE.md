# 3D PointCloud Aligner Parameter Tuning & Integration Guide

This guide explains how to integrate and tune `3D PointCloud Aligner` for custom 3D repositories, SLAM systems (e.g. SLAM3R, MASt3R-SLAM, ORB-SLAM), LiDAR scanners, and spatial AI pipelines.

---

## 1. Fast Integration into Custom Repos

### Option A: Install via Pip
From your custom repository:
```bash
pip install -e /path/to/3D-PointCloud-Aligner
```
Then in your Python code:
```python
from src.pipeline import PointCloudPipeline
import open3d as o3d

# In-memory processing directly from SLAM or reconstruction outputs
pipeline = PointCloudPipeline.from_preset("arkit", target_height=2.8)
aligned_pcd, metadata = pipeline.process(raw_pointcloud)
```

### Option B: Drop-in Source
Copy the `src/` folder directly into your own project:
```
my_custom_project/
├── aligner/               <-- copied from 3D-PointCloud-Aligner/src/
│   ├── floor_aligner.py
│   ├── walls_aligner.py
│   ├── scale_calibrator.py
│   ├── center_aligner.py
│   ├── cleaner.py
│   ├── preview.py
│   └── validator.py
└── my_slam_pipeline.py
```

---

## 2. Parameter Tuning Matrix

| Stage | Parameter | Default | When to Tune | Recommended Values |
| :--- | :--- | :--- | :--- | :--- |
| **Floor / Up-Axis** | `--preset` | `None` | You know the camera convention used during capture | `arkit`, `opengl` (Y-up), `opencv` (-Y up), `scannet` (Z-up) |
| | `--up-axis` | `auto` | Auto-consensus struggles due to slanted roofs or extreme clutter | `z`, `y`, `x` |
| | `distance_thresh` | `0.035m` | Noisy depth sensors (RealSense, Kinect) vs precision LiDAR | `0.05m - 0.08m` for noisy cameras, `0.02m` for LiDAR |
| **Scale Calibrator** | `--target-height` | `2.65m` | Scanning high-ceiling lofts, offices, or commercial venues | `2.6m - 2.8m` (apartments), `3.0m - 3.8m` (commercial) |
| | `--force-scale` | `False` | Working with **monocular SLAM** where scale is completely uncalibrated | Pass `--force-scale` |
| **Walls Aligner** | `wall_mask` $Z$ tolerance | `< 0.25` | Walls are slightly non-vertical or camera pitched during scan | `0.30 - 0.35` |
| **Cleaner** | `--voxel-size` | `0.03m` | Balancing processing speed vs fine object geometry (e.g., chair legs) | `0.015m - 0.02m` (fine detail), `0.04m - 0.05m` (fast) |
| | `--std-ratio` | `2.0` | Heavy sensor reflection/dust floating outside windows | `1.5` (stricter cleaning) to `2.5` (gentler cleaning) |
| | `--nb-neighbors` | `25` | Very sparse point clouds | `10 - 15` |
| **Centering** | `use_median` | `True` | Room has large outdoor balcony or one-sided open hallway | `True` prevents bounding box center skew |

---

## 3. Scenario-Specific Guidance

### Scenario 1: Monocular SLAM (e.g. ORB-SLAM3, DROID-SLAM)
* **Characteristics:** Monocular systems have scale drift or arbitrary unit scale (e.g. the room is 0.4 units tall).
* **Action:** Enable `--force-scale` and specify the true or estimated room height:
  ```bash
  python run.py -i scan.ply -o aligned.ply --target-height 2.8 --force-scale
  ```

### Scenario 2: Mobile AR Scans (Apple ARKit / RoomPlan / Polycam)
* **Characteristics:** Point clouds are already metric and follow the ARKit convention (+Y is Up, -Z is camera look direction).
* **Action:** Use `--preset arkit`. This skips the auto-consensus calculation and deterministically rotates +Y into +Z:
  ```bash
  python run.py -i arkit_scan.ply -o aligned.ply --preset arkit
  ```

### Scenario 3: Partial Scans (Missing Ceilings)
* **Characteristics:** Scans captured by human walking with a chest-mounted camera often fail to record the ceiling plane.
* **Risk:** Scale calibration might detect height between the floor and the top of a door frame (~2.0m).
* **Action:** Disable automatic scaling (`--scale` omitted or pass a lower `--target-height` reflecting camera trajectory span), or rely on external depth sensors like [Depth Pro](https://github.com/apple/ml-depth-pro) to calibrate metric depth prior to alignment.

### Scenario 4: Non-Rectangular or L-Shaped Rooms
* **Characteristics:** 6 or more walls instead of 4.
* **Why it works:** Because all walls in L-shaped rooms meet at 90° angles, the 4-fold periodic circular voting algorithm $\sum \cos(4(\theta - \alpha))$ automatically groups all walls into the exact same orientation consensus. No special flag is needed!

### Scenario 5: High-Precision Metric LiDAR (ScanNet, Matterport, Leica)
* **Characteristics:** The point cloud is already strictly metric.
* **Action:** Do NOT use `--force-scale`. The default conservative calibrator will inspect the detected height, confirm that it is between $0.8\text{m}$ and $50.0\text{m}$, and preserve the original metric coordinates without scaling.

---

## 4. In-Memory Python API Examples

### Full Pipeline with Previews
```python
from src.pipeline import PointCloudPipeline

pipeline = PointCloudPipeline(
    up_axis="auto",
    target_height=2.8,
    voxel_size=0.03,
    do_preview=True,
    do_validate=True
)

aligned_pcd, meta = pipeline.process("input_room.ply", output_path="output_room.ply")

print(f"Detected scale factor: {meta['transforms'].get('scale_factor', 1.0)}")
print(f"Wall yaw angle: {meta['transforms'].get('yaw_angle_rad', 0.0)} rad")
```

### Running Individual Modules
```python
import open3d as o3d
from src.floor_aligner import FloorAligner
from src.walls_aligner import WallsAligner
from src.center_aligner import CenterAligner

pcd = o3d.io.read_point_cloud("raw_scan.ply")

# 1. Align vertical axis to +Z
pcd, up_vec, R_up = FloorAligner.align_up(pcd, up_axis="auto")

# 2. Align walls to X/Y grid
pcd, yaw_angle, R_yaw = WallsAligner.align_walls(pcd)

# 3. Ground floor to Z=0 and center X, Y
pcd, translation = CenterAligner.center_and_ground(pcd)
```
