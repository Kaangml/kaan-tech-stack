# Computer Vision

3D reconstruction, Gaussian Splatting, and spatial computing technologies.

## 3D Reconstruction Pipeline

### Classic Pipeline (COLMAP-based)

```
Images → Feature Extraction → Feature Matching → 
Structure from Motion (SfM) → Dense Reconstruction → Mesh Generation
```

**COLMAP Workflow:**
```bash
# 1. Feature extraction
colmap feature_extractor \
    --database_path database.db \
    --image_path ./images

# 2. Feature matching
colmap exhaustive_matcher \
    --database_path database.db

# 3. Sparse reconstruction (SfM)
colmap mapper \
    --database_path database.db \
    --image_path ./images \
    --output_path ./sparse

# 4. Dense reconstruction
colmap image_undistorter \
    --image_path ./images \
    --input_path ./sparse/0 \
    --output_path ./dense

colmap patch_match_stereo \
    --workspace_path ./dense

colmap stereo_fusion \
    --workspace_path ./dense \
    --output_path ./dense/fused.ply
```

### Neural Radiance Fields (NeRF)

Implicit neural representation for novel view synthesis:

```python
# Simplified NeRF concept
class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(60, 256),  # Positional encoding input
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)   # RGB + density
        )
    
    def forward(self, x, direction):
        # x: 3D point, direction: viewing direction
        encoded = positional_encoding(x, direction)
        return self.mlp(encoded)

def render_ray(model, ray_origin, ray_direction, near, far, n_samples):
    # Sample points along ray
    t = torch.linspace(near, far, n_samples)
    points = ray_origin + t * ray_direction
    
    # Query network
    outputs = model(points, ray_direction)
    rgb, density = outputs[..., :3], outputs[..., 3]
    
    # Volume rendering
    weights = compute_weights(density, t)
    color = (weights * rgb).sum(dim=-1)
    return color
```

## Gaussian Splatting (3DGS)

State-of-the-art for real-time novel view synthesis:

### Core Concept

```
Point Cloud → 3D Gaussians (position, covariance, color, opacity) → 
Differentiable Rasterization → Rendered Image
```

**Key Properties per Gaussian:**
- Position (μ): 3D center point
- Covariance (Σ): 3D ellipsoid shape
- Color (SH): Spherical harmonics for view-dependent color
- Opacity (α): Transparency

### Training Pipeline

```python
# Simplified 3DGS training loop
class GaussianSplatting:
    def __init__(self, initial_points):
        self.positions = nn.Parameter(initial_points)
        self.covariances = nn.Parameter(torch.eye(3).expand(len(points), -1, -1))
        self.colors_sh = nn.Parameter(torch.rand(len(points), 16, 3))  # SH coeffs
        self.opacities = nn.Parameter(torch.ones(len(points)))
    
    def render(self, camera):
        # Project 3D Gaussians to 2D
        projected = project_gaussians(
            self.positions, 
            self.covariances, 
            camera
        )
        
        # Differentiable rasterization (CUDA)
        image = rasterize_gaussians(
            projected,
            self.colors_sh,
            self.opacities,
            camera
        )
        return image
    
    def train_step(self, gt_image, camera):
        rendered = self.render(camera)
        loss = l1_loss(rendered, gt_image) + ssim_loss(rendered, gt_image)
        
        loss.backward()
        self.optimizer.step()
        
        # Adaptive density control
        self.densify_and_prune()
```

### Practical Implementation

```bash
# Using official implementation
git clone https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting

# Train on your images
python train.py \
    -s /path/to/your/images \
    --eval \
    --iterations 30000

# Real-time viewer
./SIBR_viewers/bin/SIBR_gaussianViewer_app \
    -m /path/to/output
```

### 3DGS vs NeRF Comparison

| Aspect | NeRF | 3D Gaussian Splatting |
|--------|------|----------------------|
| Representation | Implicit (MLP) | Explicit (Gaussians) |
| Training Time | Hours | Minutes |
| Rendering Speed | ~1 FPS | 100+ FPS |
| Quality | Excellent | Excellent |
| Editability | Difficult | Easy (point manipulation) |
| Memory | Low | Higher (stores primitives) |

## Depth Estimation

### Monocular Depth

```python
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Using DPT (Dense Prediction Transformer)
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def estimate_depth(image):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth
    
    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic"
    ).squeeze()
    
    return depth.numpy()
```

### Stereo Depth

```python
import cv2

# Classic stereo matching
def stereo_depth(left_img, right_img):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    disparity = stereo.compute(left_img, right_img)
    depth = (focal_length * baseline) / disparity
    return depth
```

## Point Cloud Processing

### Open3D Basics

```python
import open3d as o3d

# Load and visualize
pcd = o3d.io.read_point_cloud("scene.ply")
o3d.visualization.draw_geometries([pcd])

# Preprocessing pipeline
def process_point_cloud(pcd):
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Remove outliers
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    
    # Estimate normals
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )
    
    return pcd_clean

# Mesh reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_clean, depth=9
)
```

### Point Cloud Registration

```python
# ICP (Iterative Closest Point) alignment
def align_point_clouds(source, target):
    # Initial rough alignment
    trans_init = np.eye(4)
    
    # Point-to-plane ICP
    reg = o3d.pipelines.registration.registration_icp(
        source, target, 
        max_correspondence_distance=0.02,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    source.transform(reg.transformation)
    return source, reg.transformation
```

## Camera Calibration

```python
import cv2
import numpy as np

def calibrate_camera(images, pattern_size=(9, 6)):
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return mtx, dist  # Camera matrix and distortion coefficients
```

## Use Cases

### Autonomous Vehicles
- LiDAR + Camera fusion
- Real-time depth estimation
- SLAM (Simultaneous Localization and Mapping)

### Real Estate / Architecture
- 3D property scanning
- Virtual tours (3DGS renders)
- Measurement from images

### Manufacturing
- Quality inspection
- Robot guidance
- Digital twin creation

## Related Resources

- [ML-Ops](../ml-ops/README.md) - Model training infrastructure
- [Geospatial Pipeline](../../99-blueprints/geospatial-pipeline/README.md) - Large-scale 3D processing
