# Geospatial Data Pipeline

Blueprint for processing large-scale geospatial data from 3D scans and point clouds.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Acquisition                            │
│              (LiDAR, Photogrammetry, Drone Scans)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Raw Storage                                │
│                   S3 / Azure Blob Storage                        │
│              (LAZ, LAS, PLY, Images)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │   Point   │     │   Image   │     │   Mesh    │
    │   Cloud   │     │ Processing│     │Generation │
    │ Processing│     │ (SfM)     │     │           │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
          └────────────────┬┴─────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Extraction                            │
│         (Segmentation, Classification, Measurements)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processed Storage                             │
│            PostGIS / Tile Server / Delta Lake                    │
└─────────────────────────────────────────────────────────────────┘
```

## Point Cloud Processing

### Reading and Basic Operations

```python
import laspy
import numpy as np
import open3d as o3d

def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """Load LAS/LAZ file into Open3D point cloud."""
    las = laspy.read(file_path)
    
    points = np.vstack([
        las.x, las.y, las.z
    ]).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors if available
    if hasattr(las, 'red'):
        colors = np.vstack([
            las.red / 65535,
            las.green / 65535,
            las.blue / 65535
        ]).T
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def preprocess_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Standard preprocessing pipeline."""
    
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
    
    # Orient normals consistently
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)
    
    return pcd_clean
```

### Ground Classification

```python
import pdal
import json

def classify_ground(input_path: str, output_path: str):
    """Classify ground points using PDAL."""
    
    pipeline = {
        "pipeline": [
            input_path,
            {
                "type": "filters.assign",
                "assignment": "Classification[:]=0"
            },
            {
                "type": "filters.elm"  # Extended Local Minimum
            },
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 12,
                "multiplier": 2.2
            },
            {
                "type": "filters.smrf",  # Simple Morphological Filter
                "slope": 0.2,
                "window": 16,
                "threshold": 0.45,
                "scalar": 1.2
            },
            {
                "type": "writers.las",
                "filename": output_path,
                "compression": "laszip"
            }
        ]
    }
    
    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()
    return p.metadata

# Classification codes (ASPRS)
# 2 = Ground
# 3 = Low Vegetation
# 4 = Medium Vegetation
# 5 = High Vegetation
# 6 = Building
```

### Building Extraction

```python
def extract_buildings(pcd: o3d.geometry.PointCloud) -> list:
    """Extract building footprints from classified point cloud."""
    
    # Assume non-ground, non-vegetation points are buildings
    points = np.asarray(pcd.points)
    
    # Cluster points
    labels = np.array(pcd.cluster_dbscan(
        eps=0.5,
        min_points=100,
        print_progress=True
    ))
    
    buildings = []
    for label in np.unique(labels):
        if label == -1:  # Noise
            continue
        
        cluster_points = points[labels == label]
        
        # Calculate building properties
        building = {
            "centroid": cluster_points.mean(axis=0).tolist(),
            "min_height": cluster_points[:, 2].min(),
            "max_height": cluster_points[:, 2].max(),
            "point_count": len(cluster_points),
            "footprint": calculate_footprint(cluster_points)
        }
        buildings.append(building)
    
    return buildings

def calculate_footprint(points: np.ndarray) -> list:
    """Calculate 2D convex hull of points."""
    from scipy.spatial import ConvexHull
    
    xy_points = points[:, :2]
    hull = ConvexHull(xy_points)
    
    return xy_points[hull.vertices].tolist()
```

## Distributed Processing

### Dask for Large Point Clouds

```python
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

def process_large_dataset(input_dir: str, output_dir: str):
    """Process multiple point cloud files in parallel."""
    
    client = Client(n_workers=4, threads_per_worker=2)
    
    # Read all LAZ files
    files = list(Path(input_dir).glob("*.laz"))
    
    # Create Dask bag of files
    import dask.bag as db
    file_bag = db.from_sequence(files)
    
    # Process each file
    results = file_bag.map(process_single_file).compute()
    
    # Merge results
    merged = merge_results(results)
    save_results(merged, output_dir)
    
    client.close()

def process_single_file(file_path: str) -> dict:
    """Process a single point cloud file."""
    pcd = load_point_cloud(file_path)
    pcd_clean = preprocess_point_cloud(pcd)
    buildings = extract_buildings(pcd_clean)
    
    return {
        "file": str(file_path),
        "point_count": len(pcd.points),
        "buildings": buildings
    }
```

### Tiling Strategy

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Tile:
    x: int
    y: int
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    file_path: str

class TileManager:
    def __init__(self, tile_size: float = 1000):  # 1km tiles
        self.tile_size = tile_size
        self.tiles: dict[Tuple[int, int], Tile] = {}
    
    def get_tile_index(self, x: float, y: float) -> Tuple[int, int]:
        tx = int(x // self.tile_size)
        ty = int(y // self.tile_size)
        return (tx, ty)
    
    def partition_points(self, points: np.ndarray) -> dict:
        """Partition points into tiles."""
        partitions = {}
        
        for point in points:
            tile_idx = self.get_tile_index(point[0], point[1])
            if tile_idx not in partitions:
                partitions[tile_idx] = []
            partitions[tile_idx].append(point)
        
        return {k: np.array(v) for k, v in partitions.items()}
    
    def save_tiles(self, partitions: dict, output_dir: Path):
        """Save partitioned data as individual tile files."""
        for (tx, ty), points in partitions.items():
            tile_path = output_dir / f"tile_{tx}_{ty}.parquet"
            
            df = pd.DataFrame(points, columns=['x', 'y', 'z'])
            df.to_parquet(tile_path)
            
            self.tiles[(tx, ty)] = Tile(
                x=tx, y=ty,
                bounds=self._calculate_bounds(points),
                file_path=str(tile_path)
            )
```

## Mesh Generation

### Poisson Reconstruction

```python
def generate_mesh(pcd: o3d.geometry.PointCloud, depth: int = 9) -> o3d.geometry.TriangleMesh:
    """Generate mesh from point cloud using Poisson reconstruction."""
    
    # Ensure normals are present
    if not pcd.has_normals():
        pcd.estimate_normals()
    
    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    
    # Remove low-density vertices (artifacts)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return mesh
```

### Gaussian Splatting Integration

```python
import subprocess
from pathlib import Path

def train_gaussian_splatting(
    image_dir: Path,
    output_dir: Path,
    iterations: int = 30000
):
    """Train 3D Gaussian Splatting model from images."""
    
    # Run COLMAP for camera poses
    colmap_output = output_dir / "colmap"
    run_colmap(image_dir, colmap_output)
    
    # Train 3DGS
    cmd = [
        "python", "train.py",
        "-s", str(colmap_output),
        "-m", str(output_dir / "model"),
        "--iterations", str(iterations),
        "--eval"
    ]
    
    subprocess.run(cmd, check=True)
    
    return output_dir / "model"

def run_colmap(image_dir: Path, output_dir: Path):
    """Run COLMAP sparse reconstruction."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    
    # Feature extraction
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1"
    ], check=True)
    
    # Feature matching
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path)
    ], check=True)
    
    # Sparse reconstruction
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir)
    ], check=True)
```

## PostGIS Storage

### Schema

```sql
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_raster;

-- Point cloud tiles
CREATE TABLE point_cloud_tiles (
    id SERIAL PRIMARY KEY,
    tile_x INT,
    tile_y INT,
    bounds GEOMETRY(POLYGON, 4326),
    point_count BIGINT,
    min_z FLOAT,
    max_z FLOAT,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tiles_bounds ON point_cloud_tiles USING GIST (bounds);

-- Extracted features
CREATE TABLE buildings (
    id SERIAL PRIMARY KEY,
    tile_id INT REFERENCES point_cloud_tiles(id),
    footprint GEOMETRY(POLYGON, 4326),
    height FLOAT,
    point_count INT,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_buildings_footprint ON buildings USING GIST (footprint);
```

### Queries

```python
import geopandas as gpd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/geodb")

def get_buildings_in_area(bounds: tuple) -> gpd.GeoDataFrame:
    """Get buildings within bounding box."""
    minx, miny, maxx, maxy = bounds
    
    query = f"""
    SELECT id, footprint, height, properties
    FROM buildings
    WHERE footprint && ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326)
    """
    
    return gpd.read_postgis(query, engine, geom_col="footprint")

def calculate_building_statistics(area_geom: str) -> dict:
    """Calculate statistics for buildings in an area."""
    query = f"""
    SELECT 
        COUNT(*) as building_count,
        AVG(height) as avg_height,
        MAX(height) as max_height,
        SUM(ST_Area(footprint::geography)) as total_area
    FROM buildings
    WHERE ST_Intersects(footprint, ST_GeomFromText('{area_geom}', 4326))
    """
    
    result = pd.read_sql(query, engine)
    return result.iloc[0].to_dict()
```

## Related Resources

- [Computer Vision](../../3-ai-ml/computer-vision/README.md) - 3D reconstruction techniques
- [Data Architecture](../../2-data-engineering/architecture/README.md) - Pipeline patterns
- [AWS Serverless](../../7-infrastructure/aws-serverless/README.md) - S3 storage
- [Docker](../../7-infrastructure/docker/README.md) - Containerizing processing
