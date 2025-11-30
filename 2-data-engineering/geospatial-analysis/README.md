# 🌍 Geospatial Analysis

> Turning coordinates into actionable intelligence—from GPS points to polygon queries at scale.

---

## Concept

Geospatial analysis is the art of **working with location data**: points, lines, polygons, and their relationships. In production, this means answering questions like "Which delivery zone contains this address?" or "Find all stores within 5km of this location" — at sub-second latency.

---

## Why Use It?

**Senior perspective:**

80% of enterprise data has a location component that's underutilized. When you master geospatial, you unlock:
- **Logistics optimization**: Route planning, zone assignment, coverage analysis
- **Market intelligence**: Competitor proximity, demographic overlays
- **Compliance**: Geofencing, jurisdiction determination
- **User experience**: Location-based recommendations, ETA calculations

| Use Case | Without Geospatial | With Geospatial |
|----------|-------------------|-----------------|
| "Which region is this user in?" | String matching on address | Point-in-polygon query |
| "Find nearby stores" | Calculate all distances | Spatial index lookup (O(log n)) |
| "Delivery zone coverage" | Manual visual inspection | Automated polygon analysis |

---

## Core Concepts

### Coordinate Systems

```python
# CRITICAL: Always know your coordinate system

# WGS84 (EPSG:4326) - GPS coordinates, degrees
# This is what you get from phones, APIs, and most databases
lat, lng = 41.0082, 28.9784  # Istanbul

# Web Mercator (EPSG:3857) - Web maps, meters
# Used by Google Maps, OpenStreetMap for tile rendering
# Distance calculations are distorted!

# UTM (Universal Transverse Mercator) - Meters, zone-based
# Best for accurate distance/area calculations
# Istanbul is in UTM Zone 35N (EPSG:32635)

from pyproj import Transformer

# Transform between coordinate systems
wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
x, y = wgs84_to_utm.transform(28.9784, 41.0082)  # Note: lng, lat order for pyproj
print(f"UTM coordinates: {x:.2f}, {y:.2f}")  # Meters
```

### Polygon Creation from Coordinates

```python
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

def create_polygon_from_coordinates(coords: list[tuple[float, float]]) -> Polygon:
    """
    Create a polygon from a list of (lng, lat) coordinates.
    Coordinates should be in order (clockwise or counter-clockwise).
    """
    # Shapely uses (x, y) which is (lng, lat)
    polygon = Polygon(coords)
    
    if not polygon.is_valid:
        # Fix self-intersections
        polygon = polygon.buffer(0)
    
    return polygon

# Example: Define a neighborhood boundary
kadikoy_coords = [
    (29.0235, 40.9905),
    (29.0320, 40.9905),
    (29.0320, 40.9830),
    (29.0235, 40.9830),
    (29.0235, 40.9905),  # Close the polygon
]

kadikoy_polygon = create_polygon_from_coordinates(kadikoy_coords)
print(f"Area: {kadikoy_polygon.area:.6f} square degrees")

# For accurate area in square meters, transform to UTM first
from pyproj import Transformer
from shapely.ops import transform

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
kadikoy_utm = transform(transformer.transform, kadikoy_polygon)
print(f"Area: {kadikoy_utm.area:.2f} square meters")
```

### Point-in-Polygon Queries

```python
from shapely.geometry import Point
from shapely.strtree import STRtree
import geopandas as gpd

class NeighborhoodLocator:
    """
    Production-grade point-in-polygon lookup with spatial indexing.
    """
    
    def __init__(self, neighborhoods_gdf: gpd.GeoDataFrame):
        self.neighborhoods = neighborhoods_gdf
        # Build spatial index (R-tree) for O(log n) lookups
        self._tree = STRtree(neighborhoods_gdf.geometry.values)
        self._idx_to_name = dict(enumerate(neighborhoods_gdf['name'].values))
    
    def find_neighborhood(self, lat: float, lng: float) -> str | None:
        """
        Find which neighborhood contains a point.
        Returns None if point is not in any neighborhood.
        """
        point = Point(lng, lat)  # Shapely uses (x, y) = (lng, lat)
        
        # Get candidates from spatial index (fast)
        candidate_indices = self._tree.query(point)
        
        # Precise check on candidates (slower but accurate)
        for idx in candidate_indices:
            if self.neighborhoods.geometry.iloc[idx].contains(point):
                return self.neighborhoods['name'].iloc[idx]
        
        return None
    
    def find_neighborhoods_batch(self, points: list[tuple[float, float]]) -> list[str | None]:
        """
        Batch lookup for multiple points - more efficient than individual calls.
        """
        return [self.find_neighborhood(lat, lng) for lat, lng in points]


# Usage
neighborhoods = gpd.read_file("istanbul_neighborhoods.geojson")
locator = NeighborhoodLocator(neighborhoods)

# Single lookup
store_location = (40.9891, 29.0273)  # lat, lng
neighborhood = locator.find_neighborhood(*store_location)
print(f"Store is in: {neighborhood}")

# Batch lookup (e.g., from database)
customer_locations = [
    (40.9891, 29.0273),
    (41.0082, 28.9784),
    (41.0421, 29.0078),
]
results = locator.find_neighborhoods_batch(customer_locations)
```

---

## OpenStreetMap Data Processing

```python
import requests
import geopandas as gpd
from shapely.geometry import shape

class OSMDataExtractor:
    """Extract and process OpenStreetMap data via Overpass API."""
    
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    
    def get_neighborhood_boundaries(self, city: str) -> gpd.GeoDataFrame:
        """
        Extract neighborhood (admin_level=10) boundaries from OSM.
        """
        query = f"""
        [out:json][timeout:300];
        area["name"="{city}"]->.searchArea;
        (
          relation["admin_level"="10"](area.searchArea);
          relation["boundary"="administrative"]["admin_level"="9"](area.searchArea);
        );
        out body;
        >;
        out skel qt;
        """
        
        response = requests.post(self.OVERPASS_URL, data={"data": query})
        data = response.json()
        
        # Parse OSM elements to GeoDataFrame
        features = self._parse_osm_to_features(data)
        return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    
    def get_pois(self, bbox: tuple, poi_type: str) -> gpd.GeoDataFrame:
        """
        Extract Points of Interest within a bounding box.
        
        Args:
            bbox: (min_lat, min_lng, max_lat, max_lng)
            poi_type: OSM tag like "amenity=restaurant" or "shop=supermarket"
        """
        key, value = poi_type.split("=")
        query = f"""
        [out:json][timeout:60];
        (
          node["{key}"="{value}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          way["{key}"="{value}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        );
        out center;
        """
        
        response = requests.post(self.OVERPASS_URL, data={"data": query})
        data = response.json()
        
        pois = []
        for element in data.get("elements", []):
            if "center" in element:
                lat, lng = element["center"]["lat"], element["center"]["lon"]
            elif "lat" in element:
                lat, lng = element["lat"], element["lon"]
            else:
                continue
            
            pois.append({
                "geometry": Point(lng, lat),
                "osm_id": element["id"],
                "name": element.get("tags", {}).get("name", "Unknown"),
                "tags": element.get("tags", {}),
            })
        
        return gpd.GeoDataFrame(pois, crs="EPSG:4326")


# Example: Find all cafes in Kadıköy and determine which neighborhood each is in
extractor = OSMDataExtractor()

# Kadıköy bounding box
bbox = (40.97, 29.01, 41.01, 29.07)
cafes = extractor.get_pois(bbox, "amenity=cafe")

# Load neighborhood boundaries
neighborhoods = gpd.read_file("kadikoy_neighborhoods.geojson")

# Spatial join: assign each cafe to its neighborhood
cafes_with_neighborhoods = gpd.sjoin(cafes, neighborhoods, how="left", predicate="within")
print(cafes_with_neighborhoods[["name", "neighborhood_name"]].head())
```

---

## Production Example: Store Location Analyzer

```python
from dataclasses import dataclass
from shapely.geometry import Point
import geopandas as gpd

@dataclass
class StoreAnalysis:
    store_id: str
    address: str
    neighborhood: str | None
    district: str | None
    competitor_count_500m: int
    nearest_competitor_m: float | None
    population_density: float | None

class StoreLocationAnalyzer:
    """
    Production service for analyzing store locations.
    Used for site selection, competitive analysis, and market coverage.
    """
    
    def __init__(
        self,
        neighborhoods: gpd.GeoDataFrame,
        districts: gpd.GeoDataFrame,
        competitors: gpd.GeoDataFrame,
        demographics: gpd.GeoDataFrame,
    ):
        self.neighborhoods = neighborhoods
        self.districts = districts
        self.competitors = competitors
        self.demographics = demographics
        
        # Pre-build spatial indices
        self._neighborhood_tree = neighborhoods.sindex
        self._competitor_tree = competitors.sindex
    
    def analyze(self, store_id: str, lat: float, lng: float, address: str) -> StoreAnalysis:
        """Full location analysis for a store."""
        point = Point(lng, lat)
        
        # 1. Find containing neighborhood and district
        neighborhood = self._find_containing_polygon(point, self.neighborhoods, "name")
        district = self._find_containing_polygon(point, self.districts, "name")
        
        # 2. Count competitors within 500m
        buffer_500m = self._create_meter_buffer(point, 500)
        competitor_count = self._count_points_in_polygon(buffer_500m, self.competitors)
        
        # 3. Find nearest competitor
        nearest_dist = self._nearest_point_distance(point, self.competitors)
        
        # 4. Get population density for the neighborhood
        pop_density = None
        if neighborhood:
            match = self.demographics[self.demographics["neighborhood"] == neighborhood]
            if not match.empty:
                pop_density = match.iloc[0]["population_per_sqkm"]
        
        return StoreAnalysis(
            store_id=store_id,
            address=address,
            neighborhood=neighborhood,
            district=district,
            competitor_count_500m=competitor_count,
            nearest_competitor_m=nearest_dist,
            population_density=pop_density,
        )
    
    def _find_containing_polygon(
        self, point: Point, gdf: gpd.GeoDataFrame, name_col: str
    ) -> str | None:
        candidates = gdf[gdf.intersects(point)]
        if not candidates.empty:
            return candidates.iloc[0][name_col]
        return None
    
    def _create_meter_buffer(self, point: Point, meters: float) -> Polygon:
        """Create a buffer in meters around a point."""
        # Transform to UTM, buffer in meters, transform back
        from pyproj import Transformer
        from shapely.ops import transform
        
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
        to_wgs = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
        
        point_utm = transform(to_utm.transform, point)
        buffer_utm = point_utm.buffer(meters)
        return transform(to_wgs.transform, buffer_utm)
    
    def _count_points_in_polygon(self, polygon: Polygon, points_gdf: gpd.GeoDataFrame) -> int:
        return int(points_gdf.within(polygon).sum())
    
    def _nearest_point_distance(self, point: Point, points_gdf: gpd.GeoDataFrame) -> float | None:
        if points_gdf.empty:
            return None
        
        # Calculate distances
        distances = points_gdf.geometry.distance(point)
        min_idx = distances.idxmin()
        
        # Convert to meters
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        
        nearest_point = points_gdf.geometry.loc[min_idx]
        _, _, distance = geod.inv(
            point.x, point.y,
            nearest_point.x, nearest_point.y
        )
        return distance
```

---

## Tools I Use

| Tool | Purpose |
|------|---------|
| **GeoPandas** | Pandas + geometry support |
| **Shapely** | Geometric operations |
| **PostGIS** | Spatial queries in PostgreSQL |
| **H3** | Uber's hexagonal indexing |
| **pyproj** | Coordinate transformations |
| **Overpass API** | OpenStreetMap data extraction |
| **Folium / Kepler.gl** | Map visualization |

---

## Checklist

```
□ Know your coordinate system (EPSG:4326 for storage, UTM for calculations)
□ Use spatial indices for lookups (R-tree, STRtree)
□ Validate polygon geometry (is_valid, buffer(0) for fixes)
□ Batch operations when possible (spatial joins vs. individual lookups)
□ Cache frequently-used boundaries in memory
□ Transform to appropriate CRS before distance/area calculations
```

---

*Every data point has a story. Location tells you where that story happens.*
