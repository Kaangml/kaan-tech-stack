# Data Engineering

ETL pipelines, data architecture, and processing at scale.

## Topics

| Area | Description |
|------|-------------|
| [ETL Pipelines](./etl-pipelines/) | Pipeline patterns, scraping, orchestration |
| [Architecture](./architecture/) | Medallion, Star Schema, Data Vault |
| [Geospatial](./geospatial-analysis/) | GIS, point clouds, spatial processing |
| [DB Optimization](./db-optimization/) | Query tuning, indexing strategies |

## Pipeline Patterns

```
Sources → Ingestion → Bronze (Raw) → Silver (Clean) → Gold (Business) → Consumption
```

## Key Technologies

| Category | Tools |
|----------|-------|
| Processing | Pandas, Polars, Dask, PySpark |
| Orchestration | Airflow, Dagster, Prefect |
| Storage | S3, Delta Lake, Parquet |
| Scraping | Playwright, HTTPX, BeautifulSoup |

## Related Blueprints
- [Data Pipelines](../99-blueprints/data-pipelines/) - Full medallion implementation
- [Scalable Scraping](../99-blueprints/scalable-scraping/) - High-volume extraction
- [Geospatial Pipeline](../99-blueprints/geospatial-pipeline/) - 3D data processing
