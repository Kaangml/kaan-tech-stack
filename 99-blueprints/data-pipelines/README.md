# Scalable Data Pipelines

Blueprint for enterprise data pipeline architecture using Medallion pattern.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Data Sources                               │
│         (APIs, Databases, Files, Streaming)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ingestion Layer                             │
│              (Airbyte / Custom Connectors)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │  Bronze   │────►│  Silver   │────►│   Gold    │
    │   (Raw)   │     │ (Cleaned) │     │(Business) │
    └───────────┘     └───────────┘     └───────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
│                (Airflow / Dagster / Prefect)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Consumption Layer                             │
│           (BI Tools, ML Pipelines, APIs)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Medallion Architecture Implementation

### Bronze Layer (Raw)

```python
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, lit

spark = SparkSession.builder.appName("bronze-ingestion").getOrCreate()

def ingest_to_bronze(
    source_path: str,
    bronze_path: str,
    source_name: str,
    format: str = "json"
):
    """Ingest raw data to bronze layer with metadata."""
    
    df = spark.read.format(format).load(source_path)
    
    # Add ingestion metadata
    df_with_metadata = df \
        .withColumn("_ingested_at", current_timestamp()) \
        .withColumn("_source_file", input_file_name()) \
        .withColumn("_source_name", lit(source_name)) \
        .withColumn("_ingestion_date", lit(datetime.now().strftime("%Y-%m-%d")))
    
    # Write as Delta (or Parquet)
    df_with_metadata.write \
        .format("delta") \
        .mode("append") \
        .partitionBy("_ingestion_date") \
        .save(bronze_path)
    
    return df_with_metadata.count()

# Example usage
ingest_to_bronze(
    source_path="s3://raw-data/orders/*.json",
    bronze_path="s3://datalake/bronze/orders",
    source_name="ecommerce_api"
)
```

### Silver Layer (Cleaned)

```python
from pyspark.sql.functions import col, when, trim, lower, to_timestamp
from pyspark.sql.types import StructType

def bronze_to_silver(bronze_path: str, silver_path: str, schema: StructType = None):
    """Transform bronze to silver with cleaning and validation."""
    
    bronze_df = spark.read.format("delta").load(bronze_path)
    
    # Data quality rules
    silver_df = bronze_df \
        .dropDuplicates(["order_id"]) \
        .filter(col("order_id").isNotNull()) \
        .filter(col("amount") > 0) \
        .withColumn("email", lower(trim(col("email")))) \
        .withColumn("created_at", to_timestamp(col("created_at"))) \
        .withColumn("status", when(col("status").isNull(), "unknown")
                              .otherwise(col("status")))
    
    # Schema enforcement
    if schema:
        silver_df = spark.createDataFrame(silver_df.rdd, schema)
    
    # Write with merge for idempotency
    silver_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(silver_path)
    
    return silver_df.count()

# With Delta merge for incremental updates
def silver_merge(bronze_path: str, silver_path: str, key_columns: list):
    """Incremental merge to silver layer."""
    from delta.tables import DeltaTable
    
    bronze_df = spark.read.format("delta").load(bronze_path)
    silver_table = DeltaTable.forPath(spark, silver_path)
    
    merge_condition = " AND ".join([f"target.{c} = source.{c}" for c in key_columns])
    
    silver_table.alias("target").merge(
        bronze_df.alias("source"),
        merge_condition
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
```

### Gold Layer (Business)

```python
def create_gold_aggregates(silver_paths: dict, gold_path: str):
    """Create business-level aggregates for gold layer."""
    
    orders = spark.read.format("delta").load(silver_paths["orders"])
    customers = spark.read.format("delta").load(silver_paths["customers"])
    products = spark.read.format("delta").load(silver_paths["products"])
    
    # Customer 360 view
    customer_metrics = orders \
        .join(customers, "customer_id") \
        .groupBy(
            "customer_id",
            "customer_name",
            "customer_segment",
            "region"
        ) \
        .agg(
            count("order_id").alias("total_orders"),
            sum("amount").alias("lifetime_value"),
            avg("amount").alias("avg_order_value"),
            max("order_date").alias("last_order_date"),
            min("order_date").alias("first_order_date")
        ) \
        .withColumn("customer_tenure_days", 
                    datediff(col("last_order_date"), col("first_order_date")))
    
    customer_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .save(f"{gold_path}/customer_360")
    
    # Product performance
    product_metrics = orders \
        .join(products, "product_id") \
        .groupBy("product_id", "product_name", "category") \
        .agg(
            sum("quantity").alias("units_sold"),
            sum("amount").alias("revenue"),
            countDistinct("customer_id").alias("unique_customers")
        )
    
    product_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .save(f"{gold_path}/product_performance")
```

## Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="medallion_pipeline",
    default_args=default_args,
    schedule_interval="0 */4 * * *",  # Every 4 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["medallion", "etl"]
) as dag:
    
    ingest_orders = SparkSubmitOperator(
        task_id="ingest_orders_bronze",
        application="/jobs/bronze_ingestion.py",
        conf={"spark.executor.memory": "4g"},
        application_args=["--source", "orders", "--date", "{{ ds }}"]
    )
    
    clean_orders = SparkSubmitOperator(
        task_id="clean_orders_silver",
        application="/jobs/silver_transformation.py",
        application_args=["--source", "orders"]
    )
    
    aggregate_metrics = SparkSubmitOperator(
        task_id="aggregate_gold_metrics",
        application="/jobs/gold_aggregation.py"
    )
    
    ingest_orders >> clean_orders >> aggregate_metrics
```

## Data Quality Framework

```python
from dataclasses import dataclass
from typing import List, Callable
import great_expectations as gx

@dataclass
class QualityCheck:
    name: str
    check_fn: Callable
    severity: str  # "critical", "warning", "info"

class DataQualityFramework:
    def __init__(self, spark):
        self.spark = spark
        self.checks: List[QualityCheck] = []
        self.results = []
    
    def add_check(self, check: QualityCheck):
        self.checks.append(check)
    
    def run_checks(self, df) -> dict:
        results = {"passed": [], "failed": [], "warnings": []}
        
        for check in self.checks:
            try:
                passed = check.check_fn(df)
                if passed:
                    results["passed"].append(check.name)
                elif check.severity == "critical":
                    results["failed"].append(check.name)
                else:
                    results["warnings"].append(check.name)
            except Exception as e:
                results["failed"].append(f"{check.name}: {str(e)}")
        
        return results

# Common checks
def check_not_null(column: str):
    def check(df):
        null_count = df.filter(col(column).isNull()).count()
        return null_count == 0
    return check

def check_unique(column: str):
    def check(df):
        total = df.count()
        unique = df.select(column).distinct().count()
        return total == unique
    return check

def check_range(column: str, min_val, max_val):
    def check(df):
        out_of_range = df.filter(
            (col(column) < min_val) | (col(column) > max_val)
        ).count()
        return out_of_range == 0
    return check

# Usage
quality = DataQualityFramework(spark)
quality.add_check(QualityCheck("order_id_not_null", check_not_null("order_id"), "critical"))
quality.add_check(QualityCheck("order_id_unique", check_unique("order_id"), "critical"))
quality.add_check(QualityCheck("amount_positive", check_range("amount", 0, 1000000), "warning"))

results = quality.run_checks(silver_df)
if results["failed"]:
    raise Exception(f"Critical quality checks failed: {results['failed']}")
```

## Monitoring and Alerting

```python
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()

# Metrics
records_processed = Counter(
    'pipeline_records_processed_total',
    'Total records processed',
    ['layer', 'source']
)

processing_time = Histogram(
    'pipeline_processing_seconds',
    'Time spent processing',
    ['layer', 'source']
)

data_freshness = Gauge(
    'pipeline_data_freshness_seconds',
    'Seconds since last successful run',
    ['layer', 'source']
)

def log_pipeline_metrics(layer: str, source: str, count: int, duration: float):
    records_processed.labels(layer=layer, source=source).inc(count)
    processing_time.labels(layer=layer, source=source).observe(duration)
    data_freshness.labels(layer=layer, source=source).set(0)
    
    logger.info(
        "pipeline_completed",
        layer=layer,
        source=source,
        records=count,
        duration_seconds=duration
    )
```

## Related Resources

- [Data Architecture](../../2-data-engineering/architecture/) - Medallion, Star Schema patterns
- [PostgreSQL](../../6-databases/postgres-advanced/) - OLTP source systems
- [AWS Serverless](../../7-infrastructure/aws-serverless/) - Glue jobs
- [Data Stack](../../5-python-production/data-stack/) - Pandas, Spark
