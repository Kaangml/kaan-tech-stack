# Data Architecture

Data modeling patterns, OLAP vs OLTP, and modern data architectures.

## OLTP vs OLAP

### OLTP (Online Transaction Processing)

- Optimized for **write-heavy** workloads
- Row-oriented storage
- Normalized schemas (3NF)
- Low latency, high concurrency
- Examples: PostgreSQL, MySQL, MongoDB

```sql
-- OLTP: Normalized tables
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    order_date TIMESTAMP,
    status VARCHAR(20)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(id),
    product_id INT REFERENCES products(id),
    quantity INT,
    unit_price DECIMAL(10,2)
);
```

### OLAP (Online Analytical Processing)

- Optimized for **read-heavy** workloads
- Column-oriented storage
- Denormalized schemas
- Complex aggregations
- Examples: Snowflake, BigQuery, ClickHouse, DuckDB

```sql
-- OLAP: Denormalized fact table
CREATE TABLE sales_fact (
    order_id INT,
    order_date DATE,
    customer_id INT,
    customer_name VARCHAR(255),
    customer_city VARCHAR(100),
    product_id INT,
    product_name VARCHAR(255),
    category VARCHAR(100),
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2)
);
```

### When to Use What

| Scenario | OLTP | OLAP |
|----------|------|------|
| User transactions | ✅ | ❌ |
| Real-time dashboards | ❌ | ✅ |
| Reporting/BI | ❌ | ✅ |
| Application backend | ✅ | ❌ |
| Data exploration | ❌ | ✅ |

## Medallion Architecture

Layered data architecture for data lakes.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Bronze    │ ──► │   Silver    │ ──► │    Gold     │
│ (Raw Data)  │     │ (Cleaned)   │     │ (Business)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Bronze Layer

Raw data, exact copy from source:

```python
# Ingest raw data
def ingest_to_bronze(source_path: str, bronze_path: str):
    df = spark.read.json(source_path)
    
    df_with_metadata = df.withColumn("_ingested_at", current_timestamp()) \
                         .withColumn("_source_file", input_file_name())
    
    df_with_metadata.write \
        .mode("append") \
        .format("delta") \
        .save(bronze_path)
```

### Silver Layer

Cleaned, deduplicated, validated:

```python
def transform_to_silver(bronze_path: str, silver_path: str):
    bronze_df = spark.read.format("delta").load(bronze_path)
    
    silver_df = bronze_df \
        .dropDuplicates(["id"]) \
        .filter(col("id").isNotNull()) \
        .withColumn("email", lower(trim(col("email")))) \
        .withColumn("created_at", to_timestamp(col("created_at")))
    
    silver_df.write \
        .mode("overwrite") \
        .format("delta") \
        .save(silver_path)
```

### Gold Layer

Business-ready, aggregated:

```python
def create_gold_table(silver_path: str, gold_path: str):
    orders = spark.read.format("delta").load(f"{silver_path}/orders")
    customers = spark.read.format("delta").load(f"{silver_path}/customers")
    
    # Business metrics
    gold_df = orders \
        .join(customers, "customer_id") \
        .groupBy("customer_id", "customer_name", "region") \
        .agg(
            count("order_id").alias("total_orders"),
            sum("amount").alias("total_revenue"),
            avg("amount").alias("avg_order_value")
        )
    
    gold_df.write \
        .mode("overwrite") \
        .format("delta") \
        .save(f"{gold_path}/customer_metrics")
```

## Star Schema

Dimensional modeling for analytics.

```
                    ┌─────────────────┐
                    │   dim_product   │
                    └────────┬────────┘
                             │
┌─────────────────┐    ┌─────┴─────┐    ┌─────────────────┐
│   dim_customer  │────│ fact_sales│────│    dim_date     │
└─────────────────┘    └─────┬─────┘    └─────────────────┘
                             │
                    ┌────────┴────────┐
                    │   dim_store     │
                    └─────────────────┘
```

### Implementation

```sql
-- Dimension tables
CREATE TABLE dim_customer (
    customer_key SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    name VARCHAR(255),
    email VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(100),
    -- SCD Type 2 columns
    valid_from DATE,
    valid_to DATE,
    is_current BOOLEAN
);

CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,  -- YYYYMMDD
    full_date DATE,
    year INT,
    quarter INT,
    month INT,
    month_name VARCHAR(20),
    week INT,
    day_of_week INT,
    day_name VARCHAR(20),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN
);

-- Fact table
CREATE TABLE fact_sales (
    sale_id SERIAL PRIMARY KEY,
    date_key INT REFERENCES dim_date(date_key),
    customer_key INT REFERENCES dim_customer(customer_key),
    product_key INT REFERENCES dim_product(product_key),
    store_key INT REFERENCES dim_store(store_key),
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    discount_amount DECIMAL(10,2)
);

-- Analytics query
SELECT 
    d.year,
    d.quarter,
    c.country,
    p.category,
    SUM(f.total_amount) as revenue,
    COUNT(DISTINCT f.customer_key) as unique_customers
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE d.year = 2024
GROUP BY d.year, d.quarter, c.country, p.category
ORDER BY revenue DESC;
```

## Data Vault

Flexible, auditable data modeling.

### Core Components

- **Hubs**: Business keys (unique identifiers)
- **Links**: Relationships between hubs
- **Satellites**: Descriptive attributes with history

```sql
-- Hub: Core business entity
CREATE TABLE hub_customer (
    hub_customer_id SERIAL PRIMARY KEY,
    customer_bk VARCHAR(50) NOT NULL,  -- Business key
    load_date TIMESTAMP,
    record_source VARCHAR(100)
);

-- Satellite: Attributes with history
CREATE TABLE sat_customer (
    hub_customer_id INT REFERENCES hub_customer,
    load_date TIMESTAMP,
    load_end_date TIMESTAMP,
    name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    hash_diff VARCHAR(64),  -- Change detection
    record_source VARCHAR(100),
    PRIMARY KEY (hub_customer_id, load_date)
);

-- Link: Relationships
CREATE TABLE link_customer_order (
    link_customer_order_id SERIAL PRIMARY KEY,
    hub_customer_id INT REFERENCES hub_customer,
    hub_order_id INT REFERENCES hub_order,
    load_date TIMESTAMP,
    record_source VARCHAR(100)
);
```

## Slowly Changing Dimensions (SCD)

### Type 1: Overwrite

```sql
UPDATE dim_customer 
SET email = 'new@email.com' 
WHERE customer_id = 'C001';
```

### Type 2: Historical Tracking

```python
def apply_scd_type_2(existing: DataFrame, incoming: DataFrame) -> DataFrame:
    # Find changed records
    changed = incoming.join(existing, "customer_id") \
        .filter(
            (incoming.name != existing.name) | 
            (incoming.email != existing.email)
        )
    
    # Close existing records
    closed = existing.join(changed, "customer_id") \
        .withColumn("valid_to", current_date()) \
        .withColumn("is_current", lit(False))
    
    # Insert new versions
    new_versions = changed \
        .withColumn("valid_from", current_date()) \
        .withColumn("valid_to", lit("9999-12-31").cast("date")) \
        .withColumn("is_current", lit(True))
    
    return unchanged.union(closed).union(new_versions)
```

## Modern Data Stack

```
┌─────────────────────────────────────────────────────────┐
│                      Presentation                        │
│              (Metabase, Superset, Tableau)              │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     Transformation                       │
│                    (dbt, Spark, SQL)                    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                       Storage                            │
│        (Snowflake, BigQuery, Delta Lake, S3)            │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                      Ingestion                           │
│            (Airbyte, Fivetran, Custom ETL)              │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Orchestration                         │
│               (Airflow, Dagster, Prefect)               │
└─────────────────────────────────────────────────────────┘
```

## Related Resources

- [ETL Pipelines](../etl-pipelines/README.md) - Pipeline implementation
- [PostgreSQL](../../6-databases/postgres-advanced/README.md) - OLTP database
- [Data Stack](../../5-python-production/data-stack/README.md) - Processing tools
- [Scalable Pipelines Blueprint](../../99-blueprints/scalable-data-pipelines/README.md) - Production patterns
