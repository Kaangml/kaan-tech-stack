# Data Processing Stack

Python libraries for data manipulation, from single-machine to distributed processing.

## Library Selection Guide

| Library | Data Size | Best For | Memory Model |
|---------|-----------|----------|--------------|
| Pandas | < 10 GB | General analysis, prototyping | In-memory |
| Polars | < 100 GB | Fast analysis, large datasets | In-memory (lazy) |
| Dask | > 100 GB | Distributed, Pandas-like API | Out-of-core |
| PySpark | Petabytes | Big data, cluster computing | Distributed |
| NumPy | N/A | Numerical computing, arrays | In-memory |

## NumPy

Foundation for numerical computing.

### Core Operations

```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
range_arr = np.arange(0, 10, 0.5)
linspace = np.linspace(0, 1, 100)

# Reshaping
matrix = arr.reshape(5, 1)
flat = matrix.flatten()

# Broadcasting
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])     # (3,)
result = a + b  # (3, 3) - automatic broadcasting

# Vectorized operations (avoid loops!)
# Bad
result = [x**2 for x in arr]
# Good
result = arr ** 2
```

### Linear Algebra

```python
# Matrix operations
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)

# Matrix multiplication
C = A @ B  # or np.dot(A, B)

# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)

# Solve linear system Ax = b
b = np.array([1, 2, 3])
x = np.linalg.solve(A, b)
```

## Pandas

Standard for data analysis.

### DataFrame Operations

```python
import pandas as pd

# Reading data
df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_sql("SELECT * FROM table", connection)

# Basic exploration
df.head()
df.info()
df.describe()
df.shape
df.columns
df.dtypes

# Selection
df['column']              # Single column (Series)
df[['col1', 'col2']]     # Multiple columns (DataFrame)
df.loc[0:5, 'col1']      # Label-based
df.iloc[0:5, 0]          # Position-based
df.query("col1 > 5 and col2 == 'value'")  # Query syntax

# Filtering
mask = (df['age'] > 25) & (df['city'] == 'NYC')
filtered = df[mask]
```

### Transformation

```python
# Apply functions
df['new_col'] = df['col'].apply(lambda x: x * 2)
df['new_col'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# Groupby
grouped = df.groupby('category').agg({
    'sales': 'sum',
    'quantity': 'mean',
    'price': ['min', 'max']
})

# Pivot tables
pivot = pd.pivot_table(
    df,
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    fill_value=0
)

# Merge/Join
merged = pd.merge(df1, df2, on='key', how='left')
joined = df1.join(df2, on='key')

# Concat
combined = pd.concat([df1, df2], axis=0, ignore_index=True)
```

### Performance Tips

```python
# Use categorical for low-cardinality strings
df['category'] = df['category'].astype('category')

# Use appropriate dtypes
df['id'] = df['id'].astype('int32')  # Not int64
df['price'] = df['price'].astype('float32')  # Not float64

# Avoid iterrows - use vectorized operations
# Bad
for idx, row in df.iterrows():
    df.loc[idx, 'new'] = row['a'] + row['b']
# Good
df['new'] = df['a'] + df['b']

# Use query for complex filters
df.query("age > 25 and city == 'NYC'")

# Chunk large files
for chunk in pd.read_csv("large.csv", chunksize=100000):
    process(chunk)
```

## Polars

Fast DataFrame library written in Rust.

### Basic Operations

```python
import polars as pl

# Reading data
df = pl.read_csv("data.csv")
df = pl.read_parquet("data.parquet")

# Lazy evaluation (recommended)
lazy_df = pl.scan_csv("data.csv")

# Selection and filtering
result = (
    df.lazy()
    .filter(pl.col("age") > 25)
    .select([
        pl.col("name"),
        pl.col("salary").alias("income"),
        (pl.col("bonus") / pl.col("salary")).alias("bonus_ratio")
    ])
    .collect()
)

# Groupby with expressions
result = (
    df.lazy()
    .group_by("department")
    .agg([
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("salary").max().alias("max_salary"),
        pl.count().alias("employee_count"),
        pl.col("name").n_unique().alias("unique_names")
    ])
    .sort("avg_salary", descending=True)
    .collect()
)
```

### Polars Expressions

```python
# String operations
df.with_columns([
    pl.col("name").str.to_uppercase().alias("name_upper"),
    pl.col("email").str.contains("@gmail").alias("is_gmail")
])

# Date operations
df.with_columns([
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.weekday().alias("weekday")
])

# Window functions
df.with_columns([
    pl.col("sales").sum().over("region").alias("region_total"),
    pl.col("sales").rank().over("region").alias("region_rank")
])

# Conditional
df.with_columns([
    pl.when(pl.col("age") > 65)
    .then(pl.lit("senior"))
    .when(pl.col("age") > 18)
    .then(pl.lit("adult"))
    .otherwise(pl.lit("minor"))
    .alias("age_group")
])
```

### Pandas vs Polars

```python
# Pandas
df_pd = pd.read_csv("data.csv")
result_pd = (
    df_pd[df_pd['age'] > 25]
    .groupby('city')['salary']
    .mean()
)

# Polars (faster, more memory efficient)
df_pl = pl.scan_csv("data.csv")
result_pl = (
    df_pl
    .filter(pl.col("age") > 25)
    .group_by("city")
    .agg(pl.col("salary").mean())
    .collect()
)
```

## Dask

Distributed computing with familiar Pandas API.

### Parallel DataFrames

```python
import dask.dataframe as dd

# Read large files
df = dd.read_csv("data_*.csv")  # Glob pattern
df = dd.read_parquet("data/")   # Directory of parquet files

# Lazy operations (same as Pandas)
result = (
    df[df['amount'] > 100]
    .groupby('category')
    .agg({'amount': 'sum', 'count': 'size'})
)

# Execute
final = result.compute()

# Persist in memory (for repeated access)
df = df.persist()
```

### Dask Arrays

```python
import dask.array as da

# Large array operations
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = x + x.T
z = y.mean(axis=0)
result = z.compute()
```

## PySpark

Big data processing.

### SparkSession

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Read data
df = spark.read.parquet("s3://bucket/data/")
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Transformations
result = (
    df
    .filter(F.col("age") > 25)
    .groupBy("city")
    .agg(
        F.avg("salary").alias("avg_salary"),
        F.count("*").alias("count")
    )
    .orderBy(F.desc("avg_salary"))
)

# Actions
result.show()
result.write.parquet("output/")
```

### Window Functions

```python
from pyspark.sql.window import Window

window = Window.partitionBy("department").orderBy(F.desc("salary"))

df_ranked = df.withColumn(
    "rank",
    F.row_number().over(window)
).filter(F.col("rank") <= 3)  # Top 3 per department
```

## Visualization

### Matplotlib

```python
import matplotlib.pyplot as plt

# Basic plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x, y, 'b-', label='Line')
axes[0].scatter(x, y, c='red', alpha=0.5)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].legend()

axes[1].bar(categories, values)
axes[1].set_title('Bar Chart')

plt.tight_layout()
plt.savefig('plot.png', dpi=150)
plt.show()
```

### Seaborn

```python
import seaborn as sns

# Statistical visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data=df, x='age', hue='gender', ax=axes[0, 0])
sns.boxplot(data=df, x='category', y='value', ax=axes[0, 1])
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 0])
sns.scatterplot(data=df, x='x', y='y', hue='category', ax=axes[1, 1])

plt.tight_layout()
```

## Related Resources

- [ETL Pipelines](../../2-data-engineering/etl-pipelines/README.md) - Data pipeline patterns
- [PostgreSQL](../../6-databases/postgres-advanced/README.md) - Database integration
- [ML-Ops](../../3-ai-ml/ml-ops/README.md) - Using data for ML
