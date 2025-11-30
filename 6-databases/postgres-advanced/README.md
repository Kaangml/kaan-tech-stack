# Relational Databases

PostgreSQL and MySQL for production applications.

## PostgreSQL

### Connection and Basics

```python
import psycopg2
from psycopg2.extras import RealDictCursor

# Connection
conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="postgres",
    password="password"
)

# Query with dict cursor
with conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute("SELECT * FROM users WHERE age > %s", (25,))
    users = cur.fetchall()

# Context manager
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO users (name) VALUES (%s)", ("Alice",))
    conn.commit()
```

### Async with asyncpg

```python
import asyncpg

async def main():
    conn = await asyncpg.connect(
        host='localhost',
        database='mydb',
        user='postgres',
        password='password'
    )
    
    # Single row
    row = await conn.fetchrow('SELECT * FROM users WHERE id = $1', 1)
    
    # Multiple rows
    rows = await conn.fetch('SELECT * FROM users WHERE age > $1', 25)
    
    # Execute
    await conn.execute(
        'INSERT INTO users (name, age) VALUES ($1, $2)',
        'Alice', 30
    )
    
    # Transaction
    async with conn.transaction():
        await conn.execute('UPDATE accounts SET balance = balance - $1 WHERE id = $2', 100, 1)
        await conn.execute('UPDATE accounts SET balance = balance + $1 WHERE id = $2', 100, 2)
    
    await conn.close()

# Connection pool
pool = await asyncpg.create_pool(
    host='localhost',
    database='mydb',
    user='postgres',
    password='password',
    min_size=5,
    max_size=20
)

async with pool.acquire() as conn:
    rows = await conn.fetch('SELECT * FROM users')
```

### JSONB Operations

```sql
-- Create table with JSONB
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    attributes JSONB DEFAULT '{}'
);

-- Insert JSON
INSERT INTO products (name, attributes) VALUES (
    'Laptop',
    '{"brand": "Dell", "specs": {"ram": 16, "storage": 512}}'
);

-- Query JSON fields
SELECT * FROM products WHERE attributes->>'brand' = 'Dell';
SELECT * FROM products WHERE attributes->'specs'->>'ram' = '16';
SELECT * FROM products WHERE (attributes->'specs'->>'ram')::int >= 16;

-- Containment
SELECT * FROM products WHERE attributes @> '{"brand": "Dell"}';

-- JSON path
SELECT * FROM products WHERE attributes @? '$.specs.ram ? (@ > 8)';

-- Update JSON
UPDATE products 
SET attributes = jsonb_set(attributes, '{specs,ram}', '32')
WHERE id = 1;

-- Index on JSONB
CREATE INDEX idx_products_brand ON products ((attributes->>'brand'));
CREATE INDEX idx_products_attributes ON products USING GIN (attributes);
```

### Indexing Strategies

```sql
-- B-tree (default, equality and range)
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_created ON users (created_at DESC);

-- Composite index (order matters!)
CREATE INDEX idx_users_city_age ON users (city, age);
-- Good for: WHERE city = 'NYC' AND age > 25
-- Good for: WHERE city = 'NYC'
-- Bad for: WHERE age > 25 (can't use without city)

-- Partial index
CREATE INDEX idx_active_users ON users (email) WHERE is_active = true;

-- GiST (geometric, full-text)
CREATE INDEX idx_locations ON places USING GIST (location);

-- GIN (arrays, JSONB, full-text)
CREATE INDEX idx_tags ON articles USING GIN (tags);

-- BRIN (large tables, sequential data)
CREATE INDEX idx_logs_time ON logs USING BRIN (created_at);

-- Check index usage
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### Full-Text Search

```sql
-- Add tsvector column
ALTER TABLE articles ADD COLUMN search_vector tsvector;

-- Update with content
UPDATE articles SET search_vector = 
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(content, '')), 'B');

-- Create index
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- Search
SELECT title, ts_rank(search_vector, query) as rank
FROM articles, to_tsquery('english', 'python & database') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- With highlighting
SELECT ts_headline('english', content, to_tsquery('python'))
FROM articles
WHERE search_vector @@ to_tsquery('python');
```

### Window Functions

```sql
-- Rank within partition
SELECT 
    department,
    name,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM employees;

-- Running total
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;

-- Moving average
SELECT 
    date,
    value,
    AVG(value) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as weekly_avg
FROM metrics;

-- Lead/Lag
SELECT 
    date,
    value,
    value - LAG(value) OVER (ORDER BY date) as daily_change,
    LEAD(value) OVER (ORDER BY date) as next_value
FROM metrics;
```

### CTEs and Recursive Queries

```sql
-- Simple CTE
WITH active_users AS (
    SELECT * FROM users WHERE is_active = true
)
SELECT * FROM active_users WHERE age > 25;

-- Recursive CTE (org hierarchy)
WITH RECURSIVE org_tree AS (
    -- Base case
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT e.id, e.name, e.manager_id, t.level + 1
    FROM employees e
    INNER JOIN org_tree t ON e.manager_id = t.id
)
SELECT * FROM org_tree ORDER BY level;
```

## MySQL

### Connection

```python
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    database="mydb",
    user="root",
    password="password"
)

cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM users WHERE age > %s", (25,))
users = cursor.fetchall()

conn.close()
```

### MySQL vs PostgreSQL

| Feature | PostgreSQL | MySQL |
|---------|------------|-------|
| JSONB | Native, indexed | JSON (less efficient) |
| Full-Text | Built-in | Basic |
| Arrays | Native | No (use JSON) |
| CTEs | Full support | Limited (pre-8.0) |
| Replication | Logical + Physical | Physical only |
| Concurrent writes | MVCC | Table/Row locking |

### MySQL-Specific Features

```sql
-- AUTO_INCREMENT
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255)
);

-- REPLACE (insert or update)
REPLACE INTO users (id, name) VALUES (1, 'Alice');

-- ON DUPLICATE KEY UPDATE
INSERT INTO counters (name, count) VALUES ('visits', 1)
ON DUPLICATE KEY UPDATE count = count + 1;

-- LIMIT with OFFSET
SELECT * FROM users LIMIT 10 OFFSET 20;
-- PostgreSQL: SELECT * FROM users LIMIT 10 OFFSET 20
```

## Performance Tuning

### Query Analysis

```sql
-- PostgreSQL
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- Detailed output
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) 
SELECT * FROM users WHERE email = 'test@example.com';
```

### Connection Pooling

```python
# PostgreSQL with psycopg2 pool
from psycopg2 import pool

connection_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host="localhost",
    database="mydb",
    user="postgres",
    password="password"
)

conn = connection_pool.getconn()
try:
    # Use connection
    pass
finally:
    connection_pool.putconn(conn)
```

### Bulk Operations

```python
# PostgreSQL bulk insert with COPY
import io

def bulk_insert(conn, data: list[dict], table: str):
    buffer = io.StringIO()
    for row in data:
        buffer.write('\t'.join(str(v) for v in row.values()) + '\n')
    buffer.seek(0)
    
    with conn.cursor() as cur:
        cur.copy_from(buffer, table, columns=data[0].keys())
    conn.commit()

# Or with executemany
cursor.executemany(
    "INSERT INTO users (name, age) VALUES (%s, %s)",
    [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
)
```

## Related Resources

- [Data Stack](../../5-python-production/data-stack/README.md) - Pandas with SQL
- [Web Frameworks](../../5-python-production/web-frameworks/README.md) - SQLAlchemy ORM
- [Data Architecture](../../2-data-engineering/architecture/README.md) - OLAP vs OLTP
