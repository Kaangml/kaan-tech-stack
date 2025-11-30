# ⚡ Database Optimization

> The difference between a query that takes 10ms and one that takes 10 minutes.

---

## 1. Why It Matters

Database performance is **the most common bottleneck** in production systems.

**The reality:**
- Most "slow applications" are actually "slow queries"
- A single missing index can bring down production
- ORMs hide performance problems until they explode

**Cost of bad queries:**
- 100ms delay = 1% conversion drop
- 10x query time = 10x server cost
- Long queries = lock contention = cascading failures

> *"The fastest query is the one you don't execute."*

---

## 2. Real-World Case

### Problem: Dashboard Timeout at 30 Seconds

**Scenario:** Analytics dashboard times out. Query scans 500M rows for daily metrics.

```sql
-- BEFORE: 45 seconds, full table scan
SELECT DATE(created_at), COUNT(*), SUM(amount)
FROM orders
WHERE created_at >= '2024-01-01' AND status = 'completed'
GROUP BY DATE(created_at);
```

### Fix: Index + Partition + Materialized View

```sql
-- 1. Covering index
CREATE INDEX CONCURRENTLY idx_orders_created_status 
ON orders (created_at, status) INCLUDE (amount)
WHERE status = 'completed';

-- 2. Partition by month
CREATE TABLE orders_part PARTITION BY RANGE (created_at);

-- 3. Materialized view for aggregates
CREATE MATERIALIZED VIEW daily_stats AS
SELECT DATE(created_at), status, COUNT(*), SUM(amount)
FROM orders GROUP BY 1, 2;

REFRESH MATERIALIZED VIEW CONCURRENTLY daily_stats;
```

**Result:** 45 seconds → 50ms (partition pruning + covering index + pre-aggregated)

---

## 3. Code: Query Performance Analyzer

```python
import psycopg

def analyze_query(conn, query: str) -> dict:
    """Analyze query and identify performance issues."""
    with conn.cursor() as cur:
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
        plan = cur.fetchone()[0][0]
    
    warnings = []
    
    def check_node(node):
        # Detect sequential scans on large tables
        if node.get("Node Type") == "Seq Scan":
            rows = node.get("Actual Rows", 0)
            if rows > 10000:
                warnings.append(f"Seq Scan on {node['Relation Name']}: {rows:,} rows")
        
        # Check estimate accuracy
        planned, actual = node.get("Plan Rows", 0), node.get("Actual Rows", 0)
        if planned and actual and max(planned, actual) / min(planned, actual) > 10:
            warnings.append("Row estimate off by 10x+ - run ANALYZE")
        
        for child in node.get("Plans", []):
            check_node(child)
    
    check_node(plan["Plan"])
    return {"time_ms": plan["Execution Time"], "warnings": warnings}
```

---

## 4. Architecture Notes

### Index Selection Guide

```
Equality (WHERE x = ?)      → B-Tree
Range (WHERE x > ?)         → B-Tree
Pattern (LIKE 'prefix%')    → B-Tree
Pattern (LIKE '%suffix')    → GIN + pg_trgm
Full-text search            → GIN + tsvector
JSON queries                → GIN
Geospatial                  → GiST / SP-GiST
```

### Query Anti-Patterns

```sql
-- ❌ Function on indexed column
WHERE DATE(created_at) = '2024-01-01'
-- ✅ Range query
WHERE created_at >= '2024-01-01' AND created_at < '2024-01-02'

-- ❌ SELECT * 
SELECT * FROM orders WHERE user_id = 1
-- ✅ Select needed columns (covering index)
SELECT id, status, amount FROM orders WHERE user_id = 1
```

---

## 5. Tools I Use

| Tool | Purpose |
|------|---------|
| **EXPLAIN ANALYZE** | Query plan analysis |
| **pg_stat_statements** | Query performance tracking |
| **PgBouncer** | Connection pooling |
| **pg_repack** | Online table rebuild |
| **pgBadger** | Log analysis |

### Essential PostgreSQL Queries

```sql
-- Find slow queries
SELECT query, mean_exec_time, calls FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;

-- Find unused indexes
SELECT indexrelname, idx_scan FROM pg_stat_user_indexes
WHERE idx_scan = 0;

-- Table bloat
SELECT relname, n_dead_tup, n_live_tup,
       round(n_dead_tup * 100.0 / nullif(n_live_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables ORDER BY n_dead_tup DESC;
```

---

*Fast databases aren't magic. They're well-indexed and well-maintained.*
