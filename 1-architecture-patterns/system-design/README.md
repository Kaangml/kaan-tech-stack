# 🌐 System Design

> The art of making technical decisions that survive contact with reality.

---

## 1. Senior Explanation

System design isn't about drawing boxes and arrows — it's about **making tradeoffs explicit**.

Every design decision has costs:
- **Consistency vs Availability** (CAP theorem)
- **Latency vs Throughput** (batching)
- **Simplicity vs Flexibility** (abstraction layers)
- **Cost vs Performance** (infrastructure)

**Why it's critical:**
- Wrong architecture at scale = rewrite (6-12 months)
- Right architecture = system grows with business
- Most "scaling problems" are actually "design problems" discovered late

**Senior perspective:**
- Start simple. Add complexity when you have evidence.
- Measure everything. Intuition fails at scale.
- Design for 10x, not 100x. You'll rewrite before 100x anyway.
- The best system is the one your team can operate.

> *"Premature optimization is the root of all evil, but premature architecture is the root of all rewrites."*

---

## 2. Real Issue & Fix

### Problem: Monolith Performance Collapse

**Scenario:** E-commerce monolith. Black Friday traffic = 20x normal. System dies at 5x.

**Symptoms:**
- Database connections exhausted
- Memory OOM from large queries
- Single slow endpoint blocks entire app
- Deploy takes 45 min, rollback takes 45 min

### Fix: Strategic Decomposition (Not Microservices!)

```
BEFORE (Monolith):
┌─────────────────────────────────────┐
│              Monolith               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │Users│ │Cart │ │Order│ │Inv. │   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
│     └───────┴───────┴───────┘       │
│              Single DB              │
└─────────────────────────────────────┘

AFTER (Modular Monolith + Extracted Hot Paths):
┌─────────────────────────────────────┐
│         Modular Monolith            │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │Users│ │Order│ │Ship.│           │
│  └─────┘ └─────┘ └─────┘           │
└───────────────┬─────────────────────┘
                │ Async (Queue)
                ▼
┌───────────────────┐  ┌───────────────────┐
│   Cart Service    │  │ Inventory Service │
│   (High Traffic)  │  │   (Read-Heavy)    │
│   Redis-backed    │  │   Read Replicas   │
└───────────────────┘  └───────────────────┘
```

**What we did:**
1. **Extracted hot paths only** — Cart and Inventory get 80% of traffic
2. **Kept core domain together** — Users, Orders, Shipping stay in monolith
3. **Async communication** — Queue between services, not sync HTTP
4. **Right database for job** — Redis for cart, read replicas for inventory

**Result:**
- Handles 50x traffic (10x headroom for growth)
- Cart service deploys in 2 min
- Monolith deploys in 10 min (smaller, less risk)

---

## 3. Code Snippet: Rate Limiter Design

```python
import time
from dataclasses import dataclass
import redis

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: float

class SlidingWindowRateLimiter:
    """Production rate limiter using Redis sorted sets."""
    
    def __init__(self, redis_client: redis.Redis, limit: int, window_seconds: int):
        self.redis = redis_client
        self.limit = limit
        self.window = window_seconds
    
    def check(self, key: str) -> RateLimitResult:
        now = time.time()
        window_start = now - self.window
        pipe = self.redis.pipeline()
        
        # Remove old entries, add new, count, set expiry
        redis_key = f"ratelimit:{key}"
        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zadd(redis_key, {f"{now}": now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, self.window)
        
        _, _, count, _ = pipe.execute()
        
        return RateLimitResult(
            allowed=count <= self.limit,
            remaining=max(0, self.limit - count),
            reset_at=now + self.window,
        )

# Usage
limiter = SlidingWindowRateLimiter(redis_client, limit=100, window_seconds=60)
result = limiter.check(f"user:{user_id}")
if not result.allowed:
    raise HTTPException(429, headers={"X-RateLimit-Reset": str(result.reset_at)})
```

---

## 4. Anti-Pattern Warning

### ⚠️ Resume-Driven Architecture

**Symptom:** Choosing technologies because they look good on a resume, not because they solve a problem.

```
❌ "Let's use Kubernetes!" (for 2 services and 1000 users)
❌ "Let's use Kafka!" (for 100 events/second)
❌ "Let's use microservices!" (team of 3)
❌ "Let's use GraphQL!" (one mobile app, one API consumer)
```

**Right sizing:**

| Scale | Solution |
|-------|----------|
| 1K users | Single server, SQLite |
| 100K users | Single server, PostgreSQL |
| 1M users | Load balancer + 2-3 servers + managed DB |
| 10M users | Maybe now you need that queue |
| 100M users | Now we can talk about microservices |

> *"The best architecture is the simplest one that solves today's problem with room for tomorrow's growth."*

---

## 5. My Stack

| Tool | Purpose |
|------|---------|
| **Excalidraw** | Quick architecture sketches |
| **Mermaid** | Diagrams as code in docs |
| **draw.io** | Detailed system diagrams |
| **k6 / Locust** | Load testing before scaling decisions |
| **Grafana + Prometheus** | Observability to validate design |
| **AWS Well-Architected Tool** | Structured design review |

---

## System Design Checklist

```
Requirements:
□ What's the expected QPS? (read vs write)
□ What's the data volume? (storage, bandwidth)
□ What's the latency requirement? (p50, p99)
□ What's the availability requirement? (99.9%? 99.99%?)

Data:
□ Read-heavy or write-heavy?
□ Strong consistency needed? Or eventual OK?
□ What's the access pattern? (by ID? by time range? full-text?)

Scaling:
□ Can we scale horizontally? What's the bottleneck?
□ What happens if component X fails?
□ How do we deploy without downtime?

Operations:
□ How do we monitor health?
□ How do we debug issues in production?
□ What's the incident response?
```

---

## Quick Reference: When to Use What

```
Need caching?
├── Single server → In-memory (dict, lru_cache)
├── Distributed → Redis
└── Heavy read traffic → CDN at edge

Need async processing?
├── Simple background jobs → Celery + Redis
├── Event streaming → Kafka
└── Workflow orchestration → Temporal

Need database scaling?
├── Read-heavy → Read replicas
├── Write-heavy → Sharding
└── Both → CQRS (separate read/write models)
```

---

*Good architecture is invisible. You only notice it when it's wrong.*
