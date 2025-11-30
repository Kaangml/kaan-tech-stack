# ⚡ Algorithms & Data Structures

> The foundation that separates engineers who solve problems from those who create them.

---

## 1. Senior Explanation

Algorithms aren't just interview fodder — they're **the difference between systems that scale and systems that collapse**.

**Why it's critical in production:**

| Scenario | Bad Choice | Good Choice | Impact |
|----------|------------|-------------|--------|
| Autocomplete | Linear scan O(n) | Trie O(k) | 100ms → 2ms |
| Deduplication | Nested loops O(n²) | Hash set O(n) | 10min → 0.5s |
| Scheduling | Brute force | Priority Queue | Timeout → Real-time |
| Graph traversal | Recursion (stack overflow) | Iterative BFS | Crash → Stable |

**Senior perspective:**
- Know Big-O, but optimize for **real bottlenecks**, not theoretical ones
- Data structure choice is 80% of algorithm performance
- Sometimes O(n²) on small n beats O(n log n) with high constants
- Memory access patterns matter more than instruction count

> *"You don't need to memorize algorithms. You need to recognize problem shapes."*

---

## 2. Real Issue & Fix

### Problem: Real-time Leaderboard at Scale

**Scenario:** Gaming platform with 10M users needs real-time leaderboard. Show top 100 + user's rank.

**Initial (Bad) Approach:**
```python
def get_rank(user_id: str) -> int:
    all_scores = db.query("SELECT user_id, score FROM scores ORDER BY score DESC")
    for i, row in enumerate(all_scores):
        if row.user_id == user_id:
            return i + 1
    return -1
# O(n) per request × 1000 RPS = Database melts
```

### Fix: Sorted Set with Skip List (Redis ZSET)

```python
import redis

r = redis.Redis()

# Update score: O(log n)
def update_score(user_id: str, score: int):
    r.zadd("leaderboard", {user_id: score})

# Get top 100: O(log n + 100)
def get_top_100() -> list[tuple[str, float]]:
    return r.zrevrange("leaderboard", 0, 99, withscores=True)

# Get user rank: O(log n)
def get_rank(user_id: str) -> int:
    rank = r.zrevrank("leaderboard", user_id)
    return rank + 1 if rank is not None else -1

# Get surrounding players: O(log n + k)
def get_neighbors(user_id: str, k: int = 5) -> list:
    rank = r.zrevrank("leaderboard", user_id)
    start = max(0, rank - k)
    return r.zrevrange("leaderboard", start, rank + k, withscores=True)
```

**Result:**
- 10M users, sub-millisecond responses
- Memory: ~400MB for 10M entries
- Handles 50K+ RPS

---

## 3. Code Snippet: LRU Cache from Scratch

```python
from collections import OrderedDict
from typing import TypeVar, Generic

K, V = TypeVar("K"), TypeVar("V")

class LRUCache(Generic[K, V]):
    """O(1) get/put LRU cache using OrderedDict."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[K, V] = OrderedDict()
    
    def get(self, key: K) -> V | None:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]
    
    def put(self, key: K, value: V) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Evict LRU

# Usage
cache = LRUCache[str, dict](capacity=1000)
cache.put("user:123", {"name": "Kaan", "score": 9500})
user = cache.get("user:123")  # O(1), promotes to MRU
```

---

## 4. Anti-Pattern Warning

### ⚠️ Premature Optimization

**Symptom:** Implementing complex algorithms before proving they're needed.

```python
# OVER-ENGINEERED: Bloom filter for 100 items
bloom = BloomFilter(capacity=100, error_rate=0.01)
for item in small_list:
    bloom.add(item)

# JUST DO THIS:
item_set = set(small_list)  # O(1) lookup, zero complexity
```

**When to optimize:**
1. **Profile first** — Measure, don't guess
2. **Quantify the gain** — Is 10ms → 5ms worth 200 lines of code?
3. **Consider maintenance** — Complex code has bugs

**Rule:** Make it work → Make it right → Make it fast (in that order)

---

## 5. My Stack

| Tool | Purpose |
|------|---------|
| **Python collections** | deque, Counter, defaultdict, OrderedDict |
| **heapq** | Priority queues without external deps |
| **bisect** | Binary search on sorted lists |
| **Redis** | Production-grade sorted sets, HyperLogLog |
| **NetworkX** | Graph algorithms prototyping |
| **NumPy** | Vectorized operations for numerical algorithms |

---

## Problem Shape Recognition

```
"Find/Search something"
├── Sorted data → Binary Search
├── Unsorted, need all → Linear scan
├── Prefix matching → Trie
└── Approximate → Bloom Filter / LSH

"Track frequency/count"
├── Exact count → HashMap
├── Top-K → Min-Heap of size K
└── Approximate count → Count-Min Sketch

"Process in order"
├── FIFO → Queue
├── LIFO → Stack
├── By priority → Heap
└── By time window → Deque

"Find relationships"
├── Is connected? → Union-Find
├── Shortest path → BFS (unweighted) / Dijkstra (weighted)
└── All paths → DFS with backtracking
```

---

*Algorithms are patterns. Learn the shapes, not the implementations.*
