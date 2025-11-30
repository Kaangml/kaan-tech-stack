# Python Advanced Patterns

Production-grade Python patterns for building scalable, maintainable systems.

## Topics

| Area | Description |
|------|-------------|
| [Web Frameworks](./web-frameworks/) | FastAPI, async patterns, middleware |
| [Data Stack](./data-stack/) | Pandas, Polars, Dask, memory optimization |
| [Testing](./testing/) | pytest, mocking, fixtures, property-based testing |

## What's NOT Here

- **PDF Processing** → [Document Processing](../2-data-engineering/document-processing/)
- **Basic Python** → This is senior-level patterns only

---

## Advanced Patterns Overview

### Concurrency & Async

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

# IO-bound: Use asyncio or ThreadPoolExecutor
async def fetch_all(urls: list[str]) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# CPU-bound: Use ProcessPoolExecutor
def process_heavy(data: list) -> list:
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(heavy_computation, data))
    return results

# Mixed: Combine both
async def mixed_workload(urls: list[str], data: list):
    loop = asyncio.get_event_loop()
    
    # IO-bound
    responses = await fetch_all(urls)
    
    # CPU-bound in thread pool (won't block event loop)
    with ProcessPoolExecutor() as pool:
        processed = await loop.run_in_executor(
            pool, partial(process_batch, data)
        )
    
    return responses, processed
```

### Type Safety at Runtime

```python
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Annotated
from functools import wraps
import inspect

class StrictConfig(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra='forbid'
    )

class Order(StrictConfig):
    order_id: str
    items: list[str]
    total: Annotated[float, "Must be positive"]
    
    @field_validator('total')
    @classmethod
    def validate_total(cls, v):
        if v <= 0:
            raise ValueError('Total must be positive')
        return v

# Runtime type checking decorator
def typecheck(func):
    hints = func.__annotations__
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        
        for param, value in bound.arguments.items():
            if param in hints:
                expected = hints[param]
                if not isinstance(value, expected):
                    raise TypeError(f"{param} must be {expected}, got {type(value)}")
        
        return func(*args, **kwargs)
    return wrapper
```

### Memory Optimization

```python
import sys
from dataclasses import dataclass
from typing import NamedTuple

# Regular class: ~152 bytes per instance
class PointClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# __slots__: ~56 bytes per instance
class PointSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# NamedTuple: ~72 bytes, immutable
class PointTuple(NamedTuple):
    x: float
    y: float

# Dataclass with slots: ~56 bytes, type hints
@dataclass(slots=True)
class PointDataclass:
    x: float
    y: float

# For 1M points:
# Regular: 152MB
# Slots:   56MB  ← 63% savings
```

### Dependency Injection

```python
from typing import Protocol
from functools import lru_cache
from contextlib import contextmanager

class Repository(Protocol):
    def get(self, id: str) -> dict: ...
    def save(self, entity: dict) -> None: ...

class DatabaseRepository:
    def __init__(self, connection_string: str):
        self.conn = create_connection(connection_string)
    
    def get(self, id: str) -> dict:
        return self.conn.execute("SELECT * FROM items WHERE id = ?", [id])
    
    def save(self, entity: dict) -> None:
        self.conn.execute("INSERT INTO items ...", entity)

class MockRepository:
    def __init__(self):
        self.data = {}
    
    def get(self, id: str) -> dict:
        return self.data.get(id)
    
    def save(self, entity: dict) -> None:
        self.data[entity['id']] = entity

# Simple DI container
class Container:
    _instances: dict = {}
    
    @classmethod
    def register(cls, interface, implementation):
        cls._instances[interface] = implementation
    
    @classmethod
    def resolve(cls, interface):
        return cls._instances[interface]

# Usage
Container.register(Repository, DatabaseRepository("postgresql://..."))
repo = Container.resolve(Repository)
```

---

## Quick Reference

| Pattern | Use Case | Module |
|---------|----------|--------|
| AsyncIO | High-concurrency IO | `asyncio`, `aiohttp` |
| Multiprocessing | CPU-bound parallelism | `concurrent.futures` |
| Pydantic | Runtime validation | `pydantic` |
| __slots__ | Memory optimization | Built-in |
| Protocol | Duck typing + hints | `typing` |
| Dependency Injection | Testing, modularity | Manual/`dependency-injector` |

## Related Resources

- [Clean Architecture](../1-architecture-patterns/clean-architecture/) - Project structure
- [Docker](../7-infrastructure/docker/) - Containerizing Python apps
- [FastAPI + Lambda](../7-infrastructure/aws-serverless/) - Serverless deployment
