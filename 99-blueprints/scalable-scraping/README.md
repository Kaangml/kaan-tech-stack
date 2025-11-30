# Scalable Scraping Architecture

Blueprint for high-volume, reliable web scraping at 10K+ pages/day.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Orchestrator                               │
│                    (FastAPI + Redis)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │  Worker 1 │     │  Worker 2 │     │  Worker N │
    │(Playwright)│    │(Playwright)│    │(Playwright)│
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
          └────────────────┬┴─────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Proxy Manager                               │
│              (Rotating Residential Proxies)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                     ┌───────────────┐
                     │   Target      │
                     │   Websites    │
                     └───────────────┘
```

## Core Components

### Job Queue (Redis)

```python
import redis
import json
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ScrapeJob:
    id: str
    url: str
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    status: JobStatus = JobStatus.PENDING

class JobQueue:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.queue_key = "scrape:jobs"
        self.processing_key = "scrape:processing"
    
    def enqueue(self, job: ScrapeJob):
        self.redis.zadd(
            self.queue_key,
            {json.dumps(asdict(job)): -job.priority}  # Higher priority = lower score
        )
    
    def dequeue(self) -> Optional[ScrapeJob]:
        # Atomic move from queue to processing
        result = self.redis.zpopmin(self.queue_key, 1)
        if not result:
            return None
        
        job_data = json.loads(result[0][0])
        job = ScrapeJob(**job_data)
        
        # Track processing
        self.redis.hset(self.processing_key, job.id, json.dumps(asdict(job)))
        return job
    
    def complete(self, job_id: str, result: dict):
        self.redis.hdel(self.processing_key, job_id)
        self.redis.hset("scrape:results", job_id, json.dumps(result))
    
    def fail(self, job: ScrapeJob, error: str):
        self.redis.hdel(self.processing_key, job.id)
        
        if job.retry_count < job.max_retries:
            job.retry_count += 1
            job.priority -= 1  # Lower priority on retry
            self.enqueue(job)
        else:
            self.redis.hset("scrape:failed", job.id, json.dumps({
                "job": asdict(job),
                "error": error
            }))
```

### Worker Pool

```python
import asyncio
from playwright.async_api import async_playwright
from typing import Callable

class ScraperWorker:
    def __init__(
        self,
        worker_id: str,
        queue: JobQueue,
        proxy_manager: 'ProxyManager',
        parser: Callable
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.proxy_manager = proxy_manager
        self.parser = parser
        self.running = False
    
    async def start(self):
        self.running = True
        async with async_playwright() as p:
            while self.running:
                job = self.queue.dequeue()
                if not job:
                    await asyncio.sleep(1)
                    continue
                
                try:
                    result = await self.process_job(p, job)
                    self.queue.complete(job.id, result)
                except Exception as e:
                    self.queue.fail(job, str(e))
    
    async def process_job(self, playwright, job: ScrapeJob) -> dict:
        proxy = self.proxy_manager.get_proxy()
        
        browser = await playwright.chromium.launch(
            headless=True,
            proxy={"server": proxy} if proxy else None
        )
        
        context = await browser.new_context(
            user_agent=self._get_random_user_agent(),
            viewport={"width": 1920, "height": 1080}
        )
        
        page = await context.new_page()
        
        try:
            await page.goto(job.url, wait_until="networkidle", timeout=30000)
            await self._human_like_delay()
            
            content = await page.content()
            data = self.parser(content, job.url)
            
            return {"status": "success", "data": data}
        finally:
            await browser.close()
    
    async def _human_like_delay(self):
        import random
        await asyncio.sleep(random.uniform(1, 3))
    
    def _get_random_user_agent(self) -> str:
        agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ]
        import random
        return random.choice(agents)

class WorkerPool:
    def __init__(self, num_workers: int, queue: JobQueue, proxy_manager, parser):
        self.workers = [
            ScraperWorker(f"worker-{i}", queue, proxy_manager, parser)
            for i in range(num_workers)
        ]
    
    async def start(self):
        await asyncio.gather(*[w.start() for w in self.workers])
    
    def stop(self):
        for w in self.workers:
            w.running = False
```

### Proxy Management

```python
import random
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Proxy:
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    failures: int = 0
    
    @property
    def url(self) -> str:
        if self.username:
            return f"http://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"http://{self.host}:{self.port}"

class ProxyManager:
    def __init__(self, proxies: List[Proxy], max_failures: int = 5):
        self.proxies = proxies
        self.max_failures = max_failures
        self._current_index = 0
    
    def get_proxy(self) -> Optional[str]:
        if not self.proxies:
            return None
        
        # Round-robin with health check
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.proxies)
            
            if proxy.failures < self.max_failures:
                return proxy.url
            attempts += 1
        
        return None
    
    def report_failure(self, proxy_url: str):
        for proxy in self.proxies:
            if proxy.url == proxy_url:
                proxy.failures += 1
                break
    
    def report_success(self, proxy_url: str):
        for proxy in self.proxies:
            if proxy.url == proxy_url:
                proxy.failures = max(0, proxy.failures - 1)
                break
```

### Rate Limiting

```python
import time
from collections import defaultdict
import asyncio

class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0):
        self.rps = requests_per_second
        self.domain_last_request: dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()
    
    async def acquire(self, domain: str):
        async with self._lock:
            now = time.time()
            last = self.domain_last_request[domain]
            wait_time = (1.0 / self.rps) - (now - last)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            self.domain_last_request[domain] = time.time()
    
    def set_rate(self, domain: str, rps: float):
        """Adjust rate for specific domains."""
        # Could store per-domain rates
        pass
```

### API Orchestrator

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
import uuid

app = FastAPI()
queue = JobQueue()

class ScrapeRequest(BaseModel):
    urls: List[str]
    priority: int = 0

class ScrapeResponse(BaseModel):
    batch_id: str
    job_count: int

@app.post("/scrape", response_model=ScrapeResponse)
async def submit_scrape_jobs(request: ScrapeRequest):
    batch_id = str(uuid.uuid4())
    
    for url in request.urls:
        job = ScrapeJob(
            id=f"{batch_id}:{uuid.uuid4()}",
            url=url,
            priority=request.priority
        )
        queue.enqueue(job)
    
    return ScrapeResponse(batch_id=batch_id, job_count=len(request.urls))

@app.get("/status/{batch_id}")
async def get_batch_status(batch_id: str):
    # Count jobs in various states
    completed = queue.redis.hkeys("scrape:results")
    completed_count = sum(1 for k in completed if k.decode().startswith(batch_id))
    
    failed = queue.redis.hkeys("scrape:failed")
    failed_count = sum(1 for k in failed if k.decode().startswith(batch_id))
    
    return {
        "batch_id": batch_id,
        "completed": completed_count,
        "failed": failed_count
    }

@app.get("/results/{batch_id}")
async def get_batch_results(batch_id: str):
    results = {}
    for key in queue.redis.hkeys("scrape:results"):
        if key.decode().startswith(batch_id):
            results[key.decode()] = json.loads(queue.redis.hget("scrape:results", key))
    return results
```

## Anti-Detection Strategies

### Browser Fingerprint Randomization

```python
async def create_stealth_context(browser):
    context = await browser.new_context(
        user_agent=random_user_agent(),
        viewport={"width": random.randint(1200, 1920), "height": random.randint(800, 1080)},
        locale=random.choice(["en-US", "en-GB", "en-CA"]),
        timezone_id=random.choice(["America/New_York", "Europe/London", "Asia/Tokyo"]),
        geolocation={"latitude": 40.7128, "longitude": -74.0060},
        permissions=["geolocation"]
    )
    
    # Inject anti-detection scripts
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    """)
    
    return context
```

### Request Pattern Variation

```python
async def human_like_browse(page, url):
    # Random delay before navigation
    await asyncio.sleep(random.uniform(0.5, 2))
    
    await page.goto(url)
    
    # Scroll behavior
    for _ in range(random.randint(1, 3)):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.3)")
        await asyncio.sleep(random.uniform(0.5, 1.5))
    
    # Mouse movement
    await page.mouse.move(
        random.randint(100, 500),
        random.randint(100, 500)
    )
```

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  orchestrator:
    build: ./orchestrator
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379

  worker:
    build: ./worker
    deploy:
      replicas: 5
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

## Related Resources

- [Scraping Tools](../../2-data-engineering/etl-pipelines/scraping-tools/) - HTTP clients and parsers
- [Browser Automation](../../4-automation/browser-automation/) - Playwright patterns
- [Docker](../../7-infrastructure/docker/) - Containerization
