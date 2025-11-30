# 🔄 ETL Pipelines & Web Scraping

> Extracting, transforming, and loading data at scale—from websites to warehouses.

---

## Concept

ETL pipelines move data from sources (APIs, databases, websites) through transformations (cleaning, enriching, aggregating) to destinations (data warehouses, lakes, operational systems). Scraping is a specialized extraction that pulls data from HTML when APIs don't exist.

---

## Why Use It?

**Senior perspective:**

In the real world, 60% of valuable data lives behind websites without APIs. Mastering web scraping unlocks:
- **Competitive intelligence**: Price monitoring, product catalogs
- **Market research**: Sentiment analysis, trend detection
- **Data aggregation**: Building datasets that don't exist elsewhere

But scraping at scale is an **adversarial problem**. Websites actively defend against it.

---

## Beautiful Soup vs lxml Performance

```python
import time
from bs4 import BeautifulSoup
from lxml import html as lxml_html

# Test HTML (typical e-commerce page)
test_html = open("product_page.html").read()  # ~500KB page

def benchmark_beautifulsoup():
    soup = BeautifulSoup(test_html, "lxml")  # lxml parser is fastest
    products = soup.select("div.product-card")
    return [
        {
            "name": p.select_one("h2.title").text.strip(),
            "price": p.select_one("span.price").text.strip(),
        }
        for p in products
    ]

def benchmark_lxml():
    tree = lxml_html.fromstring(test_html)
    products = tree.xpath("//div[contains(@class, 'product-card')]")
    return [
        {
            "name": p.xpath(".//h2[contains(@class, 'title')]/text()")[0].strip(),
            "price": p.xpath(".//span[contains(@class, 'price')]/text()")[0].strip(),
        }
        for p in products
    ]

# Benchmark results (1000 iterations):
# BeautifulSoup + lxml parser: ~2.3 seconds
# Pure lxml:                   ~0.8 seconds
# 
# lxml is ~3x faster for parsing
# BeautifulSoup is more readable for complex extraction
```

### When to Use What

| Library | Best For | Trade-offs |
|---------|----------|------------|
| **BeautifulSoup** | Complex HTML, broken markup, CSS selectors | Slower, more memory |
| **lxml** | Performance-critical, well-formed HTML, XPath | Less forgiving of bad HTML |
| **selectolax** | Ultra-fast, simple extractions | Limited features |

---

## Scalable Scraping Architecture

```python
import httpx
import asyncio
from dataclasses import dataclass
from typing import Iterator
import random

@dataclass
class ProxyConfig:
    host: str
    port: int
    username: str | None = None
    password: str | None = None
    
    def to_url(self) -> str:
        if self.username:
            return f"http://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"http://{self.host}:{self.port}"

class ScalableScraper:
    """
    Production-grade scraper with:
    - IP rotation
    - User-Agent rotation
    - Rate limiting
    - Retry logic
    - Session management
    """
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/17.2",
    ]
    
    def __init__(
        self,
        proxies: list[ProxyConfig] | None = None,
        requests_per_second: float = 2.0,
        max_retries: int = 3,
    ):
        self.proxies = proxies or []
        self.rate_limit = 1.0 / requests_per_second
        self.max_retries = max_retries
        self._last_request_time = 0
        self._proxy_index = 0
    
    def _get_next_proxy(self) -> str | None:
        if not self.proxies:
            return None
        proxy = self.proxies[self._proxy_index % len(self.proxies)]
        self._proxy_index += 1
        return proxy.to_url()
    
    def _get_random_headers(self) -> dict:
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def fetch(self, url: str, session: httpx.AsyncClient) -> str | None:
        """
        Fetch a URL with retries, rotation, and rate limiting.
        """
        await self._rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                proxy = self._get_next_proxy()
                response = await session.get(
                    url,
                    headers=self._get_random_headers(),
                    proxy=proxy,
                    timeout=30.0,
                    follow_redirects=True,
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    # Rate limited - back off
                    await asyncio.sleep(2 ** attempt * 5)
                elif response.status_code == 403:
                    # Blocked - rotate proxy and retry
                    continue
                else:
                    response.raise_for_status()
                    
            except httpx.TimeoutException:
                await asyncio.sleep(2 ** attempt)
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def fetch_many(self, urls: list[str], concurrency: int = 10) -> dict[str, str]:
        """
        Fetch multiple URLs concurrently with controlled parallelism.
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = {}
        
        async def fetch_with_semaphore(url: str, session: httpx.AsyncClient):
            async with semaphore:
                results[url] = await self.fetch(url, session)
        
        async with httpx.AsyncClient() as session:
            tasks = [fetch_with_semaphore(url, session) for url in urls]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return results


# Usage
async def scrape_product_catalog():
    proxies = [
        ProxyConfig(host="proxy1.example.com", port=8080),
        ProxyConfig(host="proxy2.example.com", port=8080),
    ]
    
    scraper = ScalableScraper(
        proxies=proxies,
        requests_per_second=5.0,
        max_retries=3,
    )
    
    product_urls = [f"https://example.com/product/{i}" for i in range(1, 1001)]
    html_pages = await scraper.fetch_many(product_urls, concurrency=20)
    
    return html_pages
```

---

## Anti-Bot Bypass Strategies

```python
from playwright.async_api import async_playwright
import asyncio

class StealthBrowser:
    """
    Playwright-based browser with anti-detection measures.
    Use when httpx gets blocked.
    """
    
    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ]
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="en-US",
            timezone_id="America/New_York",
        )
        
        # Remove webdriver property
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)
        
        return self
    
    async def __aexit__(self, *args):
        await self._browser.close()
        await self._playwright.stop()
    
    async def fetch(self, url: str, wait_for: str = None) -> str:
        """
        Fetch page with JavaScript rendering.
        
        Args:
            url: Target URL
            wait_for: CSS selector to wait for before extracting HTML
        """
        page = await self._context.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle")
            
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=10000)
            
            # Random delay to appear human
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            return await page.content()
        finally:
            await page.close()


# Usage
async def scrape_js_heavy_site():
    async with StealthBrowser() as browser:
        html = await browser.fetch(
            "https://js-heavy-site.com/products",
            wait_for="div.product-grid"
        )
        return html
```

---

## Data Cleaning Pipeline

```python
import pandas as pd
import re
from typing import Any

class DataCleaner:
    """
    Production data cleaning utilities for scraped data.
    """
    
    @staticmethod
    def clean_price(price_str: str) -> float | None:
        """Extract numeric price from string like '$1,234.56' or '€1.234,56'."""
        if not price_str:
            return None
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[^\d.,]', '', price_str.strip())
        
        # Handle European format (1.234,56)
        if ',' in cleaned and '.' in cleaned:
            if cleaned.rindex(',') > cleaned.rindex('.'):
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Could be decimal or thousands separator
            if len(cleaned.split(',')[-1]) == 2:
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and remove control characters."""
        if not text:
            return ""
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    @staticmethod
    def extract_number(text: str) -> int | None:
        """Extract first number from text like 'Rating: 4.5 (1,234 reviews)'."""
        match = re.search(r'[\d,]+', text.replace(',', ''))
        return int(match.group().replace(',', '')) if match else None
    
    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
        """
        Apply cleaning rules based on schema.
        
        Schema format: {"column_name": "cleaner_type"}
        Types: "text", "price", "number", "date"
        """
        df = df.copy()
        
        for col, cleaner_type in schema.items():
            if col not in df.columns:
                continue
            
            if cleaner_type == "text":
                df[col] = df[col].apply(lambda x: cls.clean_text(str(x)) if pd.notna(x) else None)
            elif cleaner_type == "price":
                df[col] = df[col].apply(lambda x: cls.clean_price(str(x)) if pd.notna(x) else None)
            elif cleaner_type == "number":
                df[col] = df[col].apply(lambda x: cls.extract_number(str(x)) if pd.notna(x) else None)
            elif cleaner_type == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df


# Usage
raw_data = pd.DataFrame({
    "name": ["  Product A\n", "Product B  ", "  Product C "],
    "price": ["$1,234.56", "€1.234,56", "£999.00"],
    "reviews": ["(1,234 reviews)", "256 ratings", "No reviews"],
})

cleaned = DataCleaner.clean_dataframe(raw_data, {
    "name": "text",
    "price": "price",
    "reviews": "number",
})

print(cleaned)
#        name     price  reviews
# 0  Product A  1234.56   1234.0
# 1  Product B  1234.56    256.0
# 2  Product C   999.00      NaN
```

---

## Tools I Use

| Tool | Purpose |
|------|---------|
| **httpx** | Async HTTP client (replaces requests) |
| **BeautifulSoup** | HTML parsing (complex/broken HTML) |
| **lxml** | High-performance parsing |
| **Playwright** | JavaScript rendering, stealth browsing |
| **Polars** | Fast DataFrames for processing |
| **Pandas** | Data cleaning, exploration |

---

## Checklist

```
□ Rate limiting implemented (respect robots.txt)
□ Proxy rotation for large-scale scraping
□ User-Agent rotation
□ Retry logic with exponential backoff
□ JavaScript rendering fallback (Playwright)
□ Data validation after extraction
□ Deduplication pipeline
□ Incremental scraping (track last scraped)
```

---

*Scraping is a cat-and-mouse game. The mouse needs to be smarter.*
