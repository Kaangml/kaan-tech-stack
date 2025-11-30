# Web Scraping Tools

HTTP clients, parsers, and browser automation for data extraction.

## Tool Selection

| Tool | Best For | Speed | JavaScript | Anti-Bot |
|------|----------|-------|------------|----------|
| requests | Simple APIs, static pages | Fast | ❌ | ❌ |
| httpx | Async scraping, HTTP/2 | Fast | ❌ | ❌ |
| BeautifulSoup | HTML parsing | - | ❌ | ❌ |
| lxml | High-performance parsing | Very Fast | ❌ | ❌ |
| Playwright | Dynamic pages, SPA | Slow | ✅ | ✅ |
| Selenium | Legacy automation | Slow | ✅ | Moderate |

## HTTPX

Modern async HTTP client.

### Basic Usage

```python
import httpx

# Sync
response = httpx.get("https://api.example.com/data")
data = response.json()

# Async
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com/data")
    data = response.json()

# With configuration
client = httpx.AsyncClient(
    timeout=30.0,
    headers={"User-Agent": "MyBot/1.0"},
    follow_redirects=True
)
```

### Concurrent Requests

```python
import httpx
import asyncio

async def fetch_all(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for url, response in zip(urls, responses):
            if isinstance(response, Exception):
                results.append({"url": url, "error": str(response)})
            else:
                results.append({"url": url, "data": response.json()})
        return results

# Rate-limited fetching
async def fetch_with_rate_limit(urls: list[str], rate: int = 10):
    semaphore = asyncio.Semaphore(rate)
    
    async def fetch_one(client, url):
        async with semaphore:
            response = await client.get(url)
            await asyncio.sleep(0.1)  # Polite delay
            return response
    
    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### Session Management

```python
async with httpx.AsyncClient() as client:
    # Login
    login_response = await client.post(
        "https://example.com/login",
        data={"username": "user", "password": "pass"}
    )
    
    # Cookies are automatically persisted
    protected_response = await client.get("https://example.com/dashboard")
```

## BeautifulSoup

HTML/XML parsing library.

### Basic Parsing

```python
from bs4 import BeautifulSoup
import httpx

response = httpx.get("https://example.com")
soup = BeautifulSoup(response.text, "lxml")  # or "html.parser"

# Find elements
title = soup.find("title").text
links = soup.find_all("a", href=True)
paragraphs = soup.find_all("p", class_="content")

# CSS selectors
items = soup.select("div.product-card > h2")
prices = soup.select("span.price")

# Get attributes
for link in links:
    href = link.get("href")
    text = link.text.strip()
```

### Table Extraction

```python
def extract_table(soup, table_id: str = None) -> list[dict]:
    table = soup.find("table", id=table_id) if table_id else soup.find("table")
    
    headers = [th.text.strip() for th in table.find_all("th")]
    
    rows = []
    for tr in table.find_all("tr")[1:]:  # Skip header row
        cells = [td.text.strip() for td in tr.find_all("td")]
        if cells:
            rows.append(dict(zip(headers, cells)))
    
    return rows
```

### Navigation

```python
# Parent/sibling navigation
element = soup.find("div", class_="target")
parent = element.parent
siblings = element.find_next_siblings("div")
prev = element.find_previous_sibling()

# Nested search
container = soup.find("div", id="main")
items = container.find_all("article")

# Text search
element = soup.find(string=lambda text: "keyword" in text.lower())
```

## lxml

High-performance XML/HTML parsing with XPath.

### XPath Queries

```python
from lxml import html
import httpx

response = httpx.get("https://example.com")
tree = html.fromstring(response.content)

# XPath selectors
titles = tree.xpath("//h1/text()")
links = tree.xpath("//a/@href")
prices = tree.xpath("//span[@class='price']/text()")

# Complex XPath
products = tree.xpath("//div[contains(@class, 'product')]")
for product in products:
    name = product.xpath(".//h2/text()")[0]
    price = product.xpath(".//span[@class='price']/text()")[0]
```

### vs BeautifulSoup

```python
# BeautifulSoup
soup = BeautifulSoup(html, "lxml")
items = soup.select("div.item > span.name")
names = [item.text for item in items]

# lxml (faster for large documents)
tree = html.fromstring(html_content)
names = tree.xpath("//div[@class='item']/span[@class='name']/text()")
```

## Playwright

Browser automation for dynamic content.

### Basic Usage

```python
from playwright.async_api import async_playwright

async def scrape_dynamic_page(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        
        # Wait for specific element
        await page.wait_for_selector(".product-list")
        
        # Extract data
        products = await page.query_selector_all(".product-card")
        data = []
        for product in products:
            name = await product.query_selector("h2")
            price = await product.query_selector(".price")
            data.append({
                "name": await name.inner_text(),
                "price": await price.inner_text()
            })
        
        await browser.close()
        return data
```

### Handling Infinite Scroll

```python
async def scrape_infinite_scroll(url: str, max_scrolls: int = 10):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        
        previous_height = 0
        for _ in range(max_scrolls):
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)  # Wait for content to load
            
            # Check if new content loaded
            current_height = await page.evaluate("document.body.scrollHeight")
            if current_height == previous_height:
                break
            previous_height = current_height
        
        # Extract all loaded content
        content = await page.content()
        await browser.close()
        return content
```

### Form Interaction

```python
async def login_and_scrape(url: str, username: str, password: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto(url)
        
        # Fill login form
        await page.fill("input[name='username']", username)
        await page.fill("input[name='password']", password)
        await page.click("button[type='submit']")
        
        # Wait for navigation
        await page.wait_for_url("**/dashboard**")
        
        # Now scrape authenticated content
        data = await page.inner_text(".user-data")
        
        # Save cookies for later
        cookies = await context.cookies()
        
        await browser.close()
        return data, cookies
```

## Anti-Bot Techniques

### Headers and Fingerprinting

```python
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

async with httpx.AsyncClient(headers=headers) as client:
    response = await client.get(url)
```

### Proxy Rotation

```python
import random

proxies = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080",
    "http://proxy3.example.com:8080",
]

async def fetch_with_proxy(url: str):
    proxy = random.choice(proxies)
    async with httpx.AsyncClient(proxies={"all://": proxy}) as client:
        return await client.get(url)
```

### Request Delays

```python
import asyncio
import random

async def polite_scrape(urls: list[str]):
    async with httpx.AsyncClient() as client:
        results = []
        for url in urls:
            response = await client.get(url)
            results.append(response)
            
            # Random delay between 1-3 seconds
            await asyncio.sleep(random.uniform(1, 3))
        
        return results
```

## Structured Scraping Pipeline

```python
from dataclasses import dataclass
from typing import Optional
import httpx
from bs4 import BeautifulSoup

@dataclass
class ScrapedItem:
    url: str
    title: str
    price: Optional[float]
    description: str

class Scraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=30,
            headers={"User-Agent": "..."}
        )
    
    async def fetch(self, path: str) -> BeautifulSoup:
        response = await self.client.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    
    async def parse_item(self, soup: BeautifulSoup, url: str) -> ScrapedItem:
        return ScrapedItem(
            url=url,
            title=soup.select_one("h1").text.strip(),
            price=self._parse_price(soup.select_one(".price")),
            description=soup.select_one(".description").text.strip()
        )
    
    def _parse_price(self, element) -> Optional[float]:
        if not element:
            return None
        text = element.text.strip().replace("$", "").replace(",", "")
        return float(text)
    
    async def close(self):
        await self.client.aclose()
```

## Related Resources

- [Browser Automation](../../4-automation/browser-automation/README.md) - Playwright deep dive
- [ETL Pipelines](../etl-pipelines/README.md) - Pipeline integration
- [Scalable Scraping Blueprint](../../99-blueprints/scalable-scraping-architecture/README.md) - Production patterns
