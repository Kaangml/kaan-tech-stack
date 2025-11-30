# Browser Automation

Programmatic browser control for testing, scraping, and automation.

## Tool Comparison

| Tool | Best For | Speed | Ease |
|------|----------|-------|------|
| Playwright | Modern apps, multi-browser | Fast | ⭐⭐⭐ |
| Selenium | Legacy apps, wide support | Medium | ⭐⭐ |
| Puppeteer | Chrome-specific | Fast | ⭐⭐⭐ |

## Playwright Basics

```python
from playwright.async_api import async_playwright

async def scrape_page(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        
        title = await page.title()
        content = await page.content()
        
        await browser.close()
        return {"title": title, "content": content}
```

## Key Patterns

### Wait Strategies
```python
await page.wait_for_selector(".loaded")
await page.wait_for_load_state("networkidle")
await page.wait_for_timeout(1000)
```

### Element Interaction
```python
await page.click("button.submit")
await page.fill("input[name='email']", "test@example.com")
await page.select_option("select#country", "US")
```

### Screenshots & PDFs
```python
await page.screenshot(path="screenshot.png", full_page=True)
await page.pdf(path="page.pdf", format="A4")
```

## Sub-Topics
- [Playwright](./playwright/) - Deep dive into Playwright

## Related Resources
- [Scraping Tools](../2-data-engineering/etl-pipelines/scraping-tools/) - HTTP clients
- [Browser Agent Blueprint](../99-blueprints/autonomous-browser-agent/) - AI + browser
