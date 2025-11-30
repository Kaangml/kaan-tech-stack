# Automation

Browser automation, workflows, and event-driven systems.

## Topics

| Area | Description |
|------|-------------|
| [Browser Automation](./browser-automation/) | Playwright, headless browsers |
| [n8n Workflows](./n8n-workflows/) | Visual workflow automation |
| [Event-Driven](./event-driven/) | Message queues, pub/sub |

## Browser Automation Stack

```python
# Playwright for modern web automation
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto("https://example.com")
    await page.click("button.submit")
```

## Use Cases

| Scenario | Solution |
|----------|----------|
| Web scraping | Playwright + HTTPX |
| Form filling | Playwright + AI agents |
| CI/CD testing | Playwright Test |
| Visual workflows | n8n |
| Real-time triggers | Event-driven + webhooks |

## Related Blueprints
- [Autonomous Browser Agent](../99-blueprints/autonomous-browser-agent/) - LLM + browser control
- [Scalable Scraping](../99-blueprints/scalable-scraping/) - High-volume automation
