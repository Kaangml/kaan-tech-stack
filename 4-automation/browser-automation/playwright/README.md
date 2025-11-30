# Playwright

Modern browser automation framework by Microsoft.

## Installation

```bash
pip install playwright
playwright install  # Download browsers
```

## Core Concepts

### Browser Contexts
```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    
    # Isolated context (like incognito)
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Custom UA",
        locale="en-US"
    )
    
    page = await context.new_page()
    await page.goto("https://example.com")
```

### Selectors

```python
# CSS
await page.click("button.submit")
await page.click("#login-btn")

# Text
await page.click("text=Sign In")
await page.click("button:has-text('Submit')")

# XPath
await page.click("//button[@type='submit']")

# Role-based (recommended for accessibility)
await page.get_by_role("button", name="Submit").click()
await page.get_by_label("Email").fill("test@example.com")
```

### Network Interception

```python
async def handle_route(route):
    if "analytics" in route.request.url:
        await route.abort()
    else:
        await route.continue_()

await page.route("**/*", handle_route)

# Mock API response
await page.route(
    "**/api/users",
    lambda route: route.fulfill(json={"users": []})
)
```

### Authentication State

```python
# Save auth state
await context.storage_state(path="auth.json")

# Reuse auth state
context = await browser.new_context(storage_state="auth.json")
```

## Testing with Playwright

```python
import pytest
from playwright.sync_api import Page, expect

def test_login(page: Page):
    page.goto("https://example.com/login")
    
    page.get_by_label("Email").fill("user@example.com")
    page.get_by_label("Password").fill("password")
    page.get_by_role("button", name="Login").click()
    
    expect(page.get_by_text("Welcome")).to_be_visible()
```

## Anti-Detection

```python
context = await browser.new_context(
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
    viewport={"width": 1920, "height": 1080},
    locale="en-US",
    timezone_id="America/New_York"
)

# Hide webdriver
await context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
""")
```

## Related Resources
- [Scraping Tools](../../2-data-engineering/etl-pipelines/scraping-tools/) - Full scraping stack
- [Browser Agent](../../99-blueprints/autonomous-browser-agent/) - AI-powered automation
