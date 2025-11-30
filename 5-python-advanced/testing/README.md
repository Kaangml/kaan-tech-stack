# 🧪 Testing

> Tests are not about finding bugs. They're about enabling change.

---

## 1. Senior Explanation

Testing isn't a QA activity — it's a **design activity**.

**Why testing is critical:**
- **Confidence to refactor**: Without tests, refactoring is gambling
- **Living documentation**: Tests show how code is meant to be used
- **Design feedback**: Hard-to-test code is poorly designed code
- **Deployment velocity**: No tests = manual QA bottleneck = slow releases

**The testing pyramid reality:**

```
                    /\
                   /  \  E2E (few, slow, brittle)
                  /____\
                 /      \  Integration (some, medium)
                /________\
               /          \  Unit (many, fast, stable)
              /____________\
```

**Senior take:**
- Unit tests are cheap. Write many.
- Integration tests catch real bugs. Write enough.
- E2E tests are expensive. Write few, for critical paths only.
- **The goal isn't coverage %. It's confidence to deploy.**

> *"A test that doesn't give you confidence to deploy is waste."*

---

## 2. Real Issue & Fix

### Problem: Flaky Integration Tests

**Scenario:** CI pipeline fails randomly. Same tests pass locally. Team starts ignoring failures.

**Root causes identified:**
1. Tests depend on insertion order (database)
2. Tests share state (one test pollutes another)
3. Tests depend on wall clock time
4. Tests hit real external services

### Fix: Isolated, Deterministic Tests

```python
import pytest
from datetime import datetime
from unittest.mock import patch
from freezegun import freeze_time

# 1. Database isolation: Each test gets fresh state
@pytest.fixture
def db_session(test_db):
    """Each test runs in a transaction that's rolled back."""
    connection = test_db.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()  # Clean slate for next test
    connection.close()

# 2. Time isolation: Freeze time
@freeze_time("2024-01-15 10:30:00")
def test_subscription_expires():
    sub = Subscription(expires_at=datetime(2024, 1, 14))
    assert sub.is_expired() is True

# 3. External service isolation: Mock at boundary
@pytest.fixture
def mock_payment_gateway():
    with patch("app.payments.gateway.StripeClient") as mock:
        mock.return_value.charge.return_value = PaymentResult(success=True)
        yield mock

def test_checkout_charges_card(mock_payment_gateway, db_session):
    order = create_test_order(db_session)
    result = checkout_service.process(order)
    assert result.status == "completed"
    mock_payment_gateway.return_value.charge.assert_called_once()
```

**Result:**
- CI passes 100% of the time
- Tests run in parallel (no shared state)
- Test suite: 2 min → 30 sec

---

## 3. Code Snippet: Parameterized Testing

```python
import pytest
from decimal import Decimal

def calculate_tax(amount: Decimal, country: str) -> Decimal:
    rates = {"US": Decimal("0.08"), "DE": Decimal("0.19"), "JP": Decimal("0.10")}
    return amount * rates.get(country, Decimal("0"))

# ❌ REPETITIVE
def test_tax_us():
    assert calculate_tax(Decimal("100"), "US") == Decimal("8")

def test_tax_de():
    assert calculate_tax(Decimal("100"), "DE") == Decimal("19")

# ✅ PARAMETERIZED
@pytest.mark.parametrize("amount,country,expected", [
    (Decimal("100"), "US", Decimal("8.00")),
    (Decimal("100"), "DE", Decimal("19.00")),
    (Decimal("100"), "JP", Decimal("10.00")),
    (Decimal("100"), "XX", Decimal("0")),      # Unknown country
    (Decimal("0"), "US", Decimal("0")),        # Zero amount
    (Decimal("99.99"), "US", Decimal("7.9992")),  # Decimal precision
])
def test_calculate_tax(amount, country, expected):
    assert calculate_tax(amount, country) == expected

# ✅ PROPERTY-BASED (for edge cases you didn't think of)
from hypothesis import given, strategies as st

@given(amount=st.decimals(min_value=0, max_value=1_000_000, places=2))
def test_tax_never_negative(amount):
    for country in ["US", "DE", "JP", "XX"]:
        assert calculate_tax(amount, country) >= 0
```

---

## 4. Anti-Pattern Warning

### ⚠️ Testing Implementation, Not Behavior

**Symptom:** Tests break when refactoring, even though behavior is unchanged.

```python
# ❌ IMPLEMENTATION-COUPLED
def test_user_creation():
    service = UserService()
    service.create_user("alice", "alice@test.com")
    
    # Testing internal state
    assert service._user_cache["alice"] is not None
    assert service._email_validator.was_called
    
# ✅ BEHAVIOR-FOCUSED
def test_user_creation():
    service = UserService()
    user = service.create_user("alice", "alice@test.com")
    
    # Testing observable outcomes
    assert user.name == "alice"
    assert user.email == "alice@test.com"
    assert service.get_user("alice") == user
```

**Rule:** Test WHAT the code does, not HOW it does it. If you can refactor internals without breaking tests, your tests are good.

---

## 5. My Stack

| Tool | Purpose |
|------|---------|
| **pytest** | Test framework (fixtures, parametrize, plugins) |
| **pytest-cov** | Coverage reporting |
| **hypothesis** | Property-based testing |
| **freezegun** | Time freezing |
| **factory_boy** | Test data factories |
| **responses / respx** | HTTP mocking |
| **testcontainers** | Real databases in Docker for integration tests |

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = 
    --strict-markers
    -ra
    --tb=short
    -q
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
filterwarnings =
    error
    ignore::DeprecationWarning
```

---

## Testing Strategy by Layer

```
┌─────────────────────────────────────────────────────┐
│                    E2E Tests                         │
│  • Critical user journeys only                       │
│  • Login → Purchase → Confirmation                   │
│  Tools: Playwright, Cypress                          │
├─────────────────────────────────────────────────────┤
│               Integration Tests                      │
│  • API endpoints with real DB                        │
│  • Service interactions                              │
│  Tools: pytest + testcontainers                      │
├─────────────────────────────────────────────────────┤
│                  Unit Tests                          │
│  • Business logic                                    │
│  • Pure functions                                    │
│  • Edge cases                                        │
│  Tools: pytest + hypothesis                          │
└─────────────────────────────────────────────────────┘
```

---

*Good tests let you move fast with confidence. Bad tests slow you down with false security.*
