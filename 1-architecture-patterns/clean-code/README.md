# ✨ Clean Code

> Code is read 10x more than it's written. Optimize for the reader.

---

## 1. Senior Explanation

Clean code isn't about aesthetics — it's about **reducing the cost of change**.

**The economics:**
- Feature development: 20% of engineering time
- Reading/understanding existing code: 60%
- Debugging: 20%

Clean code attacks the 60%. When code is clear:
- Onboarding is faster
- Bugs are found in code review, not production
- Refactoring is safe
- Technical debt compounds slower

**What "clean" actually means:**
- **Readable**: Intent is obvious without comments
- **Predictable**: No surprises, no magic
- **Testable**: Dependencies are explicit and mockable
- **Changeable**: Modifications don't cascade

> *"Any fool can write code that a computer can understand. Good programmers write code that humans can understand." — Martin Fowler*

---

## 2. Real Issue & Fix

### Problem: The God Function

**Scenario:** Order processing function that grew over 2 years to 400 lines.

```python
def process_order(order_data: dict) -> dict:
    # Validate order (50 lines)
    # Check inventory (40 lines)
    # Calculate pricing (60 lines)
    # Apply discounts (45 lines)
    # Process payment (55 lines)
    # Update inventory (30 lines)
    # Send notifications (40 lines)
    # Generate invoice (50 lines)
    # Log analytics (30 lines)
    return result
```

**Issues:**
- Untestable (can't test pricing without payment)
- One bug = entire function under suspicion
- Two devs can't work on it simultaneously
- Changes ripple unpredictably

### Fix: Extract Till You Drop

```python
@dataclass
class OrderContext:
    order: Order
    user: User
    inventory: InventorySnapshot
    pricing: PricingResult | None = None
    payment: PaymentResult | None = None

class OrderProcessor:
    def __init__(
        self,
        validator: OrderValidator,
        inventory: InventoryService,
        pricing: PricingEngine,
        payments: PaymentProcessor,
        notifications: NotificationService,
    ):
        self.validator = validator
        self.inventory = inventory
        self.pricing = pricing
        self.payments = payments
        self.notifications = notifications
    
    def process(self, order: Order, user: User) -> OrderResult:
        ctx = OrderContext(order=order, user=user, inventory=self.inventory.snapshot())
        
        self.validator.validate(ctx)
        ctx.pricing = self.pricing.calculate(ctx)
        ctx.payment = self.payments.charge(ctx)
        self.inventory.reserve(ctx)
        self.notifications.send_confirmation(ctx)
        
        return OrderResult.from_context(ctx)
```

**Result:**
- Each component is testable in isolation
- Clear dependency graph
- New developer understands flow in 30 seconds
- Add new step = add one line + one class

---

## 3. Code Snippet: Self-Documenting Code

```python
# ❌ BEFORE: Comments explain what, not why
def calc(u, t):
    # Check if user is premium
    if u.type == 1:
        # Apply 20% discount
        d = t * 0.2
        return t - d
    # Check if user is VIP
    elif u.type == 2:
        # Apply 30% discount
        d = t * 0.3
        return t - d
    return t

# ✅ AFTER: Code explains itself
class UserTier(Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    VIP = "vip"

TIER_DISCOUNTS: dict[UserTier, Decimal] = {
    UserTier.STANDARD: Decimal("0"),
    UserTier.PREMIUM: Decimal("0.20"),
    UserTier.VIP: Decimal("0.30"),
}

def apply_tier_discount(user: User, total: Decimal) -> Decimal:
    discount_rate = TIER_DISCOUNTS.get(user.tier, Decimal("0"))
    discount_amount = total * discount_rate
    return total - discount_amount
```

**What changed:**
- Magic numbers → Named constants
- Cryptic names → Intention-revealing names
- Type codes → Enums
- Comments → Code that doesn't need them

---

## 4. Anti-Pattern Warning

### ⚠️ Comment-Driven Development

**Symptom:** Comments that describe WHAT instead of WHY.

```python
# ❌ USELESS COMMENT
# Increment counter by 1
counter += 1

# ❌ LIES (code changed, comment didn't)
# Send email to user
send_sms(user.phone, message)

# ✅ USEFUL: Explains business rule
# Retry 3 times because payment gateway has 2% transient failure rate
for attempt in range(3):
    ...

# ✅ USEFUL: Explains non-obvious decision
# Using insertion sort here because n < 10 and it's faster for small arrays
```

**Rule:** If you need a comment to explain WHAT the code does, refactor the code. Comments should explain WHY or warn about non-obvious behavior.

---

## 5. My Stack

| Tool | Purpose |
|------|---------|
| **Ruff** | Lightning-fast linter + formatter (replaces Black, isort, flake8) |
| **mypy / Pyright** | Static type checking |
| **pre-commit** | Enforce standards before commit |
| **SonarQube** | Code smell detection at scale |
| **Sourcery** | AI-powered refactoring suggestions |

### Pre-commit Config

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
```

---

## Clean Code Checklist

```
□ Function does ONE thing
□ Function fits on one screen (~20 lines)
□ No more than 3 parameters (use objects for more)
□ No flag arguments (split into two functions)
□ No side effects hidden in getters
□ Error handling doesn't obscure logic
□ Names reveal intent
□ Abstractions are at consistent level
□ DRY but not WET (Write Everything Twice first)
```

---

*Clean code is not a destination. It's a practice.*
