# 🏗️ Design Patterns

> Reusable solutions to recurring problems in software design.

---

## 1. Senior Explanation

Design patterns aren't academic exercises — they're **communication shortcuts** between engineers.

When I say "let's use a Strategy pattern here," every senior engineer instantly understands:
- The problem shape (interchangeable algorithms)
- The solution structure (interface + implementations)
- The tradeoffs (indirection vs flexibility)

**Why it's critical:**
- **Reduces cognitive load**: Patterns are pre-validated mental models
- **Accelerates code reviews**: "This is a Factory" ends 30 minutes of explanation
- **Prevents reinvention**: You're not the first to solve this problem
- **Enables refactoring**: Patterns make code malleable, not brittle

> *"Patterns are not about code. They're about shared understanding."*

---

## 2. Real Issue & Fix

### Problem: Payment Processing with Multiple Providers

**Scenario:** E-commerce platform needs to support Stripe, PayPal, and local bank transfers. Each provider has different APIs, error handling, and retry logic.

**Initial (Bad) Approach:**
```python
def process_payment(provider: str, amount: float):
    if provider == "stripe":
        # 50 lines of Stripe-specific code
    elif provider == "paypal":
        # 50 lines of PayPal-specific code
    elif provider == "bank":
        # 50 lines of bank-specific code
    # Adding new provider = modify this function
```

**Issues:**
- Single Responsibility violation
- Open/Closed violation (must modify to extend)
- Untestable (can't mock individual providers)
- 150+ line function that does everything

### Fix: Strategy Pattern

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PaymentResult:
    success: bool
    transaction_id: str | None
    error: str | None

class PaymentProcessor(ABC):
    @abstractmethod
    def process(self, amount: float, currency: str) -> PaymentResult:
        pass

class StripeProcessor(PaymentProcessor):
    def process(self, amount: float, currency: str) -> PaymentResult:
        # Stripe-specific implementation
        return PaymentResult(success=True, transaction_id="stripe_xxx", error=None)

class PayPalProcessor(PaymentProcessor):
    def process(self, amount: float, currency: str) -> PaymentResult:
        # PayPal-specific implementation
        return PaymentResult(success=True, transaction_id="pp_xxx", error=None)

# Usage: Inject the strategy
def checkout(processor: PaymentProcessor, amount: float):
    result = processor.process(amount, "USD")
    if not result.success:
        raise PaymentError(result.error)
    return result.transaction_id
```

**Result:**
- Each provider is isolated, testable, deployable independently
- Adding new provider = new class, zero changes to existing code
- Easy to A/B test providers

---

## 3. Code Snippet: Registry Pattern for Plugin Systems

```python
from typing import Callable, TypeVar, Generic

T = TypeVar("T")

class Registry(Generic[T]):
    """Type-safe registry for dynamic component registration."""
    
    def __init__(self):
        self._registry: dict[str, T] = {}
    
    def register(self, name: str) -> Callable[[T], T]:
        def decorator(cls: T) -> T:
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> T:
        if name not in self._registry:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._registry.keys())}")
        return self._registry[name]

# Usage
processors = Registry[type[PaymentProcessor]]()

@processors.register("stripe")
class StripeProcessor(PaymentProcessor): ...

@processors.register("paypal")  
class PayPalProcessor(PaymentProcessor): ...

# Dynamic instantiation from config
processor_cls = processors.get(config["payment_provider"])
processor = processor_cls()
```

---

## 4. Anti-Pattern Warning

### ⚠️ Pattern Obsession (Patternitis)

**Symptom:** Using patterns where simple code would suffice.

```python
# OVER-ENGINEERED: Factory for a single implementation
class LoggerFactory:
    def create_logger(self) -> Logger:
        return ConsoleLogger()

# JUST DO THIS:
logger = ConsoleLogger()
```

**Rules of Thumb:**
- If you have ONE implementation, you don't need a Factory
- If behavior never changes, you don't need a Strategy
- If you can't explain WHY the pattern helps, don't use it
- Patterns add indirection — indirection has cognitive cost

> *"The best code is no code. The second best is simple code. Patterns come third."*

---

## 5. My Stack

| Tool | Purpose |
|------|---------|
| **Python ABCs** | Abstract base classes for interface definitions |
| **Pydantic** | Data validation with pattern-friendly models |
| **pytest + dependency_injector** | DI container for testing pattern implementations |
| **PlantUML / Mermaid** | Pattern visualization in docs |
| **Refactoring.Guru** | Reference for pattern refreshers |

---

## Quick Reference: Pattern Selection

```
Need to create objects?
├── Simple → Constructor
├── Complex construction → Builder
├── Family of objects → Abstract Factory
└── Single instance → Singleton (use sparingly)

Need to structure behavior?
├── Interchangeable algorithms → Strategy
├── Step-by-step process → Template Method
├── Request chain → Chain of Responsibility
└── Undo/Redo → Command

Need to structure data?
├── Tree structures → Composite
├── Add behavior dynamically → Decorator
└── Simplify interface → Facade
```

---

*Patterns are tools. Master them, then know when NOT to use them.*
