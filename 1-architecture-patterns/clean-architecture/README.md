# Clean Architecture

Principles and patterns for maintainable, testable software.

## Core Principles

### The Dependency Rule

Dependencies point **inward**. Inner layers know nothing about outer layers.

```
┌─────────────────────────────────────────────┐
│                Frameworks                    │
│  ┌─────────────────────────────────────┐    │
│  │         Interface Adapters           │    │
│  │  ┌─────────────────────────────┐    │    │
│  │  │       Application           │    │    │
│  │  │  ┌─────────────────────┐    │    │    │
│  │  │  │      Domain         │    │    │    │
│  │  │  │    (Entities)       │    │    │    │
│  │  │  └─────────────────────┘    │    │    │
│  │  └─────────────────────────────┘    │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Contains | Example |
|-------|----------|---------|
| Domain | Business entities, rules | `User`, `Order`, `validate_order()` |
| Application | Use cases, orchestration | `CreateOrderUseCase` |
| Interface | Controllers, presenters | `OrderController`, `OrderDTO` |
| Infrastructure | Frameworks, DB, external | `PostgresOrderRepository` |

## Project Structure

```
src/
├── domain/
│   ├── entities/
│   │   ├── user.py
│   │   └── order.py
│   ├── value_objects/
│   │   └── money.py
│   ├── repositories/
│   │   └── order_repository.py  # Abstract
│   └── services/
│       └── pricing_service.py
├── application/
│   ├── use_cases/
│   │   ├── create_order.py
│   │   └── get_user_orders.py
│   └── interfaces/
│       └── email_service.py  # Abstract
├── infrastructure/
│   ├── persistence/
│   │   ├── postgres_order_repository.py
│   │   └── models.py  # SQLAlchemy models
│   ├── external/
│   │   └── sendgrid_email_service.py
│   └── config/
│       └── database.py
└── presentation/
    ├── api/
    │   ├── routes/
    │   │   └── orders.py
    │   └── schemas/
    │       └── order_schema.py  # DTOs
    └── cli/
        └── commands.py
```

## Implementation

### Domain Layer

```python
# domain/entities/order.py
from dataclasses import dataclass
from datetime import datetime
from typing import List
from domain.value_objects.money import Money

@dataclass
class OrderItem:
    product_id: str
    quantity: int
    unit_price: Money

@dataclass
class Order:
    id: str
    customer_id: str
    items: List[OrderItem]
    created_at: datetime
    status: str = "pending"
    
    @property
    def total(self) -> Money:
        return sum(
            (item.unit_price * item.quantity for item in self.items),
            Money(0, "USD")
        )
    
    def confirm(self) -> None:
        if not self.items:
            raise ValueError("Cannot confirm empty order")
        self.status = "confirmed"
    
    def cancel(self) -> None:
        if self.status == "shipped":
            raise ValueError("Cannot cancel shipped order")
        self.status = "cancelled"

# domain/repositories/order_repository.py
from abc import ABC, abstractmethod
from typing import Optional, List
from domain.entities.order import Order

class OrderRepository(ABC):
    @abstractmethod
    async def save(self, order: Order) -> None:
        pass
    
    @abstractmethod
    async def find_by_id(self, order_id: str) -> Optional[Order]:
        pass
    
    @abstractmethod
    async def find_by_customer(self, customer_id: str) -> List[Order]:
        pass
```

### Application Layer

```python
# application/use_cases/create_order.py
from dataclasses import dataclass
from typing import List
from domain.entities.order import Order, OrderItem
from domain.repositories.order_repository import OrderRepository
from application.interfaces.email_service import EmailService

@dataclass
class CreateOrderRequest:
    customer_id: str
    items: List[dict]  # [{"product_id": "...", "quantity": 1, "price": 9.99}]

@dataclass
class CreateOrderResponse:
    order_id: str
    total: float
    status: str

class CreateOrderUseCase:
    def __init__(
        self,
        order_repository: OrderRepository,
        email_service: EmailService
    ):
        self._order_repo = order_repository
        self._email_service = email_service
    
    async def execute(self, request: CreateOrderRequest) -> CreateOrderResponse:
        # Create domain entity
        order_items = [
            OrderItem(
                product_id=item["product_id"],
                quantity=item["quantity"],
                unit_price=Money(item["price"], "USD")
            )
            for item in request.items
        ]
        
        order = Order(
            id=generate_id(),
            customer_id=request.customer_id,
            items=order_items,
            created_at=datetime.now()
        )
        
        # Business rules
        order.confirm()
        
        # Persist
        await self._order_repo.save(order)
        
        # Side effects
        await self._email_service.send_order_confirmation(order)
        
        return CreateOrderResponse(
            order_id=order.id,
            total=float(order.total.amount),
            status=order.status
        )
```

### Infrastructure Layer

```python
# infrastructure/persistence/postgres_order_repository.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from domain.entities.order import Order
from domain.repositories.order_repository import OrderRepository
from infrastructure.persistence.models import OrderModel

class PostgresOrderRepository(OrderRepository):
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def save(self, order: Order) -> None:
        model = self._to_model(order)
        self._session.add(model)
        await self._session.commit()
    
    async def find_by_id(self, order_id: str) -> Optional[Order]:
        result = await self._session.execute(
            select(OrderModel).where(OrderModel.id == order_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None
    
    def _to_model(self, entity: Order) -> OrderModel:
        return OrderModel(
            id=entity.id,
            customer_id=entity.customer_id,
            status=entity.status,
            items=[...],
            created_at=entity.created_at
        )
    
    def _to_entity(self, model: OrderModel) -> Order:
        return Order(
            id=model.id,
            customer_id=model.customer_id,
            status=model.status,
            items=[...],
            created_at=model.created_at
        )
```

### Presentation Layer

```python
# presentation/api/routes/orders.py
from fastapi import APIRouter, Depends
from presentation.api.schemas.order_schema import CreateOrderDTO, OrderResponseDTO
from application.use_cases.create_order import CreateOrderUseCase, CreateOrderRequest

router = APIRouter(prefix="/orders", tags=["orders"])

@router.post("/", response_model=OrderResponseDTO, status_code=201)
async def create_order(
    dto: CreateOrderDTO,
    use_case: CreateOrderUseCase = Depends(get_create_order_use_case)
):
    request = CreateOrderRequest(
        customer_id=dto.customer_id,
        items=[item.dict() for item in dto.items]
    )
    
    response = await use_case.execute(request)
    
    return OrderResponseDTO(
        order_id=response.order_id,
        total=response.total,
        status=response.status
    )
```

## DTO Pattern

### Purpose

DTOs (Data Transfer Objects) separate external data representation from internal domain models.

```python
# presentation/api/schemas/order_schema.py
from pydantic import BaseModel, Field
from typing import List

class OrderItemDTO(BaseModel):
    product_id: str = Field(..., description="Product identifier")
    quantity: int = Field(..., gt=0, description="Quantity to order")
    price: float = Field(..., gt=0, description="Unit price")

class CreateOrderDTO(BaseModel):
    customer_id: str
    items: List[OrderItemDTO]
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "cust_123",
                "items": [{"product_id": "prod_1", "quantity": 2, "price": 29.99}]
            }
        }

class OrderResponseDTO(BaseModel):
    order_id: str
    total: float
    status: str
```

## Dependency Injection

### Container Setup

```python
# infrastructure/container.py
from dependency_injector import containers, providers
from infrastructure.persistence.postgres_order_repository import PostgresOrderRepository
from infrastructure.external.sendgrid_email_service import SendgridEmailService
from application.use_cases.create_order import CreateOrderUseCase

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Database session
    db_session = providers.Factory(
        get_async_session,
        database_url=config.database_url
    )
    
    # Repositories
    order_repository = providers.Factory(
        PostgresOrderRepository,
        session=db_session
    )
    
    # External services
    email_service = providers.Singleton(
        SendgridEmailService,
        api_key=config.sendgrid_api_key
    )
    
    # Use cases
    create_order_use_case = providers.Factory(
        CreateOrderUseCase,
        order_repository=order_repository,
        email_service=email_service
    )

# main.py
container = Container()
container.config.from_yaml("config.yaml")

app = FastAPI()

@app.on_event("startup")
def setup():
    container.wire(modules=[presentation.api.routes.orders])
```

## Testing

### Unit Testing Use Cases

```python
# tests/application/test_create_order.py
import pytest
from unittest.mock import AsyncMock
from application.use_cases.create_order import CreateOrderUseCase, CreateOrderRequest

@pytest.fixture
def mock_order_repo():
    return AsyncMock()

@pytest.fixture
def mock_email_service():
    return AsyncMock()

@pytest.mark.asyncio
async def test_create_order_success(mock_order_repo, mock_email_service):
    use_case = CreateOrderUseCase(mock_order_repo, mock_email_service)
    
    request = CreateOrderRequest(
        customer_id="cust_123",
        items=[{"product_id": "prod_1", "quantity": 2, "price": 29.99}]
    )
    
    response = await use_case.execute(request)
    
    assert response.status == "confirmed"
    assert response.total == 59.98
    mock_order_repo.save.assert_called_once()
    mock_email_service.send_order_confirmation.assert_called_once()
```

## Related Resources

- [Web Frameworks](../../5-python-production/web-frameworks/README.md) - FastAPI implementation
- [Docker](../../7-infrastructure/docker/README.md) - Containerizing clean architecture apps
- [Testing](../../5-python-production/testing/README.md) - Testing strategies
