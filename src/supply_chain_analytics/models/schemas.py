"""
Pydantic data models for data validation and serialization.
Defines the core domain models for supply chain analytics.
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
    PositiveFloat,
    PositiveInt,
    NonNegativeInt,
)


class ProductCategory(str, Enum):
    """Product category enumeration."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD_BEVERAGE = "food_beverage"
    PHARMACEUTICALS = "pharmaceuticals"
    AUTOMOTIVE = "automotive"
    FURNITURE = "furniture"
    RAW_MATERIALS = "raw_materials"
    OTHER = "other"


class WarehouseLocation(str, Enum):
    """Warehouse location enumeration."""
    NORTHEAST = "northeast"
    SOUTHEAST = "southeast"
    MIDWEST = "midwest"
    SOUTHWEST = "southwest"
    WEST = "west"
    NORTHWEST = "northwest"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"
    REFUNDED = "refunded"


class SeverityLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class Product(BaseModel):
    """Product domain model with validation."""
    
    model_config = ConfigDict(frozen=True)
    
    product_id: UUID = Field(default_factory=uuid4)
    sku: str = Field(..., min_length=3, max_length=50, pattern=r'^[A-Z0-9\-]+$')
    name: str = Field(..., min_length=1, max_length=200)
    category: ProductCategory
    unit_price: PositiveFloat
    unit_cost: PositiveFloat
    lead_time_days: PositiveInt = Field(default=7)
    minimum_order_quantity: NonNegativeInt = Field(default=1)
    reorder_point: NonNegativeInt = Field(default=0)
    safety_stock_level: NonNegativeInt = Field(default=0)
    weight_kg: Optional[PositiveFloat] = None
    dimensions_cm: Optional[Dict[str, float]] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('unit_price')
    @classmethod
    def validate_price_vs_cost(cls, v: float, info: Any) -> float:
        """Validate that price is not less than cost."""
        if 'unit_cost' in info.data and v < info.data['unit_cost']:
            raise ValueError('Unit price cannot be less than unit cost')
        return v
    
    @property
    def margin_percentage(self) -> float:
        """Calculate profit margin percentage."""
        if self.unit_price > 0:
            return ((self.unit_price - self.unit_cost) / self.unit_price) * 100
        return 0.0


class InventorySnapshot(BaseModel):
    """Inventory level snapshot for a specific point in time."""
    
    model_config = ConfigDict(frozen=True)
    
    snapshot_id: UUID = Field(default_factory=uuid4)
    product_id: UUID
    warehouse_location: WarehouseLocation
    quantity_on_hand: NonNegativeInt
    quantity_allocated: NonNegativeInt = Field(default=0)
    quantity_in_transit: NonNegativeInt = Field(default=0)
    quantity_backordered: NonNegativeInt = Field(default=0)
    last_restock_date: Optional[date] = None
    next_expected_delivery: Optional[date] = None
    days_of_supply: Optional[float] = None
    turnover_rate: Optional[float] = None
    carrying_cost_per_unit: Optional[float] = None
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def available_quantity(self) -> int:
        """Calculate available inventory."""
        return self.quantity_on_hand - self.quantity_allocated
    
    @property
    def stockout_risk(self) -> float:
        """Calculate stockout risk score (0-1)."""
        if self.days_of_supply is None:
            return 0.5
        if self.days_of_supply <= 0:
            return 1.0
        if self.days_of_supply >= 30:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.days_of_supply / 30)))


class SalesTransaction(BaseModel):
    """Sales transaction record."""
    
    model_config = ConfigDict(frozen=True)
    
    transaction_id: UUID = Field(default_factory=uuid4)
    product_id: UUID
    customer_id: Optional[UUID] = None
    quantity: PositiveInt
    unit_price: PositiveFloat
    total_amount: PositiveFloat
    discount_amount: float = Field(default=0.0, ge=0)
    order_date: datetime = Field(default_factory=datetime.utcnow)
    ship_date: Optional[datetime] = None
    delivery_date: Optional[datetime] = None
    warehouse_location: WarehouseLocation
    order_status: OrderStatus
    payment_method: Optional[str] = None
    is_b2b: bool = Field(default=False)
    channel: str = Field(default="direct")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    
    @model_validator(mode='after')
    def validate_total(self) -> 'SalesTransaction':
        expected_total = (self.quantity * self.unit_price) - self.discount_amount
        if abs(self.total_amount - expected_total) > 0.01:
            raise ValueError(
                f'Total amount {self.total_amount} does not match '
                f'calculated total {expected_total}'
            )
        return self


class SupplierPerformance(BaseModel):
    """Supplier performance metrics."""
    
    model_config = ConfigDict(frozen=True)
    
    supplier_id: UUID
    supplier_name: str = Field(..., min_length=1)
    on_time_delivery_rate: float = Field(ge=0, le=100)
    quality_acceptance_rate: float = Field(ge=0, le=100)
    average_lead_time_days: float = Field(ge=0)
    lead_time_variance: float = Field(ge=0)
    cost_competitiveness_score: float = Field(ge=0, le=100)
    responsiveness_score: float = Field(ge=0, le=100)
    total_orders: PositiveInt
    total_value: PositiveFloat
    last_evaluated: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall performance score."""
        weights = {
            'on_time_delivery': 0.30,
            'quality': 0.25,
            'lead_time': 0.20,
            'cost': 0.15,
            'responsiveness': 0.10,
        }
        score = (
            self.on_time_delivery_rate * weights['on_time_delivery'] +
            self.quality_acceptance_rate * weights['quality'] +
            max(0, 100 - (self.average_lead_time_days * 5)) * weights['lead_time'] +
            self.cost_competitiveness_score * weights['cost'] +
            self.responsiveness_score * weights['responsiveness']
        )
        return round(score, 2)


class AnomalyDetectionResult(BaseModel):
    """Results from anomaly detection analysis."""
    
    model_config = ConfigDict(frozen=True)
    
    detection_id: UUID = Field(default_factory=uuid4)
    metric_name: str
    entity_id: str
    timestamp: datetime
    actual_value: float
    expected_value: float
    deviation_percentage: float
    z_score: float
    severity: SeverityLevel
    algorithm_used: str
    confidence_score: float = Field(ge=0, le=1)
    contributing_factors: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: Optional[str] = None
    is_acknowledged: bool = Field(default=False)
    is_resolved: bool = Field(default=False)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


class ForecastResult(BaseModel):
    """Forecast results with prediction intervals."""
    
    model_config = ConfigDict(frozen=True)
    
    forecast_id: UUID = Field(default_factory=uuid4)
    metric_name: str
    entity_id: str
    forecast_date: date
    predicted_value: float
    lower_bound_80: float
    upper_bound_80: float
    lower_bound_95: float
    upper_bound_95: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    seasonality_strength: float = Field(ge=0, le=1)
    model_mape: Optional[float] = None  # Mean Absolute Percentage Error
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @model_validator(mode='after')
    def validate_bounds(self) -> 'ForecastResult':
        if self.lower_bound_95 > self.lower_bound_80:
            raise ValueError('95% lower bound should be less than 80% lower bound')
        if self.upper_bound_95 < self.upper_bound_80:
            raise ValueError('95% upper bound should be greater than 80% upper bound')
        if not (self.lower_bound_80 <= self.predicted_value <= self.upper_bound_80):
            raise ValueError('Predicted value must be within 80% confidence interval')
        return self


class DashboardMetric(BaseModel):
    """Real-time dashboard metric."""
    
    metric_id: UUID = Field(default_factory=uuid4)
    name: str
    value: Union[float, int]
    unit: str = ""
    change_percentage: Optional[float] = None
    change_direction: Optional[str] = None  # 'up', 'down', 'stable'
    trend_data: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    current_status: str = "normal"  # 'normal', 'warning', 'critical'