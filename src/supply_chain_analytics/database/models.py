"""
SQLAlchemy ORM models for supply chain analytics.
Defines the database schema with indexes, constraints, and relationships.
"""
from datetime import datetime, date
from uuid import uuid4
import enum

from sqlalchemy import (

    Column, String, Integer, Float, DateTime, Date, Boolean,
    Enum, ForeignKey, Text, Index, UniqueConstraint, CheckConstraint,
    Numeric, BigInteger, JSON, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class ProductCategoryEnum(str, enum.Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD_BEVERAGE = "food_beverage"
    PHARMACEUTICALS = "pharmaceuticals"
    AUTOMOTIVE = "automotive"
    FURNITURE = "furniture"
    RAW_MATERIALS = "raw_materials"
    OTHER = "other"


class WarehouseLocationEnum(str, enum.Enum):
    NORTHEAST = "northeast"
    SOUTHEAST = "southeast"
    MIDWEST = "midwest"
    SOUTHWEST = "southwest"
    WEST = "west"
    NORTHWEST = "northwest"


class Product(Base):
    """Products table with all product-related information."""
    
    __tablename__ = "products"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sku = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    category = Column(Enum(ProductCategoryEnum), nullable=False, index=True)
    unit_price = Column(Numeric(12, 2), nullable=False)
    unit_cost = Column(Numeric(12, 2), nullable=False)
    lead_time_days = Column(Integer, default=7)
    minimum_order_quantity = Column(Integer, default=1)
    reorder_point = Column(Integer, default=0)
    safety_stock_level = Column(Integer, default=0)
    weight_kg = Column(Float, nullable=True)
    dimensions_cm = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Relationships
    inventory_snapshots = relationship("InventorySnapshot", back_populates="product", lazy="dynamic")
    sales_transactions = relationship("SalesTransaction", back_populates="product", lazy="dynamic")
    
    __table_args__ = (
        Index("ix_products_category_active", "category", "is_active"),
        CheckConstraint("unit_price >= unit_cost", name="ck_price_vs_cost"),
    )
    
    @hybrid_property
    def margin_percentage(self):
        if self.unit_price and self.unit_price > 0:
            return ((self.unit_price - self.unit_cost) / self.unit_price) * 100
        return 0.0
    
    def __repr__(self):
        return f"<Product(sku='{self.sku}', name='{self.name}')>"


class Warehouse(Base):
    """Warehouse locations table."""
    
    __tablename__ = "warehouses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    location = Column(Enum(WarehouseLocationEnum), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    address = Column(String(300))
    capacity_sqft = Column(Float)
    storage_cost_per_unit = Column(Numeric(10, 4))
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    inventory_snapshots = relationship("InventorySnapshot", back_populates="warehouse", lazy="dynamic")
    
    def __repr__(self):
        return f"<Warehouse(location='{self.location}', name='{self.name}')>"


class InventorySnapshot(Base):
    """Inventory snapshots for tracking stock levels over time."""
    
    __tablename__ = "inventory_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    warehouse_id = Column(UUID(as_uuid=True), ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False)
    quantity_on_hand = Column(Integer, nullable=False)
    quantity_allocated = Column(Integer, default=0)
    quantity_in_transit = Column(Integer, default=0)
    quantity_backordered = Column(Integer, default=0)
    last_restock_date = Column(Date, nullable=True)
    next_expected_delivery = Column(Date, nullable=True)
    days_of_supply = Column(Float, nullable=True)
    turnover_rate = Column(Float, nullable=True)
    carrying_cost_per_unit = Column(Numeric(10, 4), nullable=True)
    recorded_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Relationships
    product = relationship("Product", back_populates="inventory_snapshots")
    warehouse = relationship("Warehouse", back_populates="inventory_snapshots")
    
    __table_args__ = (
        Index("ix_inventory_product_warehouse_time", "product_id", "warehouse_id", "recorded_at"),
        Index("ix_inventory_recorded_at", "recorded_at", postgresql_using="brin"),
    )
    
    @hybrid_property
    def available_quantity(self):
        return self.quantity_on_hand - self.quantity_allocated
    
    def __repr__(self):
        return f"<InventorySnapshot(product={self.product_id}, qty={self.quantity_on_hand})>"


class SalesTransaction(Base):
    """Sales transactions table."""
    
    __tablename__ = "sales_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=True)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Numeric(12, 2), nullable=False)
    total_amount = Column(Numeric(14, 2), nullable=False)
    discount_amount = Column(Numeric(14, 2), default=0)
    order_date = Column(DateTime(timezone=True), default=func.now(), index=True)
    ship_date = Column(DateTime(timezone=True), nullable=True)
    delivery_date = Column(DateTime(timezone=True), nullable=True)
    warehouse_id = Column(UUID(as_uuid=True), ForeignKey("warehouses.id", ondelete="SET NULL"))
    order_status = Column(String(20), default="pending", index=True)
    payment_method = Column(String(50))
    is_b2b = Column(Boolean, default=False)
    channel = Column(String(50), default="direct")
    currency = Column(String(3), default="USD")
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    product = relationship("Product", back_populates="sales_transactions")
    
    __table_args__ = (
        Index("ix_sales_order_date", "order_date", postgresql_using="brin"),
        Index("ix_sales_product_date", "product_id", "order_date"),
        Index("ix_sales_status", "order_status"),
    )
    
    def __repr__(self):
        return f"<SalesTransaction(id={self.id}, amount={self.total_amount})>"


class SupplierPerformance(Base):
    """Supplier performance metrics table."""
    
    __tablename__ = "supplier_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    supplier_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    supplier_name = Column(String(200), nullable=False)
    on_time_delivery_rate = Column(Float, nullable=False)
    quality_acceptance_rate = Column(Float, nullable=False)
    average_lead_time_days = Column(Float, nullable=False)
    lead_time_variance = Column(Float, nullable=False)
    cost_competitiveness_score = Column(Float, nullable=False)
    responsiveness_score = Column(Float, nullable=False)
    total_orders = Column(Integer, nullable=False)
    total_value = Column(Numeric(14, 2), nullable=False)
    last_evaluated = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    __table_args__ = (
        Index("ix_supplier_evaluated", "supplier_id", "last_evaluated"),
    )
    
    def __repr__(self):
        return f"<SupplierPerformance(supplier='{self.supplier_name}')>"


class AnomalyRecord(Base):
    """Anomaly detection results storage."""
    
    __tablename__ = "anomaly_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    entity_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    actual_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    deviation_percentage = Column(Float, nullable=False)
    z_score = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    algorithm_used = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    contributing_factors = Column(JSON, default=[])
    recommendation = Column(Text)
    is_acknowledged = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    detected_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    __table_args__ = (
        Index("ix_anomaly_timestamp", "detected_at", postgresql_using="brin"),
        Index("ix_anomaly_metric_time", "metric_name", "detected_at"),
        Index("ix_anomaly_severity_resolved", "severity", "is_resolved"),
    )
    
    def __repr__(self):
        return f"<AnomalyRecord(metric='{self.metric_name}', severity='{self.severity}')>"


class ForecastRecord(Base):
    """Forecast results storage."""
    
    __tablename__ = "forecast_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    entity_id = Column(String(100), nullable=False)
    forecast_date = Column(Date, nullable=False)
    predicted_value = Column(Float, nullable=False)
    lower_bound_80 = Column(Float, nullable=False)
    upper_bound_80 = Column(Float, nullable=False)
    lower_bound_95 = Column(Float, nullable=False)
    upper_bound_95 = Column(Float, nullable=False)
    trend_direction = Column(String(20), nullable=False)
    seasonality_strength = Column(Float, nullable=False)
    model_mape = Column(Float, nullable=True)
    generated_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    __table_args__ = (
        Index("ix_forecast_metric_date", "metric_name", "forecast_date"),
        UniqueConstraint("metric_name", "entity_id", "forecast_date", "generated_at",
                        name="uq_forecast_record"),
    )
    
    def __repr__(self):
        return f"<ForecastRecord(metric='{self.metric_name}', date={self.forecast_date})>"


class ETLLog(Base):
    """ETL pipeline execution logs."""
    
    __tablename__ = "etl_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    pipeline_name = Column(String(100), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # 'running', 'completed', 'failed'
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    records_skipped = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    run_metadata = Column(JSON, default={})
    
    __table_args__ = (
        Index("ix_etl_pipeline_run", "pipeline_name", "run_id"),
    )
    
    def __repr__(self):
        return f"<ETLLog(pipeline='{self.pipeline_name}', status='{self.status}')>"


class DataQualityCheck(Base):
    """Data quality check results."""
    
    __tablename__ = "data_quality_checks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    table_name = Column(String(100), nullable=False, index=True)
    check_name = Column(String(100), nullable=False)
    check_type = Column(String(50), nullable=False)  # 'completeness', 'uniqueness', 'validity', 'consistency'
    passed = Column(Boolean, nullable=False)
    total_records = Column(Integer, default=0)
    failed_records = Column(Integer, default=0)
    failure_rate = Column(Float)
    details = Column(JSON)
    checked_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    __table_args__ = (
        Index("ix_dq_table_check_time", "table_name", "check_name", "checked_at"),
    )
    
    def __repr__(self):
        return f"<DataQualityCheck(table='{self.table_name}', passed={self.passed})>"