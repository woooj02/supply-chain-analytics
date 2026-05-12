"""
Database migration and schema management.
Handles table creation, indexing, and seeding with sample data.
"""
import random
from datetime import datetime, date, timedelta
from uuid import uuid4
from typing import List, Dict, Any
import asyncio

import numpy as np
from loguru import logger
from sqlalchemy import text

from .connection import db_manager
from .models import Base, Product, Warehouse, InventorySnapshot, SalesTransaction
from .models import ProductCategoryEnum, WarehouseLocationEnum


class MigrationManager:
    """Manages database schema and data migrations."""
    
    @classmethod
    def create_tables(cls, drop_existing: bool = False) -> None:
        """Create all database tables."""
        engine = db_manager.get_engine()
        
        if drop_existing:
            logger.warning("Dropping all existing tables...")
            Base.metadata.drop_all(engine)
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    
    @classmethod
    def create_indexes(cls) -> None:
        """Create additional performance indexes."""
        engine = db_manager.get_engine()
        
        additional_indexes = [
            # Composite indexes for common query patterns
            """CREATE INDEX IF NOT EXISTS ix_sales_product_status_date 
               ON sales_transactions(product_id, order_status, order_date)""",
            
            """CREATE INDEX IF NOT EXISTS ix_inventory_product_location 
               ON inventory_snapshots(product_id, warehouse_id)""",
            
            # Partial indexes for active records only
            """CREATE INDEX IF NOT EXISTS ix_products_active_only 
               ON products(id) WHERE is_active = true""",
            
            # Gin index for JSONB queries
            """CREATE INDEX IF NOT EXISTS ix_anomaly_factors_gin 
               ON anomaly_records USING gin(contributing_factors)""",
            
            # Covering index for dashboard queries
            """CREATE INDEX IF NOT EXISTS ix_sales_dashboard 
               ON sales_transactions(order_date, total_amount, quantity) 
               INCLUDE (product_id, order_status)""",
        ]
        
        with engine.connect() as conn:
            for idx_sql in additional_indexes:
                try:
                    conn.execute(text(idx_sql))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Index creation note: {e}")
        
        logger.info("Additional indexes created")
    
    @classmethod
    def seed_sample_data(cls, num_products: int = 100, days_of_history: int = 365) -> None:
        """Seed database with realistic sample data for development/testing."""
        logger.info(f"Seeding database with {num_products} products and {days_of_history} days of history...")
        
        with db_manager.get_session() as session:
            # Create warehouses
            warehouses = []
            for location in WarehouseLocationEnum:
                wh = Warehouse(
                    id=uuid4(),
                    location=location,
                    name=f"{location.value.title()} Distribution Center",
                    address=f"123 {location.value.title()} Industrial Park",
                    capacity_sqft=random.uniform(50000, 200000),
                    storage_cost_per_unit=round(random.uniform(0.05, 0.50), 4),
                )
                session.add(wh)
                warehouses.append(wh)
            session.flush()
            
            # Create products
            products = []
            categories = list(ProductCategoryEnum)
            for i in range(num_products):
                category = random.choice(categories)
                product = Product(
                    id=uuid4(),
                    sku=f"SKU-{category.value[:3].upper()}-{i+1:04d}",
                    name=f"Product {category.value.title()} {i+1}",
                    category=category,
                    unit_price=round(random.uniform(10, 1000), 2),
                    unit_cost=round(random.uniform(5, 800), 2),
                    lead_time_days=random.randint(1, 30),
                    minimum_order_quantity=random.randint(1, 10),
                    reorder_point=random.randint(10, 100),
                    safety_stock_level=random.randint(5, 50),
                    weight_kg=round(random.uniform(0.1, 50), 2),
                )
                # Ensure price >= cost
                if product.unit_price < product.unit_cost:
                    product.unit_price = product.unit_cost * random.uniform(1.1, 2.0)
                    product.unit_price = round(product.unit_price, 2)
                
                session.add(product)
                products.append(product)
            session.flush()
            
            # Create sales transactions
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_of_history)
            
            batch_size = 1000
            total_transactions = 0
            
            for day_offset in range(days_of_history):
                current_date = start_date + timedelta(days=day_offset)
                daily_transactions = []
                
                for product in random.sample(products, min(30, len(products))):
                    # Multiple transactions per product in a day
                    for _ in range(random.randint(0, 5)):
                        quantity = random.randint(1, 20)
                        unit_price = float(product.unit_price) * random.uniform(0.9, 1.1)
                        total = quantity * unit_price
                        discount = round(total * random.uniform(0, 0.15), 2) if random.random() > 0.7 else 0
                        
                        transaction = SalesTransaction(
                            id=uuid4(),
                            product_id=product.id,
                            customer_id=uuid4(),
                            quantity=quantity,
                            unit_price=round(unit_price, 2),
                            total_amount=round(total - discount, 2),
                            discount_amount=discount,
                            order_date=current_date + timedelta(
                                hours=random.randint(0, 23),
                                minutes=random.randint(0, 59)
                            ),
                            warehouse_id=random.choice(warehouses).id,
                            order_status=random.choice(["pending", "confirmed", "shipped", "delivered"]),
                            payment_method=random.choice(["credit_card", "bank_transfer", "invoice"]),
                            is_b2b=random.random() > 0.7,
                            channel=random.choice(["direct", "online", "retail", "wholesale"]),
                        )
                        daily_transactions.append(transaction)
                
                session.add_all(daily_transactions)
                total_transactions += len(daily_transactions)
                
                if total_transactions % batch_size == 0:
                    session.flush()
                    logger.debug(f"Seeded {total_transactions} transactions...")
            
            session.flush()
            
            # Create inventory snapshots (weekly)
            for week_offset in range(0, days_of_history, 7):
                snapshot_date = start_date + timedelta(days=week_offset)
                
                for product in products:
                    for warehouse in warehouses[:3]:  # Limit to first 3 warehouses for performance
                        base_stock = random.randint(50, 500)
                        snapshot = InventorySnapshot(
                            id=uuid4(),
                            product_id=product.id,
                            warehouse_id=warehouse.id,
                            quantity_on_hand=base_stock + random.randint(-20, 20),
                            quantity_allocated=random.randint(0, int(base_stock * 0.3)),
                            quantity_in_transit=random.randint(0, 50),
                            days_of_supply=round(random.uniform(5, 60), 2),
                            turnover_rate=round(random.uniform(4, 12), 2),
                            carrying_cost_per_unit=round(random.uniform(0.01, 0.50), 4),
                            recorded_at=snapshot_date,
                        )
                        session.add(snapshot)
                
                session.flush()
            
            session.commit()
            logger.info(
                f"Successfully seeded {num_products} products and "
                f"{total_transactions} transactions"
            )
    
    @classmethod
    def run_full_migration(cls, seed_data: bool = False) -> None:
        """Run complete database migration."""
        logger.info("Starting database migration...")
        
        cls.create_tables(drop_existing=True)
        cls.create_indexes()
        
        if seed_data:
            cls.seed_sample_data()
        
        logger.info("Database migration completed successfully")
    
    @classmethod
    async def verify_migration(cls) -> Dict[str, Any]:
        """Verify migration was successful by checking table counts."""
        results = {}
        
        async with db_manager.get_async_session() as session:
            tables = {
                "products": Product,
                "warehouses": Warehouse,
                "inventory_snapshots": InventorySnapshot,
                "sales_transactions": SalesTransaction,
            }
            
            for name, model in tables.items():
                from sqlalchemy import select, func as sql_func
                stmt = select(sql_func.count()).select_from(model)
                result = await session.execute(stmt)
                count = result.scalar()
                results[name] = count
                logger.info(f"Table '{name}': {count} records")
        
        return results


async def initialize_database(seed: bool = False):
    """Async wrapper for database initialization."""
    migration = MigrationManager()
    await asyncio.to_thread(migration.run_full_migration, seed_data=seed)
    return await migration.verify_migration()