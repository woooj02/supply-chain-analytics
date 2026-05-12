"""
Main application entry point for Supply Chain Analytics Platform.
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import asyncio
import json
from datetime import datetime
from typing import Optional

import click
from loguru import logger

from config.settings import settings
from supply_chain_analytics.core.logger import LoggerSetup
from supply_chain_analytics.database.connection import db_manager
from supply_chain_analytics.database.migrations import MigrationManager
from supply_chain_analytics.cache.redis_cache import RedisCache
from supply_chain_analytics.etl.pipeline import ETLPipeline
from supply_chain_analytics.etl.extractors import CSVExtractor, MultiSourceExtractor
from supply_chain_analytics.analytics.anomaly_detector import EnsembleAnomalyDetector
from supply_chain_analytics.analytics.forecasting import EnsembleForecaster
from supply_chain_analytics.analytics.statistical_analysis import StatisticalAnalyzer
from supply_chain_analytics.alerting.alert_manager import AlertManager
from supply_chain_analytics.dashboard.app import run_dashboard


class SupplyChainAnalyticsApp:
    def __init__(self):
        self.logger = LoggerSetup.get_logger("app")
        self.anomaly_detector = EnsembleAnomalyDetector()
        self.forecaster = EnsembleForecaster()
        self.stat_analyzer = StatisticalAnalyzer()
        self.alert_manager = AlertManager()
        self.pipeline = None
        self._running = False
        self._tasks = []

    async def initialize(self):
        self.logger.info("=" * 60)
        self.logger.info("Supply Chain Analytics Platform")
        self.logger.info("=" * 60)
        self.logger.info(f"Environment: {settings.environment}")

        try:
            db_manager.initialize()
            self.logger.info("Database OK")
        except Exception as e:
            self.logger.warning(f"Database skipped: {e}")

        try:
            await RedisCache.initialize()
            self.logger.info("Cache OK")
        except Exception as e:
            self.logger.warning(f"Cache skipped: {e}")

    async def run_etl_pipeline(self):
        self.logger.info("Starting ETL...")
        extractor = MultiSourceExtractor()
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        sample_file = data_dir / "sample_supply_chain_data.csv"
        if not sample_file.exists():
            self._create_sample_data(sample_file)
        csv_extractor = CSVExtractor(sample_file)
        extractor.add_extractor("supply_chain", csv_extractor)
        self.pipeline = ETLPipeline(name="supply_chain_etl", extractor=extractor, validate=True)
        try:
            results = await self.pipeline.run()
            self.logger.info(f"ETL done: {len(results)} sources")
        except Exception as e:
            self.logger.error(f"ETL failed: {e}")

    def _create_sample_data(self, filepath):
        import pandas as pd
        import numpy as np
        self.logger.info(f"Creating sample data: {filepath}")
        dates = pd.date_range(start="2024-01-01", periods=5000, freq="h")
        products = [f"SKU-{i:04d}" for i in range(1, 51)]
        data = {
            "order_date": np.random.choice(dates, 5000),
            "product_id": np.random.choice(products, 5000),
            "quantity": np.random.randint(1, 100, 5000),
            "unit_price": np.random.uniform(10, 500, 5000).round(2),
            "warehouse_location": np.random.choice(["northeast", "southeast", "midwest", "west"], 5000),
            "order_status": np.random.choice(["pending", "confirmed", "shipped", "delivered"], 5000, p=[0.1, 0.2, 0.3, 0.4]),
            "is_b2b": np.random.choice([True, False], 5000),
            "channel": np.random.choice(["direct", "online", "retail"], 5000),
            "quantity_on_hand": np.random.randint(0, 500, 5000),
            "quantity_allocated": np.random.randint(0, 100, 5000),
        }
        df = pd.DataFrame(data)
        df["total_amount"] = (df["quantity"] * df["unit_price"]).round(2)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Created {len(df)} rows")

    async def health_check(self):
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    async def shutdown(self):
        self.logger.info("Shutting down...")
        try:
            await RedisCache.close()
        except Exception:
            pass
        try:
            db_manager.close_all()
        except Exception:
            pass


app_instance = SupplyChainAnalyticsApp()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--seed/--no-seed", default=True)
def init(seed):
    """Initialize database tables and optionally seed with sample data."""
    async def _init():
        await app_instance.initialize()
        if seed:
            try:
                from supply_chain_analytics.database.models import Base
                from sqlalchemy import inspect
                
                engine = db_manager.get_engine()
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                
                if not tables:
                    logger.info("Creating database tables...")
                    Base.metadata.create_all(engine)
                    logger.info("Tables created successfully")
                else:
                    logger.info(f"Tables already exist: {tables}")
                
                MigrationManager.seed_sample_data(num_products=50, days_of_history=90)
                logger.info("Database seeded successfully!")
            except Exception as e:
                logger.error(f"Seed failed: {e}")
    asyncio.run(_init())


@cli.command()
def etl():
    async def _etl():
        await app_instance.initialize()
        await app_instance.run_etl_pipeline()
        await app_instance.shutdown()
    asyncio.run(_etl())


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8050)
@click.option("--debug/--no-debug", default=False)
def dashboard(host, port, debug):
    logger.info(f"Dashboard: http://{host}:{port}")
    run_dashboard(host=host, port=port, debug=debug)


@cli.command()
def health():
    async def _health():
        await app_instance.initialize()
        result = await app_instance.health_check()
        print(json.dumps(result, indent=2))
        await app_instance.shutdown()
    asyncio.run(_health())


if __name__ == "__main__":
    cli()