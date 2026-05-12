"""
Central configuration module for Supply Chain Analytics Platform.
Loads settings from environment variables with validation.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


@dataclass
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "supply_chain_analytics"))
    username: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "admin"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    @property
    def connection_string(self) -> str:
        if os.getenv("USE_SQLITE", "true").lower() == "true":
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "supply_chain.db"
            db_path.parent.mkdir(exist_ok=True)
            return f"sqlite:///{db_path}"
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_string(self) -> str:
        if os.getenv("USE_SQLITE", "true").lower() == "true":
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "supply_chain.db"
            db_path.parent.mkdir(exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis cache configuration."""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    
    @property
    def connection_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class ApiConfig:
    """External API configurations."""
    erp_endpoint: str = field(default_factory=lambda: os.getenv("ERP_API_ENDPOINT", ""))
    erp_api_key: str = field(default_factory=lambda: os.getenv("ERP_API_KEY", ""))
    warehouse_endpoint: str = field(default_factory=lambda: os.getenv("WAREHOUSE_API_ENDPOINT", ""))
    warehouse_api_key: str = field(default_factory=lambda: os.getenv("WAREHOUSE_API_KEY", ""))


@dataclass
class AnalyticsConfig:
    """Analytics engine configuration."""
    anomaly_zscore_threshold: float = field(
        default_factory=lambda: float(os.getenv("ANOMALY_THRESHOLD_ZSCORE", "3.0"))
    )
    forecast_horizon_days: int = field(
        default_factory=lambda: int(os.getenv("FORECAST_HORIZON_DAYS", "30"))
    )
    realtime_window_hours: int = field(
        default_factory=lambda: int(os.getenv("REALTIME_WINDOW_HOURS", "48"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "10000"))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("MAX_WORKERS", "4"))
    )
    confidence_level: float = 0.95
    seasonality_periods: Dict[str, int] = field(default_factory=lambda: {
        "daily": 7,
        "weekly": 52,
        "monthly": 12,
    })


@dataclass
class AlertConfig:
    """Alert and notification configuration."""
    slack_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL")
    )
    email_smtp_host: str = field(default_factory=lambda: os.getenv("EMAIL_SMTP_HOST", ""))
    email_smtp_port: int = field(default_factory=lambda: int(os.getenv("EMAIL_SMTP_PORT", "587")))
    email_from: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))
    email_to: str = field(default_factory=lambda: os.getenv("EMAIL_TO", ""))
    alert_cooldown_minutes: int = 15
    max_alerts_per_hour: int = 20


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file_path: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))


class Settings:
    """Central settings container."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = ApiConfig()
        self.analytics = AnalyticsConfig()
        self.alert = AlertConfig()
        self.logging = LoggingConfig()
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = self.environment == "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary (excluding sensitive data)."""
        return {
            "environment": self.environment,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
            },
            "analytics": {
                "anomaly_zscore_threshold": self.analytics.anomaly_zscore_threshold,
                "forecast_horizon_days": self.analytics.forecast_horizon_days,
                "batch_size": self.analytics.batch_size,
            },
        }


# Global settings instance
settings = Settings()
