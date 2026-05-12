"""
Centralized logging configuration using loguru.
Provides structured logging with context, rotation, and multiple sinks.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from config.settings import settings


class LoggerSetup:
    """Configures and manages application logging."""
    
    _initialized = False
    
    @classmethod
    def setup(cls) -> None:
        """Initialize logging configuration."""
        if cls._initialized:
            return
        
        # Remove default handler
        logger.remove()
        
        # Console handler with colorized output
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=settings.logging.level,
            colorize=True,
            backtrace=True,
            diagnose=settings.debug,
        )
        
        # File handler with rotation
        if settings.logging.file_path:
            log_path = Path(settings.logging.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_path,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "{message} | "
                    "{extra}"
                ),
                level=settings.logging.level,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                backtrace=True,
                serialize=False,
            )
        
        # Error file handler (errors only)
        if settings.logging.file_path:
            error_log = log_path.parent / f"{log_path.stem}_errors{log_path.suffix}"
            logger.add(
                error_log,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "{message} | "
                    "{extra}"
                ),
                level="ERROR",
                rotation="50 MB",
                retention="90 days",
                backtrace=True,
                diagnose=True,
            )
        
        cls._initialized = True
        logger.info(f"Logging initialized in {settings.environment} environment")
    
    @staticmethod
    def get_logger(name: str):
        """Get a logger instance with module context."""
        return logger.bind(module=name)


# Auto-initialize on import
LoggerSetup.setup()