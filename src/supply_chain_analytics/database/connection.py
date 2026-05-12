"""
Database connection management with connection pooling.
Supports both synchronous and asynchronous operations.
"""
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Generator, AsyncGenerator

from sqlalchemy import create_engine, event, Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from loguru import logger

from config.settings import settings


class DatabaseManager:
    """Manages database connections and sessions."""
    
    _instance = None
    _engine: Optional[Engine] = None
    _async_engine: Optional[AsyncEngine] = None
    _session_factory = None
    _async_session_factory = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize database engines and session factories."""
        if cls._initialized:
            return
        
        logger.info("Initializing database connections...")
        
        try:
            # Synchronous engine
            if "sqlite" in settings.database.connection_string:
                cls._engine = create_engine(
                    settings.database.connection_string,
                    echo=settings.debug,
                )
            else:
                cls._engine = create_engine(
                    settings.database.connection_string,
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=settings.debug,
                )
            
            event.listen(cls._engine, 'connect', cls._on_connect)
            
            cls._session_factory = sessionmaker(
                bind=cls._engine,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
            
            if cls._session_factory is None:
                raise RuntimeError("Failed to create session factory")
            
            # Async engine
            if "sqlite" in settings.database.connection_string:
                cls._async_engine = create_async_engine(
                    settings.database.async_connection_string,
                    echo=settings.debug,
                )
            else:
                cls._async_engine = create_async_engine(
                    settings.database.async_connection_string,
                    poolclass=NullPool,
                    echo=settings.debug,
                )
            
            cls._async_session_factory = async_sessionmaker(
                bind=cls._async_engine,
                expire_on_commit=False,
            )
            
            cls._initialized = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            cls._initialized = False
            raise
    
    @classmethod
    def _on_connect(cls, dbapi_connection, connection_record):
        """Set connection options on connect."""
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        except Exception:
            pass
    
    @classmethod
    @contextmanager
    def get_session(cls) -> Generator[Session, None, None]:
        """Get a synchronous database session."""
        if not cls._initialized or cls._session_factory is None:
            cls.initialize()
        
        if cls._session_factory is None:
            raise RuntimeError("Database session factory is not initialized")
        
        session = cls._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @classmethod
    @asynccontextmanager
    async def get_async_session(cls) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        if not cls._initialized or cls._async_session_factory is None:
            cls.initialize()
        
        if cls._async_session_factory is None:
            raise RuntimeError("Database async session factory is not initialized")
        
        async with cls._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Async database session error: {e}")
                raise
    
    @classmethod
    def get_engine(cls) -> Engine:
        """Get the synchronous engine."""
        if not cls._initialized:
            cls.initialize()
        if cls._engine is None:
            raise RuntimeError("Database engine not initialized")
        return cls._engine
    
    @classmethod
    def get_async_engine(cls) -> AsyncEngine:
        """Get the async engine."""
        if not cls._initialized:
            cls.initialize()
        if cls._async_engine is None:
            raise RuntimeError("Async database engine not initialized")
        return cls._async_engine
    
    @classmethod
    def close_all(cls) -> None:
        """Close all connections and dispose engines."""
        logger.info("Closing all database connections...")
        
        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
        
        if cls._async_engine:
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(cls._async_engine.dispose())
            cls._async_engine = None
        
        cls._session_factory = None
        cls._async_session_factory = None
        cls._initialized = False
        
        logger.info("All database connections closed")


db_manager = DatabaseManager()