"""
Data extraction layer for ETL pipeline.
Supports multiple data sources: CSV, API, database, and file systems.
"""
import asyncio
import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
import gzip
import io

import aiohttp
import pandas as pd
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from supply_chain_analytics.core.logger import LoggerSetup


class BaseExtractor(ABC):
    """Abstract base class for all data extractors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = LoggerSetup.get_logger(f"extractor.{name}")
        self.metrics = {
            "records_extracted": 0,
            "bytes_processed": 0,
            "errors": 0,
            "extraction_time_ms": 0,
        }
    
    @abstractmethod
    async def extract(self, **kwargs) -> pd.DataFrame:
        """Extract data from source and return as DataFrame."""
        pass
    
    def reset_metrics(self) -> None:
        """Reset extraction metrics."""
        self.metrics = {k: 0 for k in self.metrics}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current extraction metrics."""
        return self.metrics.copy()


class CSVExtractor(BaseExtractor):
    """Extract data from CSV files with chunking support."""
    
    def __init__(self, file_path: Path, name: str = "csv_extractor"):
        super().__init__(name)
        self.file_path = Path(file_path)
        self.chunk_size = settings.analytics.batch_size
    
    async def extract(
        self,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract data from CSV file.
        
        Args:
            columns: Specific columns to extract
            filters: Column-value pairs to filter on
            chunk_size: Number of rows per chunk
        
        Returns:
            DataFrame with extracted data
        """
        self.reset_metrics()
        start_time = datetime.utcnow()
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        self.logger.info(f"Extracting from {self.file_path}")
        
        chunk_size = chunk_size or self.chunk_size
        chunks = []
        total_rows = 0
        
        try:
            # Handle compressed files
            if self.file_path.suffix == '.gz':
                open_func = gzip.open
            else:
                open_func = open
            
            # Detect encoding
            with open_func(self.file_path, 'rt', encoding='utf-8-sig') as f:
                dialect = csv.Sniffer().sniff(f.read(8192))
                f.seek(0)
                
                for chunk in pd.read_csv(
                    f,
                    chunksize=chunk_size,
                    usecols=columns,
                    dialect=dialect,
                    low_memory=False,
                ):
                    # Apply filters if provided
                    if filters:
                        for col, val in filters.items():
                            if col in chunk.columns:
                                chunk = chunk[chunk[col] == val]
                    
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    self.metrics["records_extracted"] = total_rows
                    
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"CSV extraction error: {e}")
            raise
        
        # Combine all chunks
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
        else:
            result = pd.DataFrame()
        
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.metrics["extraction_time_ms"] = elapsed
        self.metrics["bytes_processed"] = self.file_path.stat().st_size
        
        self.logger.info(
            f"Extracted {total_rows} rows in {elapsed:.0f}ms"
        )
        
        return result
    
    async def extract_in_chunks(self, **kwargs) -> AsyncIterator[pd.DataFrame]:
        """Generator for chunked extraction (memory efficient)."""
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size, low_memory=False):
            yield chunk


class APIExtractor(BaseExtractor):
    """Extract data from REST APIs with pagination and rate limiting."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        name: str = "api_extractor",
        max_concurrent: int = 5,
    ):
        super().__init__(name)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _fetch_page(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch a single page of data with retry logic."""
        async with self._semaphore:
            session = await self._get_session()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data
    
    async def extract(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        paginate: bool = True,
        max_pages: int = 100,
    ) -> pd.DataFrame:
        """
        Extract data from API endpoint.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            paginate: Whether to handle pagination
            max_pages: Maximum pages to fetch
        
        Returns:
            DataFrame of extracted data
        """
        self.reset_metrics()
        start_time = datetime.utcnow()
        
        params = params or {}
        all_records = []
        page = 1
        
        try:
            while page <= max_pages:
                if paginate:
                    params["page"] = page
                    params["page_size"] = 500
                
                data = await self._fetch_page(endpoint, params)
                
                # Handle different response formats
                if isinstance(data, list):
                    records = data
                    has_more = False
                elif isinstance(data, dict):
                    records = data.get("results", data.get("data", []))
                    has_more = data.get("has_next", data.get("next") is not None)
                else:
                    records = []
                    has_more = False
                
                if not records:
                    break
                
                all_records.extend(records)
                self.metrics["records_extracted"] = len(all_records)
                
                if not paginate or not has_more:
                    break
                
                page += 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"API extraction error on page {page}: {e}")
            
            if not all_records:
                raise
        
        result = pd.DataFrame(all_records) if all_records else pd.DataFrame()
        
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.metrics["extraction_time_ms"] = elapsed
        
        self.logger.info(
            f"Extracted {len(all_records)} records from {endpoint} "
            f"in {elapsed:.0f}ms ({page} pages)"
        )
        
        return result
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class DatabaseExtractor(BaseExtractor):
    """Extract data from SQL databases."""
    
    def __init__(self, name: str = "database_extractor"):
        super().__init__(name)
    
    async def extract(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Extract data using SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters for parameterized queries
        
        Returns:
            DataFrame with query results
        """
        from supply_chain_analytics.database.connection import db_manager
        
        self.reset_metrics()
        start_time = datetime.utcnow()
        
        try:
            async with db_manager.get_async_session() as session:
                from sqlalchemy import text
                
                result = await session.execute(text(query), params or {})
                rows = result.fetchall()
                
                if rows:
                    columns = list(result.keys())
                    data = [dict(zip(columns, row)) for row in rows]
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame()
                
                self.metrics["records_extracted"] = len(df)
        
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Database extraction error: {e}")
            raise
        
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.metrics["extraction_time_ms"] = elapsed
        
        self.logger.info(f"Extracted {len(df)} rows from database in {elapsed:.0f}ms")
        
        return df


class MultiSourceExtractor:
    """Orchestrates extraction from multiple sources."""
    
    def __init__(self):
        self.extractors: Dict[str, BaseExtractor] = {}
        self.logger = LoggerSetup.get_logger("multi_extractor")
    
    def add_extractor(self, name: str, extractor: BaseExtractor) -> None:
        """Register an extractor."""
        self.extractors[name] = extractor
    
    async def extract_all(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Extract data from all registered sources concurrently.
        
        Returns:
            Dictionary mapping source names to DataFrames
        """
        tasks = {}
        for name, extractor in self.extractors.items():
            task = asyncio.create_task(
                extractor.extract(**kwargs.get(name, {}))
            )
            tasks[name] = task
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                self.logger.error(f"Failed to extract from {name}: {e}")
                results[name] = pd.DataFrame()
        
        total_records = sum(len(df) for df in results.values())
        self.logger.info(
            f"Multi-source extraction complete: {total_records} total records "
            f"from {len(results)} sources"
        )
        
        return results