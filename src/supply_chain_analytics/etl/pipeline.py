"""
ETL Pipeline orchestrator.
Manages the end-to-end data pipeline with monitoring and error handling.
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from uuid import uuid4
import time

import pandas as pd
from loguru import logger

from .extractors import BaseExtractor, MultiSourceExtractor
from .transformers import DataCleanser, DataEnricher, DataAggregator, DataNormalizer
from config.settings import settings
from supply_chain_analytics.core.logger import LoggerSetup


class ETLPipeline:
    """
    End-to-end ETL pipeline with:
    - Multi-source extraction
    - Configurable transformation chain
    - Data quality validation
    - Performance monitoring
    - Error recovery
    """
    
    def __init__(
        self,
        name: str,
        extractor: MultiSourceExtractor,
        transformations: Optional[List[Callable]] = None,
        validate: bool = True,
    ):
        self.name = name
        self.pipeline_id = str(uuid4())
        self.extractor = extractor
        self.transformations = transformations or [
            DataCleanser().transform,
            DataEnricher().transform,
        ]
        self.validate = validate
        self.logger = LoggerSetup.get_logger(f"pipeline.{name}")
        
        # Pipeline state
        self.status = "initialized"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.metrics = {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": name,
            "sources_processed": 0,
            "records_extracted": 0,
            "records_transformed": 0,
            "records_loaded": 0,
            "records_failed": 0,
            "transformations_applied": 0,
            "quality_checks_passed": 0,
            "quality_checks_failed": 0,
            "duration_seconds": 0,
        }
    
    async def run(
        self,
        extraction_config: Optional[Dict[str, Any]] = None,
        transformation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute the full ETL pipeline.
        
        Args:
            extraction_config: Configuration for extraction phase
            transformation_config: Configuration for transformation phase
        
        Returns:
            Dictionary of transformed DataFrames by source name
        """
        self.status = "running"
        self.start_time = datetime.utcnow()
        
        self.logger.info(f"Starting ETL pipeline: {self.name} (ID: {self.pipeline_id})")
        
        try:
            # Phase 1: Extract
            extracted_data = await self._extract(extraction_config or {})
            
            # Phase 2: Transform
            transformed_data = await self._transform(
                extracted_data,
                transformation_config or {},
            )
            
            # Phase 3: Validate
            if self.validate:
                validated_data = await self._validate(transformed_data)
            else:
                validated_data = transformed_data
            
            self.status = "completed"
            self.logger.info(
                f"Pipeline {self.name} completed successfully. "
                f"Processed {self.metrics['records_transformed']} records."
            )
            
            return validated_data
        
        except Exception as e:
            self.status = "failed"
            self.logger.error(f"Pipeline {self.name} failed: {e}")
            raise
        
        finally:
            self.end_time = datetime.utcnow()
            self.metrics["duration_seconds"] = (
                self.end_time - self.start_time
            ).total_seconds()
            await self._log_pipeline_run()
    
    async def _extract(
        self, config: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Execute extraction phase."""
        self.logger.info("Phase 1: Extraction")
        start = time.time()
        
        extracted = await self.extractor.extract_all(**config)
        
        self.metrics["sources_processed"] = len(extracted)
        self.metrics["records_extracted"] = sum(len(df) for df in extracted.values())
        
        extraction_time = time.time() - start
        self.logger.info(
            f"Extraction complete: {self.metrics['records_extracted']} records "
            f"in {extraction_time:.2f}s"
        )
        
        return extracted
    
    async def _transform(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> Dict[str, pd.DataFrame]:
        """Execute transformation phase."""
        self.logger.info("Phase 2: Transformation")
        start = time.time()
        
        transformed = {}
        total_transformed = 0
        
        for source_name, df in data.items():
            if df.empty:
                self.logger.warning(f"Empty DataFrame for source: {source_name}")
                transformed[source_name] = df
                continue
            
            try:
                result = df.copy()
                
                for transform_fn in self.transformations:
                    result = transform_fn(result)
                    self.metrics["transformations_applied"] += 1
                
                transformed[source_name] = result
                total_transformed += len(result)
                
            except Exception as e:
                self.logger.error(
                    f"Transformation failed for {source_name}: {e}"
                )
                self.metrics["records_failed"] += len(df)
                transformed[source_name] = df  # Return original on failure
        
        self.metrics["records_transformed"] = total_transformed
        
        transform_time = time.time() - start
        self.logger.info(
            f"Transformation complete: {total_transformed} records "
            f"in {transform_time:.2f}s"
        )
        
        return transformed
    
    async def _validate(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Execute data quality validation."""
        self.logger.info("Phase 3: Validation")
        
        for source_name, df in data.items():
            if df.empty:
                continue
            
            checks = {
                "no_duplicates": not df.duplicated().any(),
                "no_negative_quantities": (
                    'quantity' not in df.columns or
                    (df['quantity'] >= 0).all()
                ),
                "no_future_dates": (
                    'order_date' not in df.columns or
                    (pd.to_datetime(df['order_date']) <= datetime.utcnow()).all()
                ),
                "data_completeness": df.notnull().mean().mean() > 0.9,
            }
            
            for check_name, passed in checks.items():
                if passed:
                    self.metrics["quality_checks_passed"] += 1
                else:
                    self.metrics["quality_checks_failed"] += 1
                    self.logger.warning(
                        f"Quality check '{check_name}' failed for {source_name}"
                    )
        
        self.logger.info(
            f"Validation complete: {self.metrics['quality_checks_passed']} passed, "
            f"{self.metrics['quality_checks_failed']} failed"
        )
        
        return data
    
    async def _log_pipeline_run(self) -> None:
        """Log pipeline execution to database."""
        try:
            from supply_chain_analytics.database.connection import db_manager
            from supply_chain_analytics.database.models import ETLLog
            
            async with db_manager.get_async_session() as session:
                log_entry = ETLLog(
                    pipeline_name=self.name,
                    run_id=uuid4(),
                    status=self.status,
                    records_processed=self.metrics["records_transformed"],
                    records_failed=self.metrics["records_failed"],
                    records_skipped=0,
                    error_message=None if self.status == "completed" else "See logs",
                    started_at=self.start_time,
                    completed_at=self.end_time,
                    duration_seconds=self.metrics["duration_seconds"],
                    metadata={
                        "pipeline_id": self.pipeline_id,
                        "transformations": self.metrics["transformations_applied"],
                        "quality_checks_passed": self.metrics["quality_checks_passed"],
                        "quality_checks_failed": self.metrics["quality_checks_failed"],
                    },
                )
                session.add(log_entry)
                await session.commit()
        
        except Exception as e:
            self.logger.warning(f"Failed to log pipeline run: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return self.metrics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "pipeline_name": self.name,
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.metrics["duration_seconds"],
            "metrics": self.metrics,
        }