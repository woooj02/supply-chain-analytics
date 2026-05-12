"""
Data transformation layer for ETL pipeline.
Handles data cleansing, normalization, enrichment, and feature engineering.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
import re

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from supply_chain_analytics.core.logger import LoggerSetup


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = LoggerSetup.get_logger(f"transformer.{name}")
        self.transformations_applied: List[str] = []
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to DataFrame."""
        pass
    
    def log_transformation(self, name: str) -> None:
        """Log applied transformation."""
        self.transformations_applied.append(name)
        self.logger.debug(f"Applied transformation: {name}")


class DataCleanser(BaseTransformer):
    """Cleanse and standardize raw data."""
    
    def __init__(self):
        super().__init__("data_cleanser")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive data cleansing.
        
        Steps:
        1. Remove duplicate rows
        2. Handle missing values
        3. Remove outliers
        4. Standardize column names
        5. Fix data types
        """
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        self.log_transformation(f"remove_duplicates ({duplicates_removed} removed)")
        
        # Standardize column names
        df.columns = [
            re.sub(r'[^a-z0-9_]', '_', col.lower().strip())
            for col in df.columns
        ]
        self.log_transformation("standardize_column_names")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = self._remove_outliers(df, col)
        
        # Fix data types
        df = self._fix_data_types(df)
        
        self.logger.info(
            f"Data cleansing: {initial_count} -> {len(df)} rows "
            f"({initial_count - len(df)} removed)"
        )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value handling."""
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            
            if missing_pct == 0:
                continue
            
            if missing_pct > 0.5:
                # Drop column if >50% missing
                df = df.drop(columns=[col])
                self.log_transformation(f"drop_column_{col} (missing={missing_pct:.1%})")
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Use median for numeric columns
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.log_transformation(f"fill_median_{col}")
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Forward fill for datetime
                df[col] = df[col].fillna(method='ffill')
                self.log_transformation(f"fill_forward_{col}")
            
            else:
                # Mode for categorical, 'UNKNOWN' for strings
                if df[col].nunique() < 20:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna('UNKNOWN')
                self.log_transformation(f"fill_mode_or_unknown_{col}")
        
        return df
    
    def _remove_outliers(
        self, df: pd.DataFrame, col: str, method: str = 'iqr'
    ) -> pd.DataFrame:
        """Remove statistical outliers."""
        if df[col].nunique() < 5:
            return df
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            before = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed = before - len(df)
            
            if removed > 0:
                self.log_transformation(
                    f"remove_outliers_{col}_iqr ({removed} removed)"
                )
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            before = len(df)
            df = df[z_scores < 3.0]
            removed = before - len(df)
            
            if removed > 0:
                self.log_transformation(
                    f"remove_outliers_{col}_zscore ({removed} removed)"
                )
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common data type issues."""
        for col in df.columns:
            # Try to convert to numeric
            if df[col].dtype == 'object':
                try:
                    # Check if column looks like it should be numeric
                    sample = df[col].dropna().head(100)
                    numeric_count = sum(
                        1 for v in sample
                        if isinstance(v, str) and re.match(r'^-?\d+\.?\d*$', v.strip())
                    )
                    if numeric_count / len(sample) > 0.8:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        self.log_transformation(f"convert_numeric_{col}")
                except Exception:
                    pass
            
            # Try to convert to datetime
            if df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().head(10)
                    if any(isinstance(v, str) and re.match(r'\d{4}-\d{2}-\d{2}', v) for v in sample):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        self.log_transformation(f"convert_datetime_{col}")
                except Exception:
                    pass
        
        return df


class DataEnricher(BaseTransformer):
    """Enrich data with derived features and external data."""
    
    def __init__(self):
        super().__init__("data_enricher")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and enrichments."""
        df = df.copy()
        
        # Add date-based features if date columns exist
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df = self._add_date_features(df, col)
        
        # Add financial metrics if relevant columns exist
        if all(c in df.columns for c in ['unit_price', 'quantity']):
            df = self._add_financial_features(df)
        
        # Add inventory metrics
        if all(c in df.columns for c in ['quantity_on_hand', 'quantity_allocated']):
            df = self._add_inventory_features(df)
        
        # Add lag features for time series
        if 'order_date' in df.columns and 'total_amount' in df.columns:
            df = self._add_lag_features(df)
        
        return df
    
    def _add_date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add temporal features from date column."""
        prefix = date_col.replace('_date', '')
        
        df[f'{prefix}_year'] = df[date_col].dt.year
        df[f'{prefix}_month'] = df[date_col].dt.month
        df[f'{prefix}_quarter'] = df[date_col].dt.quarter
        df[f'{prefix}_day_of_week'] = df[date_col].dt.dayofweek
        df[f'{prefix}_day_of_month'] = df[date_col].dt.day
        df[f'{prefix}_week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
        df[f'{prefix}_is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        df[f'{prefix}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df[f'{prefix}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        self.log_transformation(f"add_date_features_{date_col}")
        return df
    
    def _add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add financial metrics."""
        df['revenue'] = df['unit_price'] * df['quantity']
        df['transaction_year_month'] = df['order_date'].dt.to_period('M').astype(str)
        
        # Price analysis
        if 'product_id' in df.columns:
            df['avg_price_by_product'] = df.groupby('product_id')['unit_price'].transform('mean')
            df['price_vs_avg'] = (df['unit_price'] - df['avg_price_by_product']) / df['avg_price_by_product'] * 100
        
        self.log_transformation("add_financial_features")
        return df
    
    def _add_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add inventory-specific metrics."""
        df['available_quantity'] = df['quantity_on_hand'] - df['quantity_allocated']
        df['allocation_rate'] = df['quantity_allocated'] / df['quantity_on_hand'].replace(0, np.nan)
        df['stock_coverage_days'] = df['quantity_on_hand'] / (
            df.groupby('product_id')['quantity'].transform('mean').replace(0, np.nan)
        )
        
        self.log_transformation("add_inventory_features")
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series analysis."""
        df = df.sort_values(['product_id', 'order_date'])
        
        for lag in [1, 7, 30]:
            df[f'revenue_lag_{lag}d'] = df.groupby('product_id')['total_amount'].shift(lag)
            df[f'quantity_lag_{lag}d'] = df.groupby('product_id')['quantity'].shift(lag)
        
        self.log_transformation("add_lag_features")
        return df


class DataAggregator(BaseTransformer):
    """Aggregate data at different granularities."""
    
    def __init__(self):
        super().__init__("data_aggregator")
    
    def transform(
        self,
        df: pd.DataFrame,
        group_by: List[str],
        agg_dict: Dict[str, Any],
        time_freq: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate data with specified groupings.
        
        Args:
            df: Input DataFrame
            group_by: Columns to group by
            agg_dict: Aggregation functions mapping
            time_freq: Resample frequency if time-based (e.g., 'D', 'W', 'M')
        
        Returns:
            Aggregated DataFrame
        """
        df = df.copy()
        
        if time_freq and 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'])
            df = df.set_index('order_date')
            
            result = df.groupby(group_by).resample(time_freq).agg(agg_dict)
            result = result.reset_index()
            
            self.log_transformation(f"time_aggregation_{time_freq}")
        else:
            result = df.groupby(group_by).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip('_') for col in result.columns]
        
        self.log_transformation(f"aggregate_by_{'_'.join(group_by)}")
        
        return result


class DataNormalizer(BaseTransformer):
    """Normalize and scale numeric features."""
    
    def __init__(self):
        super().__init__("data_normalizer")
    
    def transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'minmax',
    ) -> pd.DataFrame:
        """
        Normalize numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize (None = all numeric)
            method: Normalization method ('minmax', 'zscore', 'robust')
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[f'{col}_normalized'] = (df[col] - mean_val) / std_val
            
            elif method == 'robust':
                median_val = df[col].median()
                iqr_val = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr_val > 0:
                    df[f'{col}_normalized'] = (df[col] - median_val) / iqr_val
        
        self.log_transformation(f"normalize_{method}")
        return df