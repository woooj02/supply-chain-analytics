"""
Statistical analysis module for supply chain analytics.
Includes correlation analysis, hypothesis testing, and trend analysis.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, ttest_ind, f_oneway,
    chi2_contingency, normaltest, shapiro,
)
from loguru import logger

from supply_chain_analytics.core.logger import LoggerSetup

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for supply chain data."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger("statistical_analyzer")
    
    def correlation_analysis(
        self,
        df: pd.DataFrame,
        method: str = 'pearson',
        min_correlation: float = 0.3,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix with significance testing.
        
        Args:
            df: DataFrame with numeric columns
            method: 'pearson' or 'spearman'
            min_correlation: Minimum absolute correlation to include
        
        Returns:
            DataFrame with correlation pairs, values, and p-values
        """
        numeric_df = df.select_dtypes(include=[np.number])
        cols = numeric_df.columns
        
        results = []
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                valid = numeric_df[[col1, col2]].dropna()
                
                if len(valid) < 10:
                    continue
                
                if method == 'pearson':
                    corr, p_value = pearsonr(valid[col1], valid[col2])
                else:
                    corr, p_value = spearmanr(valid[col1], valid[col2])
                
                if abs(corr) >= min_correlation:
                    results.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': round(corr, 4),
                        'p_value': round(p_value, 6),
                        'significant': p_value < 0.05,
                        'strength': self._correlation_strength(corr),
                        'sample_size': len(valid),
                    })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('correlation', key=abs, ascending=False)
        
        self.logger.info(
            f"Correlation analysis: {len(result_df)} significant pairs found"
        )
        
        return result_df
    
    def _correlation_strength(self, corr: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return 'very_strong'
        elif abs_corr >= 0.6:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        elif abs_corr >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def trend_analysis(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        period: str = 'M',
    ) -> Dict[str, Any]:
        """
        Analyze trends in time series data.
        
        Returns:
            Dictionary with trend metrics and test results
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        # Resample
        resampled = df[value_col].resample(period).mean()
        resampled = resampled.dropna()
        
        if len(resampled) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Linear regression
        x = np.arange(len(resampled))
        y = resampled.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Mann-Kendall trend test
        mk_result = self._mann_kendall_test(y)
        
        # Calculate growth rate
        growth_rate = (y[-1] - y[0]) / y[0] * 100 if y[0] != 0 else 0
        
        # Moving averages
        ma_3 = resampled.rolling(3).mean().iloc[-1]
        ma_6 = resampled.rolling(6).mean().iloc[-1] if len(resampled) >= 6 else np.nan
        
        # Volatility
        volatility = resampled.pct_change().std()
        
        # Seasonality check
        if len(resampled) >= 12:
            seasonal_strength = self._seasonal_strength(resampled, 12)
        else:
            seasonal_strength = 0
        
        results = {
            'slope': round(slope, 6),
            'intercept': round(intercept, 4),
            'r_squared': round(r_value ** 2, 4),
            'p_value': round(p_value, 6),
            'trend_significant': p_value < 0.05,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'growth_rate_pct': round(growth_rate, 2),
            'mann_kendall_stat': mk_result['statistic'],
            'mann_kendall_p_value': mk_result['p_value'],
            'mann_kendall_trend': mk_result['trend'],
            'volatility': round(volatility, 4) if not np.isnan(volatility) else 0,
            'seasonal_strength': round(seasonal_strength, 4),
            'last_moving_avg_3': round(ma_3, 2) if not pd.isna(ma_3) else None,
            'last_moving_avg_6': round(ma_6, 2) if not pd.isna(ma_6) else None,
            'data_points': len(resampled),
            'start_value': round(y[0], 2),
            'end_value': round(y[-1], 2),
        }
        
        self.logger.info(
            f"Trend analysis: {results['trend_direction']} trend "
            f"(p={results['p_value']:.4f})"
        )
        
        return results
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Mann-Kendall trend test."""
        n = len(data)
        
        if n < 3:
            return {'statistic': 0, 'p_value': 1.0, 'trend': 'no_trend'}
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance
        unique_vals, counts = np.unique(data, return_counts=True)
        tp = sum(c * (c - 1) * (2 * c + 5) for c in counts)
        
        var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18
        
        if var_s > 0:
            z = (s - np.sign(s)) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        if p_value < 0.05:
            trend = 'increasing' if s > 0 else 'decreasing'
        else:
            trend = 'no_trend'
        
        return {
            'statistic': s,
            'p_value': round(p_value, 6),
            'trend': trend,
            'z_score': round(z, 4),
        }
    
    def _seasonal_strength(self, series: pd.Series, period: int) -> float:
        """Calculate seasonal strength using variance decomposition."""
        if len(series) < 2 * period:
            return 0
        
        # Detrend
        detrended = series - series.rolling(period, center=True).mean()
        
        # Seasonal component
        seasonal = detrended.groupby(detrended.index % period).mean()
        seasonal = seasonal - seasonal.mean()
        
        # Calculate strength
        var_seasonal = np.var(seasonal)
        var_residual = np.var(detrended.dropna())
        
        if var_residual > 0:
            strength = max(0, min(1, 1 - var_seasonal / var_residual))
        else:
            strength = 0
        
        return strength
    
    def abc_analysis(
        self, df: pd.DataFrame, value_col: str, item_col: str
    ) -> pd.DataFrame:
        """
        ABC analysis for inventory classification.
        
        Class A: Top 80% of value (most important)
        Class B: Next 15% of value
        Class C: Bottom 5% of value
        """
        df = df.copy()
        
        # Calculate total value per item
        item_values = df.groupby(item_col)[value_col].sum().sort_values(ascending=False)
        total_value = item_values.sum()
        
        # Calculate cumulative percentage
        cumulative_pct = item_values.cumsum() / total_value * 100
        
        # Classify
        classifications = pd.Series('C', index=item_values.index)
        classifications[cumulative_pct <= 80] = 'A'
        classifications[(cumulative_pct > 80) & (cumulative_pct <= 95)] = 'B'
        
        result = pd.DataFrame({
            'item': item_values.index,
            'total_value': item_values.values,
            'value_pct': (item_values / total_value * 100).values,
            'cumulative_pct': cumulative_pct.values,
            'abc_class': classifications.values,
        })
        
        self.logger.info(
            f"ABC Analysis: {len(result[result['abc_class']=='A'])} A-items, "
            f"{len(result[result['abc_class']=='B'])} B-items, "
            f"{len(result[result['abc_class']=='C'])} C-items"
        )
        
        return result
    
    def hypothesis_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        test_type: str = 'anova',
    ) -> Dict[str, Any]:
        """
        Perform hypothesis testing between groups.
        
        Args:
            df: DataFrame
            group_col: Column with group labels
            value_col: Column with numeric values
            test_type: 'ttest' for 2 groups, 'anova' for 3+ groups
        """
        groups = df.groupby(group_col)[value_col].apply(list)
        
        if test_type == 'ttest' and len(groups) == 2:
            g1, g2 = groups.iloc[0], groups.iloc[1]
            stat, p_value = ttest_ind(g1, g2)
            test_name = "Independent t-test"
        
        elif test_type == 'anova':
            stat, p_value = f_oneway(*groups)
            test_name = "One-way ANOVA"
        
        else:
            return {'error': f'Invalid test: {test_type} for {len(groups)} groups'}
        
        # Effect size (Cohen's d for t-test, eta-squared for ANOVA)
        if test_type == 'ttest':
            pooled_std = np.sqrt((np.std(g1)**2 + np.std(g2)**2) / 2)
            effect_size = abs(np.mean(g1) - np.mean(g2)) / (pooled_std + 1e-10)
        else:
            grand_mean = df[value_col].mean()
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = sum((v - grand_mean)**2 for v in df[value_col])
            effect_size = ss_between / (ss_total + 1e-10)
        
        results = {
            'test': test_name,
            'statistic': round(stat, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'effect_size': round(effect_size, 4),
            'groups_tested': len(groups),
            'group_means': {str(k): round(np.mean(v), 2) for k, v in groups.items()},
        }
        
        self.logger.info(
            f"Hypothesis test ({test_name}): p={p_value:.6f}, "
            f"significant={p_value < 0.05}"
        )
        
        return results
    
    def normality_test(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data follows normal distribution."""
        data = data.dropna()
        
        if len(data) < 8:
            return {'error': 'Insufficient data'}
        
        # D'Agostino and Pearson's test
        dagostino_stat, dagostino_p = normaltest(data)
        
        # Shapiro-Wilk test (more powerful for small samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(data.sample(min(5000, len(data))))
        else:
            shapiro_stat, shapiro_p = normaltest(data)
        
        # Skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        results = {
            'dagostino_statistic': round(dagostino_stat, 4),
            'dagostino_p_value': round(dagostino_p, 6),
            'shapiro_statistic': round(shapiro_stat, 4),
            'shapiro_p_value': round(shapiro_p, 6),
            'is_normal': dagostino_p > 0.05 and shapiro_p > 0.05,
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'skewness_interpretation': self._interpret_skewness(skewness),
            'sample_size': len(data),
        }
        
        return results
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if abs(skewness) < 0.5:
            return 'approximately_symmetric'
        elif skewness > 0:
            return 'right_skewed' if skewness < 1 else 'highly_right_skewed'
        else:
            return 'left_skewed' if skewness > -1 else 'highly_left_skewed'
    
    def descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive descriptive statistics."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        stats_df = pd.DataFrame({
            'count': numeric_df.count(),
            'missing': numeric_df.isnull().sum(),
            'missing_pct': (numeric_df.isnull().sum() / len(numeric_df) * 100).round(2),
            'mean': numeric_df.mean().round(4),
            'median': numeric_df.median().round(4),
            'std': numeric_df.std().round(4),
            'min': numeric_df.min().round(4),
            'max': numeric_df.max().round(4),
            'q1': numeric_df.quantile(0.25).round(4),
            'q3': numeric_df.quantile(0.75).round(4),
            'iqr': (numeric_df.quantile(0.75) - numeric_df.quantile(0.25)).round(4),
            'skewness': numeric_df.skew().round(4),
            'kurtosis': numeric_df.kurtosis().round(4),
            'variance': numeric_df.var().round(4),
        })
        
        # Add coefficient of variation
        stats_df['cv_pct'] = (stats_df['std'] / stats_df['mean'].replace(0, np.nan) * 100).round(2)
        
        return stats_df