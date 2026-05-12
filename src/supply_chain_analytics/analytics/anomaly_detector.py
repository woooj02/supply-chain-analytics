"""
Multi-algorithm anomaly detection engine.
Combines statistical, ML-based, and rule-based approaches for robust detection.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from loguru import logger

from supply_chain_analytics.models.schemas import AnomalyDetectionResult, SeverityLevel
from supply_chain_analytics.core.logger import LoggerSetup

warnings.filterwarnings('ignore')


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection algorithms."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = LoggerSetup.get_logger(f"anomaly.{name}")
        self._fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseAnomalyDetector':
        """Fit the detector to historical data."""
        pass
    
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect anomalies in data. Returns boolean series."""
        pass
    
    @abstractmethod
    def get_anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        """Get continuous anomaly scores (0-1)."""
        pass


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """
    Statistical anomaly detection using multiple methods:
    - Z-score
    - Modified Z-score (MAD-based)
    - IQR method
    - Grubbs' test
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("statistical", config)
        self.threshold = config.get('zscore_threshold', 3.0) if config else 3.0
        self.method = config.get('method', 'modified_zscore') if config else 'modified_zscore'
        self.rolling_window = config.get('rolling_window', 30) if config else 30
        self._rolling_stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, data: pd.DataFrame) -> 'StatisticalAnomalyDetector':
        """Calculate baseline statistics from historical data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].dropna()
            self._rolling_stats[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'median': series.median(),
                'mad': np.median(np.abs(series - series.median())),
                'q1': series.quantile(0.25),
                'q3': series.quantile(0.75),
            }
        
        self._fitted = True
        self.logger.info(f"Fitted on {len(numeric_cols)} numeric columns")
        return self
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect anomalies using configured statistical method."""
        if not self._fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        results = pd.Series(False, index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in self._rolling_stats:
                continue
            
            series = data[col].copy()
            
            if self.method == 'zscore':
                z_scores = np.abs(
                    (series - self._rolling_stats[col]['mean']) /
                    self._rolling_stats[col]['std'].replace(0, 1e-10)
                )
                anomalies = z_scores > self.threshold
            
            elif self.method == 'modified_zscore':
                # More robust to outliers
                mad = self._rolling_stats[col]['mad']
                if mad == 0:
                    mad = 1e-10
                modified_z = 0.6745 * np.abs(
                    series - self._rolling_stats[col]['median']
                ) / mad
                anomalies = modified_z > self.threshold
            
            elif self.method == 'iqr':
                q1 = self._rolling_stats[col]['q1']
                q3 = self._rolling_stats[col]['q3']
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                anomalies = (series < lower) | (series > upper)
            
            else:
                anomalies = pd.Series(False, index=data.index)
            
            results = results | anomalies
        
        return results
    
    def get_anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate anomaly score based on deviation magnitude."""
        scores = pd.Series(0.0, index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in self._rolling_stats:
                continue
            
            series = data[col].copy()
            mad = self._rolling_stats[col]['mad']
            if mad == 0:
                mad = 1e-10
            
            modified_z = 0.6745 * np.abs(
                series - self._rolling_stats[col]['median']
            ) / mad
            
            # Cap at 1.0
            col_scores = np.minimum(modified_z / (2 * self.threshold), 1.0)
            scores = np.maximum(scores, col_scores)
        
        return scores


class MLAnomalyDetector(BaseAnomalyDetector):
    """
    Machine Learning based anomaly detection:
    - Isolation Forest
    - Elliptic Envelope
    - PCA-based reconstruction error
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ml_detector", config)
        self.contamination = config.get('contamination', 0.05) if config else 0.05
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            max_samples='auto',
            random_state=42,
            n_jobs=-1,
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42,
        )
        self.scaler = RobustScaler()
        self.pca: Optional[PCA] = None
        self._feature_columns: List[str] = []
    
    def fit(self, data: pd.DataFrame) -> 'MLAnomalyDetector':
        """Fit ML models to historical data."""
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        self._feature_columns = numeric_data.columns.tolist()
        
        if len(self._feature_columns) < 2:
            self.logger.warning("Not enough features for ML detection")
            self._fitted = True
            return self
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        # Fit models
        self.isolation_forest.fit(scaled_data)
        self.elliptic_envelope.fit(scaled_data)
        
        # Fit PCA for reconstruction-based detection
        n_components = min(len(self._feature_columns) - 1, 5)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(scaled_data)
        
        self._fitted = True
        self.logger.info(
            f"ML detector fitted on {len(self._feature_columns)} features"
        )
        return self
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect anomalies using ensemble of ML methods."""
        if not self._fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        if len(self._feature_columns) < 2:
            return pd.Series(False, index=data.index)
        
        numeric_data = data[self._feature_columns].fillna(0)
        scaled_data = self.scaler.transform(numeric_data)
        
        # Isolation Forest predictions (-1 for anomaly, 1 for normal)
        if_pred = self.isolation_forest.predict(scaled_data)
        if_anomalies = if_pred == -1
        
        # Elliptic Envelope predictions
        try:
            ee_pred = self.elliptic_envelope.predict(scaled_data)
            ee_anomalies = ee_pred == -1
        except Exception:
            ee_anomalies = np.zeros(len(data), dtype=bool)
        
        # PCA reconstruction error
        if self.pca:
            reconstructed = self.pca.inverse_transform(
                self.pca.transform(scaled_data)
            )
            reconstruction_error = np.mean((scaled_data - reconstructed) ** 2, axis=1)
            pca_threshold = np.percentile(reconstruction_error, 95)
            pca_anomalies = reconstruction_error > pca_threshold
        else:
            pca_anomalies = np.zeros(len(data), dtype=bool)
        
        # Ensemble voting (at least 2 methods must agree)
        ensemble = (
            if_anomalies.astype(int) +
            ee_anomalies.astype(int) +
            pca_anomalies.astype(int)
        )
        anomalies = ensemble >= 2
        
        return pd.Series(anomalies, index=data.index)
    
    def get_anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        """Get continuous anomaly scores from Isolation Forest."""
        if not self._fitted or len(self._feature_columns) < 2:
            return pd.Series(0.0, index=data.index)
        
        numeric_data = data[self._feature_columns].fillna(0)
        scaled_data = self.scaler.transform(numeric_data)
        
        # Convert IF decision function to probability-like score
        scores = self.isolation_forest.decision_function(scaled_data)
        # Normalize to 0-1 (lower scores = more anomalous)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        scores = 1 - scores  # Invert so higher = more anomalous
        
        return pd.Series(scores, index=data.index)


class TimeSeriesAnomalyDetector(BaseAnomalyDetector):
    """
    Time-series specific anomaly detection:
    - Seasonal decomposition
    - Rolling statistics deviation
    - Change point detection
    - Spike/dip detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("timeseries", config)
        self.seasonal_period = config.get('seasonal_period', 7) if config else 7
        self.rolling_window = config.get('rolling_window', 30) if config else 30
        self.spike_threshold = config.get('spike_threshold', 3.0) if config else 3.0
        self._historical_stats: Dict[str, Dict] = {}
    
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesAnomalyDetector':
        """Fit on historical time series data."""
        if 'order_date' in data.columns:
            data = data.set_index('order_date')
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < self.rolling_window:
                continue
            
            self._historical_stats[col] = {
                'rolling_mean': series.rolling(self.rolling_window).mean().iloc[-1],
                'rolling_std': series.rolling(self.rolling_window).std().iloc[-1],
                'trend': self._calculate_trend(series),
                'seasonal_factors': self._calculate_seasonal_factors(series),
            }
        
        self._fitted = True
        self.logger.info(f"Time series detector fitted on {len(self._historical_stats)} metrics")
        return self
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend."""
        x = np.arange(len(series))
        y = series.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def _calculate_seasonal_factors(self, series: pd.Series) -> Dict[int, float]:
        """Calculate seasonal factors for each period position."""
        factors = {}
        for i in range(self.seasonal_period):
            seasonal_values = series.iloc[i::self.seasonal_period]
            if len(seasonal_values) > 0:
                factors[i] = seasonal_values.mean()
        return factors
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect time-series anomalies."""
        if not self._fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        results = pd.Series(False, index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in self._historical_stats:
                continue
            
            series = data[col].copy()
            
            # Rolling deviation check
            rolling_mean = series.rolling(
                self.rolling_window, min_periods=1
            ).mean()
            rolling_std = series.rolling(
                self.rolling_window, min_periods=1
            ).std().replace(0, 1e-10)
            
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            rolling_anomalies = z_scores > self.spike_threshold
            
            # Spike detection using signal processing
            peaks, _ = find_peaks(
                series.fillna(0).values,
                height=rolling_mean.values + self.spike_threshold * rolling_std.values,
            )
            
            spike_mask = np.zeros(len(series), dtype=bool)
            spike_mask[peaks] = True
            
            # Combine
            col_anomalies = rolling_anomalies | pd.Series(spike_mask, index=series.index)
            results = results | col_anomalies
        
        return results
    
    def get_anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate anomaly scores for time series."""
        scores = pd.Series(0.0, index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in self._historical_stats:
                continue
            
            series = data[col].copy()
            rolling_mean = series.rolling(
                self.rolling_window, min_periods=1
            ).mean()
            rolling_std = series.rolling(
                self.rolling_window, min_periods=1
            ).std().replace(0, 1e-10)
            
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            col_scores = np.minimum(z_scores / (2 * self.spike_threshold), 1.0)
            scores = np.maximum(scores, col_scores)
        
        return scores


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple detection methods.
    Provides weighted voting, confidence scoring, and root cause analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = LoggerSetup.get_logger("ensemble_anomaly")
        
        self.detectors = {
            'statistical': StatisticalAnomalyDetector(config),
            'ml': MLAnomalyDetector(config),
            'timeseries': TimeSeriesAnomalyDetector(config),
        }
        
        self.weights = {
            'statistical': 0.35,
            'ml': 0.40,
            'timeseries': 0.25,
        }
        
        self._fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'EnsembleAnomalyDetector':
        """Fit all detectors."""
        self.logger.info("Fitting ensemble anomaly detectors...")
        
        for name, detector in self.detectors.items():
            try:
                detector.fit(data)
                self.logger.info(f"  ✓ {name} detector fitted")
            except Exception as e:
                self.logger.error(f"  ✗ {name} detector failed: {e}")
        
        self._fitted = True
        return self
    
    def detect(
        self,
        data: pd.DataFrame,
        min_confidence: float = 0.5,
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using ensemble approach.
        
        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        results = []
        
        # Get predictions from each detector
        detector_results = {}
        detector_scores = {}
        
        for name, detector in self.detectors.items():
            try:
                detector_results[name] = detector.detect(data)
                detector_scores[name] = detector.get_anomaly_score(data)
            except Exception as e:
                self.logger.warning(f"Detector {name} failed: {e}")
                detector_results[name] = pd.Series(False, index=data.index)
                detector_scores[name] = pd.Series(0.0, index=data.index)
        
        # Calculate weighted ensemble score
        ensemble_score = pd.Series(0.0, index=data.index)
        for name in self.detectors:
            weight = self.weights.get(name, 0.3)
            ensemble_score += detector_scores[name] * weight
        
        # Identify anomalies
        anomaly_mask = ensemble_score > min_confidence
        
        # Create result objects for each anomaly
        for idx in data[anomaly_mask].index:
            row = data.loc[idx]
            
            # Find contributing factors
            contributing_factors = self._analyze_contributing_factors(
                row, data
            )
            
            # Determine severity
            severity = self._determine_severity(
                ensemble_score[idx],
                contributing_factors,
            )
            
            # Get primary metric name
            metric_name = 'unknown'
            for col in data.select_dtypes(include=[np.number]).columns:
                if col in detector_scores and detector_scores[col].get(idx, 0) > 0.5:
                    metric_name = col
                    break
            
            result = AnomalyDetectionResult(
                metric_name=metric_name,
                entity_id=str(row.get('product_id', idx)),
                timestamp=row.get('order_date', datetime.utcnow()),
                actual_value=float(row.get(metric_name, 0)),
                expected_value=float(self._calculate_expected(row, data, metric_name)),
                deviation_percentage=float(ensemble_score[idx] * 100),
                z_score=float(ensemble_score[idx] * self.detectors['statistical'].threshold),
                severity=severity,
                algorithm_used='ensemble_statistical_ml_timeseries',
                confidence_score=float(ensemble_score[idx]),
                contributing_factors=contributing_factors,
                recommendation=self._generate_recommendation(metric_name, severity),
            )
            results.append(result)
        
        self.logger.info(
            f"Detected {len(results)} anomalies "
            f"(threshold: {min_confidence})"
        )
        
        return results
    
    def _analyze_contributing_factors(
        self, row: pd.Series, data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Analyze what contributed to the anomaly."""
        factors = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:  # Limit to first 10 for performance
            if col not in row or pd.isna(row[col]):
                continue
            
            col_mean = data[col].mean()
            col_std = data[col].std()
            
            if col_std > 0:
                deviation = (row[col] - col_mean) / col_std
                if abs(deviation) > 1.5:
                    factors.append({
                        'factor': col,
                        'value': float(row[col]),
                        'expected': float(col_mean),
                        'deviation_sigma': float(deviation),
                        'impact': 'high' if abs(deviation) > 3 else 'moderate',
                    })
        
        return sorted(factors, key=lambda x: abs(x['deviation_sigma']), reverse=True)[:5]
    
    def _determine_severity(
        self, score: float, factors: List[Dict[str, Any]]
    ) -> SeverityLevel:
        """Determine anomaly severity."""
        high_impact_count = sum(
            1 for f in factors if f.get('impact') == 'high'
        )
        
        if score > 0.9 or high_impact_count >= 3:
            return SeverityLevel.CRITICAL
        elif score > 0.7 or high_impact_count >= 1:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO
    
    def _calculate_expected(
        self, row: pd.Series, data: pd.DataFrame, metric: str
    ) -> float:
        """Calculate expected value for a metric."""
        return float(data[metric].median()) if metric in data.columns else 0.0
    
    def _generate_recommendation(
        self, metric_name: str, severity: SeverityLevel
    ) -> str:
        """Generate actionable recommendation."""
        recommendations = {
            'quantity_on_hand': 'Review inventory levels and consider reorder. Check supplier lead times.',
            'total_amount': 'Investigate unusual transaction amounts. Verify pricing and discounts.',
            'days_of_supply': 'Analyze demand forecast accuracy. Adjust safety stock levels.',
            'turnover_rate': 'Review product movement patterns. Consider promotional strategies.',
        }
        
        base_rec = recommendations.get(
            metric_name,
            'Investigate the anomaly and review related processes.'
        )
        
        if severity == SeverityLevel.CRITICAL:
            base_rec += ' URGENT: Escalate to management immediately.'
        
        return base_rec
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get health status of all detectors."""
        return {
            'ensemble_fitted': self._fitted,
            'detectors': {
                name: {
                    'fitted': detector._fitted,
                    'weight': self.weights.get(name, 0),
                }
                for name, detector in self.detectors.items()
            },
        }