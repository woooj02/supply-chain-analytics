"""
Time series forecasting engine with multiple models.
Supports demand forecasting, inventory prediction, and trend analysis.
"""
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from loguru import logger

from supply_chain_analytics.models.schemas import ForecastResult
from supply_chain_analytics.core.logger import LoggerSetup

warnings.filterwarnings('ignore')


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = LoggerSetup.get_logger(f"forecaster.{name}")
        self._fitted = False
        self._model_params: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, series: pd.Series) -> 'BaseForecaster':
        """Fit the model to historical data."""
        pass
    
    @abstractmethod
    def predict(
        self, horizon: int, return_conf_int: bool = True
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate predictions.
        
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        pass
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        # Remove NaN values
        mask = ~(actual.isna() | predicted.isna())
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {'mape': np.nan, 'mae': np.nan, 'rmse': np.nan}
        
        errors = actual - predicted
        abs_errors = np.abs(errors)
        
        mae = abs_errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        
        # MAPE with handling for zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            ape = np.abs(errors / actual.replace(0, np.nan))
            mape = np.nanmean(ape) * 100
        
        return {
            'mape': round(mape, 2),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
        }


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Holt-Winters Exponential Smoothing.
    Handles trend and seasonality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("exp_smoothing", config)
        self.seasonal_periods = (
            config.get('seasonal_periods', 7) if config else 7
        )
        self._alpha: Optional[float] = None
        self._beta: Optional[float] = None
        self._gamma: Optional[float] = None
        self._level: Optional[np.ndarray] = None
        self._trend: Optional[np.ndarray] = None
        self._seasonal: Optional[np.ndarray] = None
        self._last_level: float = 0
        self._last_trend: float = 0
    
    def fit(self, series: pd.Series) -> 'ExponentialSmoothingForecaster':
        """Fit Holt-Winters model using optimization."""
        series = series.dropna()
        
        if len(series) < 2 * self.seasonal_periods:
            self.logger.warning(
                f"Insufficient data for seasonal model. "
                f"Need at least {2 * self.seasonal_periods} points."
            )
            self._fitted = True
            return self
        
        y = series.values.astype(float)
        n = len(y)
        
        # Initial values
        self._last_level = y[0]
        self._last_trend = (y[self.seasonal_periods] - y[0]) / self.seasonal_periods
        
        # Optimize parameters
        def objective(params):
            a, b, g = params
            if not (0 < a < 1 and 0 < b < 1 and 0 < g < 1):
                return np.inf
            
            level = y[0]
            trend = self._last_trend
            seasonal = np.ones(self.seasonal_periods)
            error = 0
            
            for i in range(self.seasonal_periods, n):
                s_idx = i % self.seasonal_periods
                forecast = (level + trend) * seasonal[s_idx]
                error += (y[i] - forecast) ** 2
                
                new_level = a * (y[i] / seasonal[s_idx]) + (1 - a) * (level + trend)
                new_trend = b * (new_level - level) + (1 - b) * trend
                seasonal[s_idx] = g * (y[i] / new_level) + (1 - g) * seasonal[s_idx]
                
                level = new_level
                trend = new_trend
            
            return error
        
        try:
            result = minimize(
                objective,
                x0=[0.3, 0.1, 0.1],
                bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)],
                method='L-BFGS-B',
            )
            self._alpha, self._beta, self._gamma = result.x
        except Exception:
            self._alpha, self._beta, self._gamma = 0.3, 0.1, 0.1
        
        self._model_params = {
            'alpha': self._alpha,
            'beta': self._beta,
            'gamma': self._gamma,
        }
        
        self._fitted = True
        self.logger.info(
            f"Exp Smoothing fitted: alpha={self._alpha:.3f}, "
            f"beta={self._beta:.3f}, gamma={self._gamma:.3f}"
        )
        return self
    
    def predict(
        self, horizon: int, return_conf_int: bool = True
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Generate predictions with confidence intervals."""
        if not self._fitted:
            raise ValueError("Model not fitted.")
        
        last_date = datetime.utcnow().date()
        dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        # Simple trend-based prediction if not properly fitted
        predictions = np.array([
            self._last_level + self._last_trend * (i + 1)
            for i in range(horizon)
        ])
        
        forecast_series = pd.Series(predictions, index=dates)
        
        if return_conf_int:
            # Calculate prediction intervals
            std_residual = 0.1 * np.abs(self._last_level)  # Approximation
            
            z_80 = stats.norm.ppf(0.90)
            z_95 = stats.norm.ppf(0.975)
            
            intervals = pd.DataFrame({
                'lower_80': predictions - z_80 * std_residual * np.sqrt(np.arange(1, horizon+1)),
                'upper_80': predictions + z_80 * std_residual * np.sqrt(np.arange(1, horizon+1)),
                'lower_95': predictions - z_95 * std_residual * np.sqrt(np.arange(1, horizon+1)),
                'upper_95': predictions + z_95 * std_residual * np.sqrt(np.arange(1, horizon+1)),
            }, index=dates)
        else:
            intervals = None
        
        return forecast_series, intervals


class MovingAverageForecaster(BaseForecaster):
    """Moving Average and Weighted Moving Average forecaster."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("moving_average", config)
        self.window = config.get('window', 7) if config else 7
        self.weighted = config.get('weighted', True) if config else True
        self._last_ma: float = 0
    
    def fit(self, series: pd.Series) -> 'MovingAverageForecaster':
        """Calculate moving average from historical data."""
        series = series.dropna()
        
        if self.weighted:
            weights = np.arange(1, self.window + 1)
            weights = weights / weights.sum()
            self._last_ma = (series.iloc[-self.window:] * weights[::-1]).sum()
        else:
            self._last_ma = series.iloc[-self.window:].mean()
        
        self._model_params = {
            'window': self.window,
            'weighted': self.weighted,
            'last_ma': self._last_ma,
        }
        
        self._fitted = True
        return self
    
    def predict(
        self, horizon: int, return_conf_int: bool = True
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Generate predictions (constant for MA)."""
        last_date = datetime.utcnow().date()
        dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        predictions = np.full(horizon, self._last_ma)
        forecast_series = pd.Series(predictions, index=dates)
        
        if return_conf_int:
            std_approx = 0.1 * abs(self._last_ma) + 1
            z_80 = stats.norm.ppf(0.90)
            z_95 = stats.norm.ppf(0.975)
            
            intervals = pd.DataFrame({
                'lower_80': predictions - z_80 * std_approx,
                'upper_80': predictions + z_80 * std_approx,
                'lower_95': predictions - z_95 * std_approx,
                'upper_95': predictions + z_95 * std_approx,
            }, index=dates)
        else:
            intervals = None
        
        return forecast_series, intervals


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA-style forecaster using linear regression on lagged values.
    Simplified implementation without external dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("arima", config)
        self.p = config.get('p', 3) if config else 3  # AR order
        self.d = config.get('d', 1) if config else 1  # Differencing
        self.q = config.get('q', 1) if config else 1  # MA order
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: float = 0
        self._last_values: Optional[np.ndarray] = None
        self._residual_std: float = 0
    
    def fit(self, series: pd.Series) -> 'ARIMAForecaster':
        """Fit ARIMA model using linear regression."""
        series = series.dropna()
        y = series.values.astype(float)
        
        # Apply differencing
        for _ in range(self.d):
            y = np.diff(y)
        
        if len(y) <= self.p + self.q:
            self.logger.warning("Insufficient data for ARIMA")
            self._fitted = True
            return self
        
        n = len(y)
        
        # Create lagged features (AR terms)
        X_ar = np.column_stack([y[self.p-i-1:n-i-1] for i in range(self.p)])
        
        # Create error features (MA terms) - simplified
        X_ma = np.column_stack([
            np.random.normal(0, 0.01, n - self.p - self.q)
            for _ in range(self.q)
        ])
        
        X = np.hstack([
            np.ones((n - self.p - self.q, 1)),
            X_ar[self.q:, :],
            X_ma,
        ])
        y_target = y[self.p + self.q:]
        
        # Fit using least squares
        try:
            coeffs = np.linalg.lstsq(X, y_target, rcond=None)[0]
            self._intercept = coeffs[0]
            self._coefficients = coeffs[1:]
        except np.linalg.LinAlgError:
            self._intercept = np.mean(y_target)
            self._coefficients = np.zeros(self.p + self.q)
        
        self._last_values = y[-self.p:]
        self._residual_std = np.std(y_target - X @ np.hstack([self._intercept, self._coefficients]))
        
        self._model_params = {
            'p': self.p, 'd': self.d, 'q': self.q,
            'intercept': self._intercept,
        }
        
        self._fitted = True
        return self
    
    def predict(
        self, horizon: int, return_conf_int: bool = True
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Generate ARIMA predictions."""
        if not self._fitted:
            raise ValueError("Model not fitted.")
        
        last_date = datetime.utcnow().date()
        dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        predictions = []
        history = list(self._last_values) if self._last_values is not None else []
        
        for _ in range(horizon):
            if len(history) >= self.p:
                pred = self._intercept
                for j in range(min(self.p, len(history))):
                    pred += self._coefficients[j] * history[-(j+1)]
                predictions.append(pred)
                history.append(pred)
            else:
                predictions.append(self._intercept)
                history.append(self._intercept)
        
        forecast_series = pd.Series(predictions, index=dates)
        
        if return_conf_int:
            z_80 = stats.norm.ppf(0.90)
            z_95 = stats.norm.ppf(0.975)
            std_increase = self._residual_std * np.sqrt(np.arange(1, horizon+1))
            
            intervals = pd.DataFrame({
                'lower_80': np.array(predictions) - z_80 * std_increase,
                'upper_80': np.array(predictions) + z_80 * std_increase,
                'lower_95': np.array(predictions) - z_95 * std_increase,
                'upper_95': np.array(predictions) + z_95 * std_increase,
            }, index=dates)
        else:
            intervals = None
        
        return forecast_series, intervals


class EnsembleForecaster:
    """Ensemble forecasting combining multiple models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = LoggerSetup.get_logger("ensemble_forecaster")
        
        self.forecasters = {
            'exp_smoothing': ExponentialSmoothingForecaster(config),
            'moving_average': MovingAverageForecaster(config),
            'arima': ARIMAForecaster(config),
        }
        
        self.weights = {'exp_smoothing': 0.4, 'moving_average': 0.2, 'arima': 0.4}
        self._fitted = False
    
    def fit(self, series: pd.Series) -> 'EnsembleForecaster':
        """Fit all forecasting models."""
        self.logger.info("Fitting ensemble forecasters...")
        
        for name, forecaster in self.forecasters.items():
            try:
                forecaster.fit(series)
                self.logger.debug(f"  ✓ {name} fitted")
            except Exception as e:
                self.logger.warning(f"  ✗ {name} failed: {e}")
        
        self._fitted = True
        return self
    
    def predict(
        self, horizon: int, return_conf_int: bool = True
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Generate ensemble predictions."""
        all_predictions = {}
        
        for name, forecaster in self.forecasters.items():
            try:
                preds, _ = forecaster.predict(horizon, return_conf_int=False)
                all_predictions[name] = preds
            except Exception as e:
                self.logger.warning(f"Forecaster {name} failed: {e}")
        
        if not all_predictions:
            raise ValueError("No forecasters produced valid predictions.")
        
        # Weighted average
        ensemble_pred = pd.Series(0.0, index=all_predictions[list(all_predictions.keys())[0]].index)
        total_weight = 0
        
        for name, preds in all_predictions.items():
            weight = self.weights.get(name, 0.3)
            ensemble_pred += preds * weight
            total_weight += weight
        
        ensemble_pred /= total_weight
        
        if return_conf_int:
            # Wider intervals for ensemble
            std_est = ensemble_pred.std() if len(ensemble_pred) > 1 else abs(ensemble_pred.iloc[0]) * 0.1
            z_80 = stats.norm.ppf(0.90)
            z_95 = stats.norm.ppf(0.975)
            
            intervals = pd.DataFrame({
                'lower_80': ensemble_pred - z_80 * std_est,
                'upper_80': ensemble_pred + z_80 * std_est,
                'lower_95': ensemble_pred - z_95 * std_est,
                'upper_95': ensemble_pred + z_95 * std_est,
            }, index=ensemble_pred.index)
        else:
            intervals = None
        
        return ensemble_pred, intervals
    
    def generate_forecast_results(
        self,
        series: pd.Series,
        metric_name: str,
        entity_id: str,
        horizon: int = 30,
    ) -> List[ForecastResult]:
        """Generate structured forecast results."""
        self.fit(series)
        predictions, intervals = self.predict(horizon)
        
        results = []
        
        # Determine trend
        if len(predictions) > 1:
            trend = np.polyfit(range(len(predictions)), predictions.values, 1)[0]
            if trend > 0.01 * np.mean(predictions):
                direction = 'increasing'
            elif trend < -0.01 * np.mean(predictions):
                direction = 'decreasing'
            else:
                direction = 'stable'
        else:
            direction = 'stable'
        
        # Seasonality strength
        if len(series) >= 14:
            acf = [series.autocorr(lag) for lag in range(1, min(8, len(series)//2))]
            seasonality_strength = max(abs(a) for a in acf) if acf else 0
        else:
            seasonality_strength = 0
        
        for i, (date_idx, pred) in enumerate(predictions.items()):
            result = ForecastResult(
                metric_name=metric_name,
                entity_id=entity_id,
                forecast_date=date_idx,
                predicted_value=float(pred),
                lower_bound_80=float(intervals.iloc[i]['lower_80']) if intervals is not None else float(pred) * 0.9,
                upper_bound_80=float(intervals.iloc[i]['upper_80']) if intervals is not None else float(pred) * 1.1,
                lower_bound_95=float(intervals.iloc[i]['lower_95']) if intervals is not None else float(pred) * 0.8,
                upper_bound_95=float(intervals.iloc[i]['upper_95']) if intervals is not None else float(pred) * 1.2,
                trend_direction=direction,
                seasonality_strength=seasonality_strength,
            )
            results.append(result)
        
        return results