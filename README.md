# Supply Chain Analytics Platform

A production-grade real-time supply chain analytics platform with anomaly detection, demand forecasting, and interactive dashboards.

## Features

- **ETL Pipeline**: Multi-source data extraction (CSV, API, Database) with validation and transformation
- **Anomaly Detection**: Ensemble of statistical, ML-based, and time-series anomaly detectors
- **Demand Forecasting**: Exponential Smoothing, Moving Average, and ARIMA models with prediction intervals
- **Statistical Analysis**: Correlation analysis, hypothesis testing, ABC analysis, trend analysis
- **Real-Time Dashboard**: Interactive Plotly Dash dashboard with KPIs, charts, and alerts
- **Alerting System**: Multi-channel notifications (Email, Slack) with rate limiting
- **Redis Caching**: High-performance data caching with circuit breaker pattern
- **PostgreSQL Database**: Time-series optimized storage with migration management

## Tech Stack

- **Backend**: Python 3.10+, asyncio, SQLAlchemy, Pandas, NumPy, SciPy, scikit-learn
- **Database**: PostgreSQL
- **Cache**: Redis
- **Dashboard**: Dash, Plotly, Bootstrap
- **Logging**: Loguru

## Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+
- Redis 6+

### Setup

1. Clone and navigate:
```bash
cd supply_chain_analytics