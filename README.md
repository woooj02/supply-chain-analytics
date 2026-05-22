# Supply Chain Analytics Platform

A real-time supply chain analytics platform with ensemble anomaly detection, demand forecasting, and an interactive operations dashboard. Built to demonstrate production-grade Python data engineering — multi-source ETL, time-series modeling, caching with backpressure, and live alerting in a single deployable unit.

## What it does

- **Ingests** inventory, order, and shipment data from CSV, REST APIs, and SQL sources via an async ETL pipeline with validation and transformation stages.
- **Detects anomalies** using an ensemble of statistical (Z-score, IQR), ML-based (Isolation Forest), and time-series (STL decomposition) detectors. The ensemble votes — a single noisy method can't trigger a false alarm.
- **Forecasts demand** with three pluggable models — Exponential Smoothing, Moving Average, and ARIMA — returning point forecasts and prediction intervals.
- **Surfaces insights** through a Plotly Dash dashboard with live KPIs, interactive charts, and an alert feed.
- **Alerts** via Slack and email with per-channel rate limiting so a sudden spike doesn't flood the on-call.

## Why it's interesting

- **Async-first architecture** — Pipeline stages run concurrently with `asyncio`; extractors and transformers are independent and back-pressured.
- **Ensemble detection** — Rather than picking one anomaly method and tuning it forever, three orthogonal detectors run in parallel and a vote determines an alert. Tunable per-method weights.
- **Circuit-breaker caching** — The Redis layer wraps reads with a circuit breaker so cache outages degrade gracefully to direct DB reads instead of cascading failures.
- **Time-series optimized PostgreSQL** — Schema designed for fast windowed queries; migrations managed in code.
- **ATS-style strict typing** — Full `pydantic` validation at every ingest boundary; `mypy --strict` clean.

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+, `asyncio` |
| Data | Pandas, NumPy, SciPy, scikit-learn, statsmodels |
| Persistence | PostgreSQL 14+ (SQLAlchemy 2.x async), Redis 6+ |
| Dashboard | Dash, Plotly, Bootstrap |
| Quality | pydantic, loguru, tenacity, mypy strict, black, pytest + pytest-asyncio |

## Quick start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 2. Configure (see .env.example for full list)
cp .env.example .env
# fill in POSTGRES_*, REDIS_*, SLACK_WEBHOOK_URL etc.

# 3. Initialize DB and seed sample data
python -m supply_chain_analytics.main init

# 4. Run a one-off ETL cycle
python -m supply_chain_analytics.main etl

# 5. Launch the dashboard
python -m supply_chain_analytics.main dashboard --host 0.0.0.0 --port 8050

# 6. Health check
python -m supply_chain_analytics.main health
```

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Redis 6+

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────┐
│  Sources    │───▶│   ETL        │───▶│  Analytics     │───▶│  Dashboard  │
│ CSV/API/DB  │    │ extract→     │    │ anomaly +      │    │  + Alerts   │
└─────────────┘    │ transform→   │    │ forecast +     │    └─────────────┘
                   │ load         │    │ stats          │           │
                   └──────────────┘    └────────────────┘           │
                          │                    │                    │
                          ▼                    ▼                    ▼
                   ┌────────────────────────────────────────────────────┐
                   │  PostgreSQL (time-series store) + Redis (cache)    │
                   └────────────────────────────────────────────────────┘
```

## Layout

```
supply_chain_analytics/
├── src/supply_chain_analytics/
│   ├── etl/                 # extractors, transformers, async pipeline
│   ├── analytics/           # anomaly_detector, forecasting, statistical_analysis
│   ├── dashboard/           # Dash app
│   ├── alerting/            # alert manager (email, Slack), rate limiter
│   ├── cache/               # Redis cache with circuit breaker
│   ├── database/            # SQLAlchemy models, migrations, connection
│   ├── models/              # pydantic schemas
│   ├── core/                # logging, config
│   └── main.py              # click CLI
├── config/                  # settings.py (loads from .env)
├── tests/
└── pyproject.toml
```

## Tests

```bash
pytest tests/ -v --cov=src
```

## License

MIT
