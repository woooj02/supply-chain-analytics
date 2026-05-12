"""
Interactive Dash dashboard for Supply Chain Analytics.
Real-time monitoring, anomaly visualization, and KPI tracking.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import settings
from supply_chain_analytics.analytics.statistical_analysis import StatisticalAnalyzer
from supply_chain_analytics.analytics.anomaly_detector import EnsembleAnomalyDetector
from supply_chain_analytics.analytics.forecasting import EnsembleForecaster
from supply_chain_analytics.cache.redis_cache import RedisCache, cached
from supply_chain_analytics.core.logger import LoggerSetup


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Supply Chain Analytics Platform",
)

# Initialize analytics components
stat_analyzer = StatisticalAnalyzer()
anomaly_detector = EnsembleAnomalyDetector()
forecaster = EnsembleForecaster()

# Sidebar
sidebar = dbc.Col(
    [
        html.Div(
            [
                html.H2("📊 Supply Chain", className="display-6"),
                html.H4("Analytics Platform", className="display-7"),
                html.Hr(),
            ],
            className="mb-4",
        ),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-tachometer-alt me-2"), "Overview"],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-exclamation-triangle me-2"), "Anomalies"],
                    href="/anomalies",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-chart-line me-2"), "Forecasting"],
                    href="/forecasting",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-boxes me-2"), "Inventory"],
                    href="/inventory",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-chart-bar me-2"), "Reports"],
                    href="/reports",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-cog me-2"), "Settings"],
                    href="/settings",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    width=2,
    className="bg-dark sidebar p-4 min-vh-100",
)

# Main content area
content = dbc.Col(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="page-content"),
                    width=12,
                ),
            ]
        ),
    ],
    width=10,
    className="p-4",
)

# App layout
app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dbc.Row([sidebar, content]),
        dcc.Interval(
            id="interval-component",
            interval=30 * 1000,  # 30 seconds
            n_intervals=0,
        ),
    ],
    fluid=True,
)


# KPI Card Component
def create_kpi_card(title: str, value: str, icon: str, color: str, trend: str = "") -> dbc.Card:
    """Create a KPI metric card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"fas {icon} fa-2x", style={"color": color}),
                        html.H5(title, className="card-title mt-2"),
                        html.H2(value, className="card-text"),
                        html.Small(trend, className="text-muted") if trend else None,
                    ],
                    className="text-center",
                )
            ]
        ),
        className="shadow-sm mb-3",
    )


# Overview Page
overview_layout = html.Div(
    [
        html.H2("📊 Supply Chain Overview", className="mb-4"),
        
        # KPI Row
        dbc.Row(
            [
                dbc.Col(create_kpi_card("Total Orders", "12,847", "fa-shopping-cart", "#4e79a7", "↑ 12.5% vs last month"), width=3),
                dbc.Col(create_kpi_card("Revenue", "$2.4M", "fa-dollar-sign", "#59a14f", "↑ 8.3% vs last month"), width=3),
                dbc.Col(create_kpi_card("Inventory Turnover", "5.2x", "fa-sync-alt", "#f28e2b", "↓ 0.3x vs last month"), width=3),
                dbc.Col(create_kpi_card("On-Time Delivery", "94.7%", "fa-truck", "#e15759", "↑ 2.1% vs target"), width=3),
            ],
            className="mb-4",
        ),
        
        # Charts Row
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Revenue Trend", className="card-title"),
                            dcc.Graph(id="revenue-trend-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Order Status Distribution", className="card-title"),
                            dcc.Graph(id="order-status-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        
        # Second Charts Row
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Inventory Levels by Warehouse", className="card-title"),
                            dcc.Graph(id="inventory-warehouse-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Top Products by Revenue", className="card-title"),
                            dcc.Graph(id="top-products-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        
        # Recent Anomalies Table
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("🚨 Recent Anomalies", className="card-title"),
                            dash_table.DataTable(
                                id="recent-anomalies-table",
                                columns=[
                                    {"name": "Time", "id": "timestamp"},
                                    {"name": "Metric", "id": "metric"},
                                    {"name": "Severity", "id": "severity"},
                                    {"name": "Deviation", "id": "deviation"},
                                    {"name": "Status", "id": "status"},
                                ],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "#303030",
                                    "color": "white",
                                    "border": "1px solid #444",
                                    "padding": "8px",
                                },
                                style_header={
                                    "backgroundColor": "#444",
                                    "fontWeight": "bold",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": '{severity} = "CRITICAL"'},
                                        "backgroundColor": "#5e0000",
                                        "color": "white",
                                    },
                                    {
                                        "if": {"filter_query": '{severity} = "WARNING"'},
                                        "backgroundColor": "#5e4d00",
                                        "color": "white",
                                    },
                                ],
                                page_size=10,
                            ),
                        ]),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ],
        ),
    ],
)


# Anomalies Page
anomalies_layout = html.Div(
    [
        html.H2("🚨 Anomaly Detection", className="mb-4"),
        
        # Filters
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Filters"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Severity"),
                                    dcc.Dropdown(
                                        id="anomaly-severity-filter",
                                        options=[
                                            {"label": "Critical", "value": "CRITICAL"},
                                            {"label": "Warning", "value": "WARNING"},
                                            {"label": "Info", "value": "INFO"},
                                        ],
                                        multi=True,
                                        value=["CRITICAL", "WARNING"],
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Time Range"),
                                    dcc.Dropdown(
                                        id="anomaly-time-filter",
                                        options=[
                                            {"label": "Last 24 hours", "value": "24h"},
                                            {"label": "Last 7 days", "value": "7d"},
                                            {"label": "Last 30 days", "value": "30d"},
                                        ],
                                        value="24h",
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Metric"),
                                    dcc.Dropdown(
                                        id="anomaly-metric-filter",
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "Sales", "value": "sales"},
                                            {"label": "Inventory", "value": "inventory"},
                                            {"label": "Delivery", "value": "delivery"},
                                        ],
                                        value="all",
                                    ),
                                ], width=4),
                            ]),
                        ]),
                        className="shadow-sm mb-4",
                    ),
                ),
            ],
        ),
        
        # Anomaly Timeline
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Anomaly Timeline", className="card-title"),
                            dcc.Graph(id="anomaly-timeline-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        
        # Anomaly Details
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Anomaly Details"),
                            dash_table.DataTable(
                                id="anomaly-details-table",
                                columns=[
                                    {"name": "Time", "id": "timestamp"},
                                    {"name": "Metric", "id": "metric_name"},
                                    {"name": "Entity", "id": "entity_id"},
                                    {"name": "Actual", "id": "actual_value"},
                                    {"name": "Expected", "id": "expected_value"},
                                    {"name": "Deviation %", "id": "deviation_pct"},
                                    {"name": "Z-Score", "id": "z_score"},
                                    {"name": "Severity", "id": "severity"},
                                    {"name": "Confidence", "id": "confidence"},
                                    {"name": "Recommendation", "id": "recommendation"},
                                ],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "#303030",
                                    "color": "white",
                                    "border": "1px solid #444",
                                    "padding": "8px",
                                    "maxWidth": "200px",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                },
                                style_header={
                                    "backgroundColor": "#444",
                                    "fontWeight": "bold",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": '{severity} = "CRITICAL"'},
                                        "backgroundColor": "#5e0000",
                                    },
                                ],
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                            ),
                        ]),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ],
        ),
    ],
)


# Forecasting Page
forecasting_layout = html.Div(
    [
        html.H2("📈 Demand Forecasting", className="mb-4"),
        
        # Controls
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Product / Metric"),
                                    dcc.Dropdown(
                                        id="forecast-metric-selector",
                                        options=[
                                            {"label": "Total Sales", "value": "total_sales"},
                                            {"label": "Product A", "value": "product_a"},
                                            {"label": "Product B", "value": "product_b"},
                                        ],
                                        value="total_sales",
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Forecast Horizon (Days)"),
                                    dcc.Slider(
                                        id="forecast-horizon-slider",
                                        min=7,
                                        max=90,
                                        step=7,
                                        value=30,
                                        marks={i: str(i) for i in [7, 14, 30, 60, 90]},
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Confidence Level"),
                                    dcc.RadioItems(
                                        id="forecast-confidence",
                                        options=[
                                            {"label": "80%", "value": 80},
                                            {"label": "95%", "value": 95},
                                        ],
                                        value=95,
                                        inline=True,
                                    ),
                                ], width=4),
                            ]),
                        ]),
                        className="shadow-sm mb-4",
                    ),
                ),
            ],
        ),
        
        # Forecast Chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Forecast with Prediction Intervals", className="card-title"),
                            dcc.Graph(id="forecast-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Forecast Metrics", className="card-title"),
                            html.Div(id="forecast-metrics"),
                            html.Hr(),
                            html.H6("Trend Analysis"),
                            html.Div(id="trend-analysis"),
                            html.Hr(),
                            html.H6("Seasonality"),
                            html.Div(id="seasonality-info"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        
        # Historical vs Predicted Table
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Forecast Data"),
                            dash_table.DataTable(
                                id="forecast-data-table",
                                columns=[
                                    {"name": "Date", "id": "date"},
                                    {"name": "Predicted", "id": "predicted"},
                                    {"name": "Lower (95%)", "id": "lower_95"},
                                    {"name": "Upper (95%)", "id": "upper_95"},
                                    {"name": "Trend", "id": "trend"},
                                ],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "#303030",
                                    "color": "white",
                                    "border": "1px solid #444",
                                    "padding": "8px",
                                },
                                style_header={
                                    "backgroundColor": "#444",
                                    "fontWeight": "bold",
                                },
                                page_size=30,
                            ),
                        ]),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ],
        ),
    ],
)


# Inventory Page
inventory_layout = html.Div(
    [
        html.H2("📦 Inventory Management", className="mb-4"),
        
        # ABC Analysis
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("ABC Analysis", className="card-title"),
                            dcc.Graph(id="abc-analysis-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Stock Level Heatmap", className="card-title"),
                            dcc.Graph(id="stock-heatmap-chart"),
                        ]),
                        className="shadow-sm",
                    ),
                    width=6,
                ),
            ],
            className="mb-4",
        ),
    ],
)


# Reports Page
reports_layout = html.Div(
    [
        html.H2("📊 Reports", className="mb-4"),
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Generate Report"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Report Type"),
                                    dcc.Dropdown(
                                        id="report-type-selector",
                                        options=[
                                            {"label": "Daily Summary", "value": "daily"},
                                            {"label": "Weekly Report", "value": "weekly"},
                                            {"label": "Monthly Analysis", "value": "monthly"},
                                            {"label": "Anomaly Report", "value": "anomaly"},
                                            {"label": "Forecast Report", "value": "forecast"},
                                        ],
                                        value="daily",
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Format"),
                                    dcc.RadioItems(
                                        id="report-format",
                                        options=[
                                            {"label": " PDF", "value": "pdf"},
                                            {"label": " Excel", "value": "excel"},
                                            {"label": " HTML", "value": "html"},
                                        ],
                                        value="pdf",
                                        inline=True,
                                    ),
                                ], width=6),
                            ]),
                            html.Br(),
                            dbc.Button(
                                "Generate Report",
                                id="generate-report-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Download",
                                id="download-report-btn",
                                color="success",
                                disabled=True,
                            ),
                        ]),
                        className="shadow-sm mb-4",
                    ),
                    width=12,
                ),
            ],
        ),
    ],
)


# Settings Page
settings_layout = html.Div(
    [
        html.H2("⚙️ Settings", className="mb-4"),
        dbc.Card(
            dbc.CardBody([
                html.H5("System Configuration"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Data Refresh Interval (seconds)"),
                        dcc.Input(
                            id="refresh-interval",
                            type="number",
                            value=30,
                            min=5,
                            max=300,
                            className="form-control",
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Anomaly Sensitivity"),
                        dcc.Slider(
                            id="anomaly-sensitivity",
                            min=1,
                            max=5,
                            step=0.5,
                            value=3,
                            marks={i: str(i) for i in range(1, 6)},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Alert Notifications"),
                        dbc.Switch(
                            id="alert-notifications",
                            label="Enable Email Alerts",
                            value=True,
                        ),
                        dbc.Switch(
                            id="slack-notifications",
                            label="Enable Slack Alerts",
                            value=True,
                        ),
                    ], width=4),
                ]),
                html.Br(),
                dbc.Button("Save Settings", id="save-settings", color="primary"),
            ]),
            className="shadow-sm",
        ),
    ],
)


# Callbacks
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page_content(pathname: str):
    """Route to correct page layout."""
    if pathname == "/" or pathname == "/overview":
        return overview_layout
    elif pathname == "/anomalies":
        return anomalies_layout
    elif pathname == "/forecasting":
        return forecasting_layout
    elif pathname == "/inventory":
        return inventory_layout
    elif pathname == "/reports":
        return reports_layout
    elif pathname == "/settings":
        return settings_layout
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not Found", className="text-danger"),
                html.Hr(),
                html.P(f"The path {pathname} was not recognized..."),
            ]
        )


@app.callback(
    Output("revenue-trend-chart", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_revenue_chart(n):
    """Update revenue trend chart."""
    # Generate sample data for demo
    dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
    revenue = np.cumsum(np.random.normal(1000, 200, 90)) + 50000
    revenue = revenue + np.sin(np.arange(90) * 2 * np.pi / 30) * 2000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=revenue,
        mode="lines",
        name="Revenue",
        line=dict(color="#4e79a7", width=2),
        fill="tozeroy",
        fillcolor="rgba(78, 121, 167, 0.2)",
    ))
    
    # Add moving average
    ma = pd.Series(revenue).rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=dates,
        y=ma,
        mode="lines",
        name="7-Day MA",
        line=dict(color="#f28e2b", width=2, dash="dash"),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font_color="white",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


@app.callback(
    Output("order-status-chart", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_order_status_chart(n):
    """Update order status pie chart."""
    statuses = ["Delivered", "Shipped", "Processing", "Pending", "Cancelled"]
    values = [45, 25, 15, 10, 5]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=statuses,
            values=values,
            hole=0.4,
            marker_colors=["#59a14f", "#4e79a7", "#f28e2b", "#76b7b2", "#e15759"],
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font_color="white",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    return fig


@app.callback(
    Output("inventory-warehouse-chart", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_inventory_warehouse_chart(n):
    """Update inventory by warehouse chart."""
    warehouses = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest"]
    inventory_levels = np.random.randint(5000, 20000, 6)
    
    fig = go.Figure(data=[
        go.Bar(
            x=warehouses,
            y=inventory_levels,
            marker_color="#4e79a7",
            text=inventory_levels,
            textposition="outside",
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font_color="white",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    return fig


@app.callback(
    Output("top-products-chart", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_top_products_chart(n):
    """Update top products bar chart."""
    products = [f"Product {chr(65+i)}" for i in range(10)]
    revenue = sorted(np.random.randint(50000, 500000, 10), reverse=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=products,
            x=revenue,
            orientation="h",
            marker_color="rgba(89, 161, 79, 0.8)",
            text=[f"${x:,.0f}" for x in revenue],
            textposition="outside",
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font_color="white",
        height=350,
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis_title="Revenue ($)",
    )
    
    return fig


@app.callback(
    Output("forecast-chart", "figure"),
    Input("forecast-metric-selector", "value"),
    Input("forecast-horizon-slider", "value"),
    Input("forecast-confidence", "value"),
)
def update_forecast_chart(metric, horizon, confidence):
    """Update forecast visualization."""
    # Generate sample data
    historical_dates = pd.date_range(end=datetime.utcnow(), periods=90, freq="D")
    historical_values = np.cumsum(np.random.normal(100, 20, 90)) + 1000
    
    forecast_dates = pd.date_range(
        start=historical_dates[-1] + timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    
    base_value = historical_values[-1]
    trend = 5
    forecast_values = base_value + np.cumsum(np.random.normal(trend, 15, horizon))
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_values,
        mode="lines",
        name="Historical",
        line=dict(color="#4e79a7", width=2),
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines",
        name="Forecast",
        line=dict(color="#f28e2b", width=3),
    ))
    
    # Confidence interval
    if confidence == 95:
        std_mult = 1.96
    else:
        std_mult = 1.28
    
    std_dev = np.arange(1, horizon+1) * 5
    upper = forecast_values + std_mult * std_dev
    lower = forecast_values - std_mult * std_dev
    
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(242, 142, 43, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{confidence}% Confidence",
    ))
    
    fig.add_vline(
        x=forecast_dates[0],
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font_color="white",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h"),
    )
    
    return fig


@app.callback(
    Output("forecast-metrics", "children"),
    Input("forecast-metric-selector", "value"),
    Input("forecast-horizon-slider", "value"),
)
def update_forecast_metrics(metric, horizon):
    """Update forecast metrics display."""
    mape = np.random.uniform(5, 15)
    mae = np.random.uniform(50, 200)
    trend_pct = np.random.uniform(-5, 10)
    
    return html.Div([
        html.P(f"MAPE: {mape:.1f}%", className="mb-2"),
        html.P(f"MAE: {mae:.0f} units", className="mb-2"),
        html.P(f"Expected Growth: {trend_pct:+.1f}%", className="mb-2"),
        html.P(f"Forecast Horizon: {horizon} days", className="mb-2"),
    ])


@app.callback(
    Output("abc-analysis-chart", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_abc_chart(n):
    """Update ABC analysis chart."""
    categories = ["A", "B", "C"]
    item_counts = [20, 30, 50]
    value_pct = [80, 15, 5]
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Items by Category", "Value by Category"),
    )
    
    fig.add_trace(
        go.Pie(labels=categories, values=item_counts, marker_colors=["#59a14f", "#f28e2b", "#e15759"]),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=categories, y=value_pct, marker_color=["#59a14f", "#f28e2b", "#e15759"]),
        row=1, col=2,
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font_color="white",
        height=350,
    )
    
    return fig


def run_dashboard(host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    """Run the Dash dashboard server."""
    logger.info(f"Starting Supply Chain Analytics Dashboard on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)