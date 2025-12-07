# Photon - Multi-Agent Trading System Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Agents](#agents)
4. [Agent Communication](#agent-communication)
5. [System Workflow](#system-workflow)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Data Sources](#data-sources)
10. [AI Model Details](#ai-model-details)

---

## Project Overview

**Photon** is a multi-agent automated trading system that uses AI to make trading decisions based on real-time market data. The system consists of three specialized agents that work together to monitor markets, make decisions, and execute trades.

### Key Features
- ✅ **3 Working Agents**: Market Monitor, Decision Maker, Execution Agent
- ✅ **Agent Communication**: Agents communicate via message passing
- ✅ **AI-Powered Decisions**: Uses Machine Learning (Random Forest, Gradient Boosting)
- ✅ **Real Market Data**: Supports stocks (yfinance) and cryptocurrencies (Bybit API)
- ✅ **Backtesting**: Historical data simulation
- ✅ **REST API**: Full API for integration
- ✅ **Error Handling**: Robust error handling and fallback mechanisms

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Trading System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────────┐ │
│  │   Market     │─────▶│   Decision   │─────▶│  Execution  │ │
│  │   Monitor    │      │    Maker     │      │    Agent    │ │
│  │   Agent      │      │    Agent     │      │             │ │
│  └──────────────┘      └──────────────┘      └─────────────┘ │
│       │                      │                      │          │
│       └──────────────────────┼──────────────────────┘          │
│                              │                                 │
│                    ┌─────────▼─────────┐                       │
│                    │   Django Backend  │                       │
│                    │   (PostgreSQL)    │                       │
│                    └───────────────────┘                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Django 5 + Django REST Framework
- **Database**: PostgreSQL
- **Task Queue**: Celery + Redis
- **AI/ML**: scikit-learn (Random Forest, Gradient Boosting)
- **Data Sources**: yfinance (stocks), Bybit API (crypto)
- **Deployment**: Docker Compose

---

## Agents

### 1. Market Monitoring Agent

**Purpose**: Monitors market prices and collects real-time data.

**Responsibilities**:
- Fetches market data from external APIs (yfinance, Bybit)
- Computes technical indicators (SMA, RSI, MACD, volatility)
- Analyzes market conditions (trend, strength, signals)
- Preprocesses data for transmission to Decision Maker
- Handles data caching and fallback mechanisms

**Key Features**:
- Automatic source detection (stocks vs crypto)
- CSV file support for backtesting
- Technical indicator calculation
- Market analysis (bull/bear/sideways trends)

**Location**: `backend/trading/agents/market_monitor.py`

**Example Output**:
```json
{
  "timestamp": "2024-12-07T10:00:00Z",
  "ticker": "BTCUSDT",
  "ohlcv": {
    "open": 88457.72,
    "high": 89000.00,
    "low": 88000.00,
    "close": 88500.00,
    "volume": 1234567
  },
  "indicators": {
    "sma10": 88000.00,
    "sma20": 87500.00,
    "rsi14": 55.5,
    "macd": 150.0,
    "macd_hist": 25.0,
    "volatility": 500.0
  },
  "analysis": {
    "trend": "bull",
    "strength": 0.75,
    "signals": {
      "rsi_state": "neutral",
      "sma_cross": 1
    }
  }
}
```

### 2. Decision-Making Agent

**Purpose**: Analyzes market data and makes trading decisions (BUY/SELL/HOLD).

**Responsibilities**:
- Receives market data from Market Monitor
- Uses AI model to predict trading actions
- Applies risk management rules
- Generates confidence scores and reasoning
- Sends decisions to Execution Agent

**AI Model Details**:
- **Model Types**: Random Forest Classifier, Gradient Boosting Classifier
- **Features**: 14 features (price, volume, technical indicators, trend analysis)
- **Classes**: SELL (0), HOLD (1), BUY (2)
- **Training**: Automatic training on historical data
- **Continuous Learning**: Retrains on real trading results

**Key Features**:
- Automatic model training on first use
- Continuous learning from real trades
- Risk management (confidence thresholds, risk scores)
- Rule-based fallback (only when AI unavailable)

**Location**: `backend/trading/agents/decision_maker.py`

**Example Output**:
```json
{
  "action": "BUY",
  "ticker": "BTCUSDT",
  "confidence": 0.85,
  "reasoning": "BUY decision (confidence: 0.85). Trend: BULL. RSI oversold (25.3). Strong trend (strength: 0.82)",
  "quantity": 1,
  "price": 88500.00,
  "risk_score": 0.3,
  "model_type": "random_forest"
}
```

### 3. Execution Agent

**Purpose**: Executes trades based on decisions from Decision Maker.

**Responsibilities**:
- Receives trading decisions
- Validates decisions (quantity, price, balance)
- Executes trades (simulated or real)
- Calculates slippage and commissions
- Records trades in database
- Updates account balance and positions

**Key Features**:
- Simulated execution mode (paper trading)
- Real execution mode (for live trading)
- Slippage simulation
- Commission calculation
- Position management
- Trade confirmation

**Location**: `backend/trading/agents/execution_agent.py`

**Example Output**:
```json
{
  "status": "executed",
  "order_id": "order_123456",
  "ticker": "BTCUSDT",
  "action": "BUY",
  "quantity": 1,
  "requested_price": 88500.00,
  "executed_price": 88588.50,
  "commission": 88.59,
  "slippage": 88.50,
  "timestamp": "2024-12-07T10:00:00Z",
  "message": "Trade executed successfully"
}
```

---

## Agent Communication

### Communication Flow

```
Market Monitor → Decision Maker → Execution Agent
     │                │                  │
     └────────────────┴──────────────────┘
              Message Model (Database)
```

### Message Types

1. **MARKET_SNAPSHOT**: Market Monitor → Decision Maker
   - Contains: OHLCV data, indicators, analysis
   - Purpose: Provide market data for decision making

2. **TRADE_DECISION**: Decision Maker → Execution Agent
   - Contains: Action (BUY/SELL/HOLD), confidence, reasoning
   - Purpose: Instruct execution of trade

3. **EXECUTION_REPORT**: Execution Agent → (logs)
   - Contains: Execution status, executed price, P&L
   - Purpose: Confirm trade execution

### Implementation

Agents communicate through the `Message` model in Django:

```python
# Market Monitor sends data
message = adapter.send_message(
    to_agent="DECISION_MAKER",
    message_type="MARKET_SNAPSHOT",
    payload=market_data
)

# Decision Maker sends decision
message = adapter.send_message(
    to_agent="EXECUTION",
    message_type="TRADE_DECISION",
    payload=decision_data
)
```

**Location**: `backend/trading/models.py` (Message model)
**Integration**: `backend/trading/agents/integration.py` (DjangoAgentAdapter)

---

## System Workflow

### Complete Trading Process

```
1. Market Monitor Agent
   ├─ Fetches market data (yfinance/Bybit/CSV)
   ├─ Computes technical indicators
   ├─ Analyzes market conditions
   └─ Sends MARKET_SNAPSHOT to Decision Maker

2. Decision-Making Agent
   ├─ Receives market data
   ├─ Extracts features (14 features)
   ├─ Uses AI model to predict action
   ├─ Applies risk management
   └─ Sends TRADE_DECISION to Execution Agent

3. Execution Agent
   ├─ Receives trading decision
   ├─ Validates decision
   ├─ Executes trade (simulated/real)
   ├─ Calculates slippage & commission
   └─ Records trade & updates balance
```

### Step-by-Step Example

1. **User adds symbol**: `POST /api/trading/symbols/ {"symbol": "BTCUSDT"}`

2. **Start monitoring**: `POST /api/trading/agents/market-monitor/ {"action": "start"}`

3. **Market Monitor fetches data**:
   - Checks CSV files first
   - Falls back to Bybit API (for crypto)
   - Falls back to yfinance (for stocks)

4. **Market Monitor processes data**:
   - Calculates SMA(10), SMA(20)
   - Calculates RSI(14)
   - Calculates MACD
   - Analyzes trend (bull/bear/sideways)

5. **Market Monitor sends to Decision Maker**:
   - Creates Message with type "MARKET_SNAPSHOT"
   - Stores in database

6. **Decision Maker receives data**:
   - Extracts 14 features
   - Uses AI model to predict (BUY/SELL/HOLD)
   - Applies risk management rules
   - Generates confidence score

7. **Decision Maker sends to Execution Agent**:
   - Creates Message with type "TRADE_DECISION"
   - Stores in database

8. **Execution Agent executes trade**:
   - Validates decision
   - Simulates execution (or real execution)
   - Updates account balance
   - Creates Position (if BUY)
   - Records Trade

9. **Results available via API**:
   - `GET /api/trading/decisions/` - View decisions
   - `GET /api/trading/trades/` - View trades
   - `GET /api/trading/portfolio/` - View portfolio

---

## Installation & Setup

### Prerequisites

- Docker & Docker Compose
- OR: Python 3.9+, PostgreSQL, Redis

### Quick Start with Docker

```bash
# Clone repository
git clone <repository-url>
cd photon-backend

# Copy environment file
cp backend/env.example backend/.env

# Start services
docker compose up --build

# Backend available at http://localhost:666
```

### Manual Setup

```bash
# Install dependencies
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Setup database
python manage.py migrate
python manage.py createsuperuser

# Start services
python manage.py runserver  # Django
celery -A config worker -l info  # Celery worker
celery -A config beat -l info  # Celery beat
```

### Environment Variables

Create `backend/.env`:

```env
# Django
SECRET_KEY=your-secret-key
DEBUG=True

# Database
DB_NAME=photon
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379/0

# Bybit API (optional, not required for public data)
BYBIT_API_KEY=your-api-key
BYBIT_SECRET_KEY=your-secret-key
```

---

## Usage Guide

### 1. Authentication

```bash
# Register
curl -X POST http://localhost:666/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Login
curl -X POST http://localhost:666/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Get token (save for next requests)
TOKEN="your-jwt-token-here"
```

### 2. Add Trading Symbol

```bash
# Add stock (AAPL)
curl -X POST http://localhost:666/api/trading/symbols/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Add cryptocurrency (BTCUSDT)
curl -X POST http://localhost:666/api/trading/symbols/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT"}'
```

### 3. Start Market Monitoring

```bash
curl -X POST http://localhost:666/api/trading/agents/market-monitor/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
```

### 4. Request Trading Decision

```bash
curl -X POST http://localhost:666/api/trading/agents/decision-maker/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol_id": 1}'
```

### 5. View Results

```bash
# View decisions
curl http://localhost:666/api/trading/decisions/ \
  -H "Authorization: Bearer $TOKEN"

# View trades
curl http://localhost:666/api/trading/trades/ \
  -H "Authorization: Bearer $TOKEN"

# View portfolio
curl http://localhost:666/api/trading/portfolio/ \
  -H "Authorization: Bearer $TOKEN"
```

### 6. Backtesting

```bash
# Download historical data
docker compose exec backend python manage.py download_historical_data \
  --symbol BTCUSDT --period 1mo --interval 1h

# Run backtest
docker compose exec backend python manage.py backtest_simulation \
  --email user@example.com \
  --symbol BTCUSDT \
  --start-date 2024-11-01 \
  --end-date 2024-12-01 \
  --interval 1h \
  --speed 0.1 \
  --initial-balance 10000.00
```

---

## API Reference

### Authentication Endpoints

- `POST /api/auth/register/` - Register new user
- `POST /api/auth/login/` - Login and get JWT token
- `POST /api/auth/refresh/` - Refresh JWT token
- `GET /api/auth/me/` - Get current user profile

### Trading Endpoints

#### Symbols
- `GET /api/trading/symbols/` - List all symbols
- `POST /api/trading/symbols/` - Add new symbol
- `GET /api/trading/symbols/{id}/` - Get symbol details
- `DELETE /api/trading/symbols/{id}/` - Delete symbol

#### Market Data
- `GET /api/trading/market-data/` - List market data
- `GET /api/trading/market-data/latest/` - Get latest data for all symbols
- `POST /api/trading/market-data/refresh/` - Refresh market data

#### Decisions
- `GET /api/trading/decisions/` - List all decisions
- `POST /api/trading/decisions/` - Create decision (via Decision Maker)
- `GET /api/trading/decisions/statistics/` - Get decision statistics

#### Trades
- `GET /api/trading/trades/` - List all trades
- `GET /api/trading/trades/{id}/` - Get trade details

#### Portfolio
- `GET /api/trading/portfolio/` - Get portfolio overview
- `GET /api/trading/portfolio/equity-curve/` - Get equity curve data

#### Agents
- `GET /api/trading/agents/market-monitor/` - Get Market Monitor status
- `POST /api/trading/agents/market-monitor/` - Start/stop monitoring
- `GET /api/trading/agents/decision-maker/` - Get Decision Maker status
- `POST /api/trading/agents/decision-maker/` - Request decision
- `GET /api/trading/agents/execution/` - Get Execution Agent status
- `GET /api/trading/agents/detail/` - Get all agents details

#### Analytics
- `GET /api/trading/analytics/performance-metrics/` - Get performance metrics
- `GET /api/trading/analytics/pnl-curve/` - Get P&L curve
- `GET /api/trading/analytics/monthly-breakdown/` - Get monthly breakdown

#### Dashboard
- `GET /api/trading/dashboard/overview/` - Get dashboard overview
- `GET /api/trading/dashboard/market-chart/` - Get market chart data
- `GET /api/trading/dashboard/market-heatmap/` - Get market heatmap

### Full API Documentation

See `backend/MDs/` directory for detailed API documentation:
- `API_AGENTS.md` - Agent endpoints
- `API_PORTFOLIO.md` - Portfolio endpoints
- `API_ANALYTICS.md` - Analytics endpoints
- `API_DASHBOARD.md` - Dashboard endpoints

---

## Data Sources

### Supported Data Sources

1. **yfinance (Yahoo Finance)**
   - Stocks: AAPL, TSLA, MSFT, GOOGL, etc.
   - Automatic detection for stock symbols
   - May be blocked in some regions (use CSV fallback)

2. **Bybit API**
   - Cryptocurrencies: BTCUSDT, ETHUSDT, etc.
   - Automatic detection for crypto symbols
   - Public endpoints (no API keys required)

3. **CSV Files**
   - Historical data for backtesting
   - Place files in `./data/` or `./backend/data/`
   - Format: `{SYMBOL}.csv` or `{SYMBOL}_{INTERVAL}.csv`

### Data Source Priority

1. CSV files (if available)
2. Cache (if available)
3. Bybit API (for crypto)
4. yfinance (for stocks)

### Downloading Historical Data

```bash
# Download data to CSV
docker compose exec backend python manage.py download_historical_data \
  --symbol BTCUSDT \
  --period 1mo \
  --interval 1h \
  --output-dir ./data
```

---

## AI Model Details

### Model Architecture

**Type**: Supervised Learning (Classification)
**Algorithms**: 
- Random Forest Classifier (default)
- Gradient Boosting Classifier

**Input Features** (14 features):
1. Close price
2. Volume
3. Price change
4. SMA(10)
5. SMA(20)
6. RSI(14)
7. MACD
8. MACD Histogram
9. Volatility
10. Trend (bull/bear/sideways)
11. Trend strength
12. RSI state (overbought/oversold/neutral)
13. SMA crossover signal
14. Additional analysis features

**Output Classes**:
- 0: SELL
- 1: HOLD
- 2: BUY

### Training Process

1. **Initial Training**:
   - Automatically trains on first use
   - Uses historical data (1 month by default)
   - Trains on real market data (yfinance/Bybit/CSV)

2. **Continuous Learning**:
   - Retrains every 10 decisions
   - Uses completed trades (BUY → SELL pairs)
   - Learns from real trading results (PnL)

3. **Training Data**:
   - Historical price movements
   - Technical indicators
   - Future price changes (for labels)

### Model Performance

- **Training Accuracy**: ~70-85% (varies by dataset)
- **Test Accuracy**: ~65-80% (varies by dataset)
- **Class Balancing**: Uses `class_weight='balanced'` to reduce HOLD predictions

### Verification

To verify AI model is being used:

```bash
# Check logs
docker compose logs backend | grep "Using AI model"

# Check decision metadata
curl http://localhost:666/api/trading/decisions/ \
  -H "Authorization: Bearer $TOKEN" | jq '.results[0].metadata.model_type'
# Should return: "random_forest" or "gradient_boosting"
```

See `AI_MODEL_VERIFICATION.md` for detailed verification steps.

---

## Error Handling

### Common Issues & Solutions

1. **yfinance Blocking**:
   - Solution: Use CSV files for backtesting
   - Download data: `download_historical_data` command

2. **Bybit 200 Candle Limit**:
   - Solution: Model adapts to smaller datasets
   - Minimum samples reduced to 30

3. **Model Not Training**:
   - Check logs: `docker compose logs backend | grep "Training"`
   - Ensure data is available (CSV/API)

4. **Negative Balance**:
   - Fixed: Balance now includes open positions value
   - Formula: `balance = free_cash + used_margin`

---

## Logs & Monitoring

### View Logs

```bash
# All logs
docker compose logs -f

# Backend logs
docker compose logs -f backend

# Celery logs
docker compose logs -f celery
```

### Agent Status

```bash
# Get agent statuses
curl http://localhost:666/api/trading/agents/detail/ \
  -H "Authorization: Bearer $TOKEN"
```

### Agent Logs

View in Django Admin: `http://localhost:666/admin/trading/agentlog/`

---

## Testing

### Quick Test

```bash
# Test agents workflow
docker compose exec backend python manage.py test_agents \
  --email user@example.com \
  --symbol BTCUSDT
```

### Long-Term Test

```bash
# Run for extended period
docker compose exec backend python manage.py long_term_test \
  --email user@example.com \
  --symbol BTCUSDT \
  --duration 3600
```

### Backtest

```bash
# Historical simulation
docker compose exec backend python manage.py backtest_simulation \
  --email user@example.com \
  --symbol BTCUSDT \
  --start-date 2024-11-01 \
  --end-date 2024-12-01
```

---

## Conclusion

This documentation covers the complete Photon multi-agent trading system. The system successfully implements:

✅ 3 working agents with clear responsibilities
✅ Agent communication via message passing
✅ AI-powered decision making
✅ Real market data integration
✅ Complete trading workflow
✅ Error handling and fallback mechanisms
✅ REST API for integration
✅ Backtesting capabilities

For more details, see the full project report in `PROJECT_REPORT.md`.

