# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

QuantsLab is a Python framework for quantitative trading research with Hummingbot. It provides a task orchestration system for data collection, backtesting, strategy optimization, and deployment. Built on Python 3.12 with MongoDB for task state management.

## Development Environment

**Setup:**
```bash
make install              # Full installation (conda env + MongoDB)
conda activate quants-lab
make run-db              # Start MongoDB
```

**Code formatting (black + isort):**
```bash
black --line-length 130 .
isort --profile black --line-length 130 .
```

**Testing and validation:**
```bash
make validate-config config=tf_pipeline.yml
make list-tasks config=tf_pipeline.yml
```

## Task Management Commands

**Run tasks continuously:**
```bash
# Docker (production):
make run-tasks config=tf_pipeline.yml

# Local (development):
make run-tasks config=tf_pipeline.yml source=1
```

**Trigger single task:**
```bash
make trigger-task task=data_collection config=tf_pipeline.yml
make trigger-task task=data_collection config=tf_pipeline.yml source=1  # local
```

**Monitor and control:**
```bash
make logs-tasks    # View logs
make ps-tasks      # List running tasks
make stop-tasks    # Stop all tasks
```

**CLI direct usage:**
```bash
python cli.py run-tasks --config config/tf_pipeline.yml
python cli.py trigger-task --task candles_downloader --config config/tf_pipeline.yml
python cli.py list-tasks --config config/tf_pipeline.yml
python cli.py validate-config --config config/tf_pipeline.yml
python cli.py serve --config config/tf_pipeline.yml --port 8000  # API server
```

## Architecture

### Core Framework (`core/`)

**`core/tasks/`** - Task orchestration system
- `base.py`: `BaseTask` abstract class with lifecycle hooks (setup, execute, cleanup, on_success, on_failure, on_retry)
- `orchestrator.py`: `TaskOrchestrator` manages task execution, dependencies, and scheduling
- `runner.py`: `TaskRunner` loads configs, initializes tasks, and runs them continuously
- `storage.py`: `MongoDBTaskStorage` persists task state and execution history
- `api.py`: FastAPI endpoints for task management
- Task execution flow: setup() → execute() → cleanup() → on_success()/on_failure()

**`core/backtesting/`** - Backtesting engine
- `engine.py`: Backtesting execution engine
- `optimizer.py`: `StrategyOptimizer` with Optuna integration for hyperparameter optimization
- `triple_barrier_method.py`: Triple barrier labeling for ML

**`core/data_sources/`** - Market data integrations
- `clob.py`: CLOB (Central Limit Order Book) data - order books, trades, candles, funding rates
- `gateway.py`: Hummingbot Gateway integration for AMM/DEX data
- `hummingbot_database.py`: Historical data from Hummingbot database
- `market_feeds/`: Real-time market data feeds (Binance Perpetual, etc.)

**`core/features/`** - Feature engineering and signal generation

**`core/data_paths.py`** - Centralized data path management

**`core/database_manager.py`** - MongoDB client singleton (`db_manager`)

### Application Layer (`app/`)

**`app/tasks/`** - Task implementations (inherit from `BaseTask`)
- `data_collection/`: Candles downloader, pools screener
- `backtesting/`: MACD BB backtesting task
- `notebook/`: Jupyter notebook execution task
- `screeners/`: Market screeners
- `quantitative_methods/`: Quantitative analysis tasks
- `deployment/`: Strategy deployment tasks

**`app/controllers/`** - Trading strategy controllers (Hummingbot format)
- `directional_trading/`: Directional strategies
- `market_making/`: Market making strategies
- `generic/`: Generic controllers

**`app/data/`** - Application data storage
**`app/outputs/`** - Task outputs (notebook results, reports, etc.)

### Configuration (`config/`)

YAML files defining task pipelines:
- `template_1_candles_optimization.yml`: Candles download → optimization pipeline
- `template_2_candles_pools_screener.yml`: Candles + pools + screener pipeline
- `template_3_periodic_reports.yml`: Periodic reporting pipeline
- `template_4_notebook_execution.yml`: Notebook execution pipeline
- `tf_pipeline.yml`: Trend follower pipeline

### Research (`research_notebooks/`)

Jupyter notebooks for strategy research and data analysis.

## Task System

### Creating Tasks

Tasks inherit from `BaseTask` and implement:
```python
async def execute(self, context: TaskContext) -> Dict[str, Any]:
    # Main task logic
    return {"result": "data"}
```

Optional lifecycle hooks:
- `setup(context)`: Initialize resources, validate prerequisites
- `cleanup(context, result)`: Clean up resources
- `on_success(context, result)`: Success handling
- `on_failure(context, result)`: Failure handling
- `on_retry(context, attempt, error)`: Retry handling

Tasks automatically have access to:
- `self.mongodb_client`: MongoDB client (initialized in setup)
- `self.notification_manager`: Notification manager (initialized in setup)
- `self.config.config`: Task-specific configuration from YAML

### Task Configuration

Tasks are defined in YAML files with:
- `enabled`: Whether task is enabled
- `task_class`: Python module path (e.g., `app.tasks.data_collection.candles_downloader_task.CandlesDownloaderTask`)
- `schedule`: Frequency or cron-based scheduling
- `dependencies`: Task dependencies with triggers (on_success, on_failure, on_completion)
- `max_retries`, `retry_delay_seconds`, `timeout_seconds`: Execution controls
- `config`: Task-specific parameters
- `tags`: Tags for filtering

### Task Dependencies

Tasks can depend on other tasks:
```yaml
dependencies:
  - task_name: "candles_downloader"
    on_completion: true  # Trigger on any completion
    delay_seconds: 300   # Wait 5 minutes after completion
```

Trigger modes:
- `on_success`: Trigger only on successful completion
- `on_failure`: Trigger only on failure
- `on_completion`: Trigger on any completion (success or failure)

### Task States

Tasks progress through states defined in `TaskStatus`:
- `PENDING`: Waiting to run
- `RUNNING`: Currently executing
- `COMPLETED`: Finished successfully
- `FAILED`: Failed execution
- `CANCELLED`: Manually cancelled
- `SKIPPED`: Skipped (e.g., already running)

## Database

**MongoDB connection:**
- URL: `mongodb://admin:admin@localhost:27017/quants_lab`
- Configuration: `.env` file (created during install)
- UI: Mongo Express at http://localhost:28081 (admin/changeme)

**Task storage:**
- Task states and execution history stored in MongoDB
- Accessed via `MongoDBTaskStorage` in `core/tasks/storage.py`
- Database manager singleton: `from core.database_manager import db_manager`

## Data Sources

Available data sources (in `core/data_sources/`):
- **CLOB**: Order books, trades, candles (OHLCV), funding rates
- **AMM/Gateway**: DEX liquidity and pool data via Hummingbot Gateway
- **GeckoTerminal**: Multi-network OHLCV data
- **CoinGecko**: Market data and statistics
- **Market Feeds**: Real-time streaming data (trades, OI) from exchanges

## Backtesting & Optimization

**Running backtests:**
- Implement strategies using Hummingbot controller format
- Use `core/backtesting/engine.py` for backtesting execution
- Results include PnL, metrics, and trade analysis

**Hyperparameter optimization:**
```bash
# Launch Optuna dashboard
make launch-optuna

# Kill dashboard
make kill-optuna
```

Optimization tasks use `StrategyOptimizer` from `core/backtesting/optimizer.py` with Optuna for hyperparameter tuning.

## Docker

**Build image:**
```bash
make build
```

**Task execution:**
- Tasks run in Docker by default (production mode)
- Add `source=1` to run locally (development mode)
- Docker mounts: `app/outputs`, `config`, `app`, `research_notebooks`
- Environment: Loaded from `.env` file
- Network: `--network host` for database access

## Common Patterns

**Creating a new task:**
1. Create task class inheriting from `BaseTask` in `app/tasks/`
2. Implement `async def execute(self, context: TaskContext) -> Dict[str, Any]`
3. Add task to a config YAML file in `config/`
4. Run with `make run-tasks config=your_config.yml`

**Accessing MongoDB:**
```python
from core.database_manager import db_manager
client = await db_manager.get_mongodb_client()
db = client["quants_lab"]
collection = db["your_collection"]
```

**Using notifications:**
```python
# Available in tasks via self.notification_manager
if self.notification_manager:
    await self.notification_manager.send_notification(
        "Task completed",
        level="info"
    )
```

**Loading candles data:**
```python
from core.data_sources.clob import CLOB
clob = CLOB(connector_name="binance_perpetual")
candles = await clob.get_candles(
    trading_pair="BTC-USDT",
    interval="1h",
    start_time=start_time,
    end_time=end_time
)
```

## Maintenance

**Clean task states:**
```bash
make cleanup-tasks         # Clean stale task states
make list-task-states      # List current task states
```

**Clean Python cache:**
```bash
make clean
```

**Clean database (WARNING: DATA LOSS):**
```bash
make clean-db
```

## Code Style

- Line length: 130 characters
- Formatter: Black with isort
- Python version: 3.12
- Async/await patterns throughout
- Pydantic models for configuration and validation
