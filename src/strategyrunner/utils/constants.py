"""Magic strings and constants used throughout the strategy runner pipeline."""

# OHLCV column names (Title-case for normalized data)
COL_OPEN = "Open"
COL_HIGH = "High"
COL_LOW = "Low"
COL_CLOSE = "Close"
COL_VOLUME = "Volume"
COL_DATE = "date"
OHLCV_COLS = (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE)

# Config keys
KEY_MARKET = "market"
KEY_SYMBOLS = "symbols"
KEY_INSTANCES = "instances"
KEY_PARAMS = "params"
KEY_PARAMS_NAME = "name"
KEY_PARAMS_DEFAULTS = "defaults"
KEY_PARAMS_PER_SYMBOL = "per_symbol"
KEY_DATA = "data"
KEY_DATA_PROVIDER = "provider"
KEY_DATA_INTERVAL = "interval"
KEY_DATA_HISTORY_DAYS = "history_days"
KEY_WEBHOOK = "webhook"
KEY_WEBHOOKS = "webhooks"
KEY_STATE = "state"
KEY_STATE_PATH = "path"
KEY_STATE_DRY_PATH = "dry_path"
KEY_SESSION_BUFFER_MIN = "session_buffer_minutes"
KEY_RISK = "risk"
KEY_SIZING = "sizing"

# Instance config keys
KEY_INSTANCE_ID = "id"
KEY_INSTANCE_SYMBOL = "symbol"
KEY_INSTANCE_OVERRIDES = "overrides"
KEY_INSTANCE_WEBHOOK = "webhook"
KEY_INSTANCE_SIZING = "sizing"

# Webhook keys
KEY_WEBHOOK_ENABLED = "enabled"
KEY_WEBHOOK_URL_ENV = "url_env"
KEY_WEBHOOK_SECRET_ENV = "secret_env"
KEY_WEBHOOK_TIMEOUT = "timeout"
KEY_WEBHOOK_SEND_METRICS = "send_metrics"
KEY_WEBHOOK_MESSAGE_TEMPLATE = "message_template"

# State keys
KEY_STATE_LAST_SIGNALS = "last_signals"
KEY_STATE_POSITIONS = "positions"
KEY_STATE_LAST_ASOF = "last_asof"
KEY_STATE_LAST_METRICS = "last_metrics"
KEY_STATE_MODE = "mode"

# Strategy names
STRATEGY_CROSSOVER = "crossover"
STRATEGY_MOMENTUM = "momentum"

# Position sizing keys
KEY_SIZING_MODE = "mode"
KEY_SIZING_MODE_FIXED = "fixed"
KEY_SIZING_MODE_PERCENT = "percent_of_cash"
KEY_SIZING_MODE_NOTIONAL = "notional"
KEY_SIZING_FIXED_QTY = "fixed_qty"
KEY_SIZING_PERCENT = "percent"
KEY_SIZING_NOTIONAL = "notional"
KEY_SIZING_CASH_ENV = "cash_env"
KEY_SIZING_LOT_SIZE = "lot_size"
KEY_SIZING_MIN_QTY = "min_qty"

# Trade/action keys
KEY_TRADE_SYMBOL = "symbol"
KEY_TRADE_INSTANCE = "instance"
KEY_TRADE_ACTION = "action"
KEY_TRADE_TYPE = "type"
KEY_TRADE_TARGET = "target"
KEY_TRADE_PRICE = "price"
KEY_TRADE_QTY = "qty"
KEY_TRADE_NOTIONAL = "notional"
KEY_TRADE_ORDER = "order"
KEY_TRADE_POS_BEFORE = "pos_before"
KEY_TRADE_POS_AFTER = "pos_after"
KEY_TRADE_LEG_INDEX = "leg_index"
KEY_TRADE_SIGNAL_PREV = "signal_prev"
KEY_TRADE_SIGNAL_CURR = "signal_curr"

# Data provider names
PROVIDER_YAHOO = "yahoo"

# Defaults
DEFAULT_MARKET = "XNAS"
DEFAULT_SESSION_BUFFER_MINUTES = 15
DEFAULT_HISTORY_DAYS = 400
DEFAULT_INTERVAL = "1d"
DEFAULT_WEBHOOK_TIMEOUT = 10
DEFAULT_WEBHOOK_RETRIES = 3
