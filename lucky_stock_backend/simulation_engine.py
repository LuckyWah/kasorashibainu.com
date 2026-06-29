import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_builder import build_adaptive_dataset

try:
    from numba import njit
except ImportError:
    njit = None


CPU_COUNT = os.cpu_count() or 1
MIN_SIMULATION_TRADING_DAYS = 20
MAX_SIMULATION_TRADING_DAYS = 100

FEATURES = [
    "drawdown",
    "momentum_20d",
    "momentum_60d",
    "ma_gap",
    "volatility_20d",
    "volume_spike",
    "spy_return",
    "spy_drawdown",
    "vix",
    "fed_rate",
    "treasury_10y",
    "cpi_yoy",
]

BOUNDS = [
    (0.0000, 0.0200),
    (0.0000, 0.3000),
    (0.0000, 5.0000),
    (0.0000, 0.1000),
]

SELL_BOUNDS = [
    (0.0000, 0.0100),
    (0.0000, 0.0500),
    (0.0000, 0.3000),
    (0.0000, 0.0100),
]

DEFAULT_CONFIG = {
    "prediction_days": 1,
    "max_daily_fraction": 0.05,
    "min_daily_investment": 0.0,
    "cash_deployment_penalty": 0.20,
    "rolling_opt_window": 120,

    # Faster realistic backtest settings.
    # The model and theta are still trained only on past data, but not refreshed every day.
    "model_refresh_days": 20,
    "theta_refresh_days": 5,

    "theta_smoothing": 0.20,
    "optimizer_workers": 1,
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "min_training_rows": 30,
    "random_state": 42,

    # Differential evolution speed settings.
    "optimizer_maxiter": 8,
    "optimizer_popsize": 5,

    # Sell simulation settings. These keep the app sell model and suggestion logic intact.
    "max_daily_sell_fraction": 0.03,
    "min_daily_sell_shares": 0.0,
    "sell_min_near_high_score": 0.70,
    "sell_new_high_scale": 0.15,
    "sell_min_runup_score": 0.03,
    "sell_min_weakening_score": 0.01,
    "sell_min_overextension_score": 0.0,
    "sell_max_predicted_return": 0.001,
    "sell_strong_climb_return": 0.003,
    "sell_overextension_scale": 0.08,
    "sell_runup_scale": 0.20,
    "sell_weakening_scale": 0.10,
    "sell_downside_scale": 0.03,
    "sell_strength": 1.0,
    "sell_day_penalty": 0.0005,
    "sell_positive_next_day_penalty": 0.25,
    "sell_share_retention_penalty": 0.05,
}


@dataclass(frozen=True)
class ThetaObjective:
    window_df: pd.DataFrame
    initial_cash: float
    config: dict

    def __post_init__(self):
        object.__setattr__(self, "prices", self.window_df["close"].to_numpy(dtype=float))
        object.__setattr__(self, "drawdowns", self.window_df["drawdown"].to_numpy(dtype=float))
        object.__setattr__(
            self,
            "predicted_returns",
            self.window_df["predicted_return"].to_numpy(dtype=float),
        )
        object.__setattr__(
            self,
            "volatilities",
            self.window_df["volatility_20d"].to_numpy(dtype=float),
        )

    def __call__(self, theta):
        return simulate_arrays_for_theta(
            theta,
            self.initial_cash,
            self.config,
            self.prices,
            self.drawdowns,
            self.predicted_returns,
            self.volatilities,
        )


@dataclass(frozen=True)
class SellThetaObjective:
    window_df: pd.DataFrame
    initial_shares: float
    config: dict

    def __call__(self, theta):
        return simulate_sell_window_for_theta(
            self.window_df,
            theta,
            self.initial_shares,
            self.config,
        )


def find_or_build_dataset(ticker, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper().strip()
    dataset_path = data_dir / f"{ticker}.csv"

    if dataset_path.exists():
        return dataset_path

    return build_adaptive_dataset(ticker, output_dir=data_dir)


def compute_metrics(cash, shares, initial_cash, price):
    invested_total = initial_cash - cash
    stock_value = shares * price
    portfolio_value = stock_value + cash
    avg_cost = invested_total / shares if shares > 0 else 0.0
    roic = stock_value / invested_total - 1 if invested_total > 0 else 0.0

    return invested_total, stock_value, portfolio_value, avg_cost, roic


def simulate_arrays_for_theta(
    theta,
    initial_cash,
    config,
    prices,
    drawdowns,
    predicted_returns,
    volatilities,
):
    return _simulate_arrays_for_theta(
        theta,
        initial_cash,
        config["max_daily_fraction"],
        config["min_daily_investment"],
        config["cash_deployment_penalty"],
        prices,
        drawdowns,
        predicted_returns,
        volatilities,
    )


def _simulate_arrays_for_theta_python(
    theta,
    initial_cash,
    max_daily_fraction,
    min_daily_investment,
    cash_deployment_penalty,
    prices,
    drawdowns,
    predicted_returns,
    volatilities,
):
    g0, a, b, c = theta
    cash = initial_cash
    shares = 0.0

    for price, drawdown, predicted_return, volatility in zip(
        prices,
        drawdowns,
        predicted_returns,
        volatilities,
    ):
        if cash <= 0:
            invest = 0.0
        else:
            daily_fraction = g0 + a * drawdown + b * predicted_return - c * volatility
            daily_fraction = max(0.0, daily_fraction)
            daily_fraction = min(daily_fraction, max_daily_fraction)
            invest = min(max(cash * daily_fraction, min_daily_investment), cash)

        shares += invest / price if price > 0 else 0.0
        cash -= invest

    invested_total = initial_cash - cash
    stock_value = shares * prices[-1]
    portfolio_value = stock_value + cash

    # Portfolio growth objective:
    # G(theta) = V(theta) / C0 - 1
    # This optimizes total portfolio growth instead of return on invested capital.
    portfolio_growth = portfolio_value / initial_cash - 1.0

    cash_ratio = cash / initial_cash
    cash_penalty = cash_ratio ** 2

    return -portfolio_growth + cash_deployment_penalty * cash_penalty


if njit is not None:
    _simulate_arrays_for_theta = njit(cache=True)(_simulate_arrays_for_theta_python)
else:
    _simulate_arrays_for_theta = _simulate_arrays_for_theta_python


def simulate_window_for_theta(window_df, theta, initial_cash, config):
    return simulate_arrays_for_theta(
        theta,
        initial_cash,
        config,
        window_df["close"].to_numpy(dtype=float),
        window_df["drawdown"].to_numpy(dtype=float),
        window_df["predicted_return"].to_numpy(dtype=float),
        window_df["volatility_20d"].to_numpy(dtype=float),
    )


def solve_theta_for_day(history_df, initial_cash, config, workers=None):
    if workers is None:
        workers = config["optimizer_workers"]

    result = differential_evolution(
        ThetaObjective(history_df, initial_cash, config),
        bounds=BOUNDS,
        seed=config["random_state"],
        polish=False,
        maxiter=config.get("optimizer_maxiter", 20),
        popsize=config.get("optimizer_popsize", 8),
        updating="deferred",
        workers=workers,
    )

    return result.x


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def build_signal_sell_row(feature_df, signal_date, predicted_return, config):
    signal_context = feature_df.loc[feature_df.index <= signal_date].tail(80).copy()
    signal_context["predicted_return"] = 0.0
    signal_context.loc[signal_date, "predicted_return"] = predicted_return
    signal_sell_df = add_inverse_sell_scores(signal_context, config)

    if not signal_sell_df.empty and signal_date in signal_sell_df.index:
        return signal_sell_df.loc[signal_date]

    signal_row = feature_df.loc[signal_date].copy()
    signal_row["predicted_return"] = predicted_return
    for column in (
        "near_high_score",
        "overextension_score",
        "runup_score",
        "momentum_weakening_score",
        "drawdown_score",
        "new_high_score",
        "predicted_downside_score",
        "peak_score",
        "peak_near_high_component",
        "peak_runup_component",
        "peak_overextension_component",
    ):
        signal_row[column] = 0.0
    return signal_row


def compute_sell_state(cash, shares, price):
    stock_value = shares * price
    portfolio_value = cash + stock_value
    return stock_value, portfolio_value


def _clip01(value):
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


def add_sell_signals(df):
    signal_df = df.copy()
    close = signal_df["close"].astype(float)
    rolling_high = close.rolling(60, min_periods=5).max()
    rolling_low = close.rolling(60, min_periods=5).min()
    moving_average = close.rolling(20, min_periods=5).mean()
    momentum_5d = close.pct_change(5)
    momentum_20d = signal_df.get("momentum_20d", close.pct_change(20)).astype(float)

    high_distance = (rolling_high - close) / rolling_high.replace(0, np.nan)
    range_position = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    overextension = (close - moving_average) / moving_average.replace(0, np.nan)

    signal_df["near_high_score"] = (1.0 - high_distance.fillna(1.0) / 0.08).clip(0.0, 1.0)
    signal_df["overextension_score"] = (overextension.fillna(0.0) / 0.12).clip(0.0, 1.0)
    signal_df["runup_score"] = range_position.fillna(0.0).clip(0.0, 1.0)
    signal_df["momentum_weakening_score"] = (
        (momentum_20d.fillna(0.0).clip(lower=0.0) - momentum_5d.fillna(0.0)) / 0.12
    ).clip(0.0, 1.0)
    signal_df["drawdown_score"] = signal_df.get("drawdown", 0.0)
    signal_df["drawdown_score"] = signal_df["drawdown_score"].fillna(0.0).clip(0.0, 1.0)

    return signal_df


def add_inverse_sell_scores(feature_df, config):
    sell_df = add_sell_signals(feature_df).copy()
    if sell_df.empty:
        return sell_df

    close = sell_df["close"].astype(float)
    prior_high = close.rolling(60, min_periods=5).max().shift(1).replace(0, np.nan)

    runup_scale = max(_safe_float(config.get("sell_runup_scale", 0.20), 0.20), 1e-9)
    overextension_scale = max(
        _safe_float(config.get("sell_overextension_scale", 0.08), 0.08),
        1e-9,
    )
    new_high_scale = max(_safe_float(config.get("sell_new_high_scale", 0.15), 0.15), 1e-9)
    downside_scale = max(_safe_float(config.get("sell_downside_scale", 0.03), 0.03), 1e-9)
    min_near_high = _safe_float(config.get("sell_min_near_high_score", 0.70), 0.70)

    sell_df["peak_near_high_component"] = sell_df["near_high_score"].apply(_clip01)
    sell_df["peak_runup_component"] = (
        sell_df["runup_score"].astype(float) / runup_scale
    ).clip(lower=0.0, upper=1.0)
    sell_df["peak_overextension_component"] = (
        sell_df["overextension_score"].astype(float) / overextension_scale
    ).clip(lower=0.0, upper=1.0)
    sell_df["new_high_score"] = ((close / prior_high - 1.0) / new_high_scale).clip(
        lower=0.0,
        upper=1.0,
    ).fillna(0.0)
    sell_df["predicted_downside_score"] = (
        -sell_df["predicted_return"].astype(float) / downside_scale
    ).clip(lower=0.0, upper=1.0)

    peak_score = (
        0.35 * sell_df["peak_near_high_component"]
        + 0.25 * sell_df["peak_runup_component"]
        + 0.25 * sell_df["peak_overextension_component"]
        + 0.15 * sell_df["new_high_score"]
    )
    peak_score = peak_score.where(sell_df["near_high_score"] >= min_near_high, 0.0)
    sell_df["peak_score"] = peak_score.clip(lower=0.0, upper=1.0)

    required = [
        "peak_score",
        "predicted_downside_score",
        "new_high_score",
        "volatility_20d",
        "predicted_return",
    ]
    sell_df[required] = sell_df[required].replace([np.inf, -np.inf], np.nan)
    return sell_df.dropna(subset=required).copy()


def empty_daily_sell(config):
    return {
        "shares_to_sell": 0.0,
        "daily_sell_fraction": 0.0,
        "target_daily_sell_fraction": 0.0,
        "sell_allowed": False,
        "hold_reason": "no_shares",
        "sell_strength": 0.0,
        "adjusted_sell_strength": 0.0,
        "user_sell_strength": float(config.get("sell_strength", 1.0)),
        "raw_sell_strength": 0.0,
        "near_high_gate": False,
        "runup_gate": False,
        "weakening_gate": False,
        "overextension_gate": False,
        "prediction_gate": False,
        "strong_climb_gate": False,
        "peak_score": 0.0,
        "predicted_downside_score": 0.0,
    }


def compute_daily_sell(theta, shares, config, row):
    """Pure inverse of the buy investment fraction.

    Buy:  g0 + a * drawdown + b * predicted_return - c * volatility
    Sell: g0 + a * peak_score + b * predicted_downside - c * volatility
    """
    shares = max(0.0, float(shares))
    g0, peak_weight, predicted_downside_weight, volatility_weight = theta

    peak_score = _safe_float(row.get("peak_score", 0.0), 0.0)
    predicted_downside_score = _safe_float(row.get("predicted_downside_score", 0.0), 0.0)
    volatility_value = _safe_float(row.get("volatility_20d", 0.0), 0.0)
    predicted_return_value = _safe_float(row.get("predicted_return", 0.0), 0.0)

    runup_score = _safe_float(row.get("runup_score", 0.0), 0.0)
    overextension_score = _safe_float(row.get("overextension_score", 0.0), 0.0)
    momentum_weakening_score = _safe_float(row.get("momentum_weakening_score", 0.0), 0.0)

    near_high_gate = peak_score > 0.0
    runup_gate = runup_score >= _safe_float(config.get("sell_min_runup_score", 0.03), 0.03)
    weakening_gate = momentum_weakening_score >= _safe_float(
        config.get("sell_min_weakening_score", 0.01),
        0.01,
    )
    overextension_gate = overextension_score >= _safe_float(
        config.get("sell_min_overextension_score", 0.0),
        0.0,
    )
    prediction_gate = predicted_return_value <= _safe_float(
        config.get("sell_max_predicted_return", 0.001),
        0.001,
    )
    strong_climb_gate = predicted_return_value >= _safe_float(
        config.get("sell_strong_climb_return", 0.003),
        0.003,
    )

    user_sell_strength = max(0.0, _safe_float(config.get("sell_strength", 1.0), 1.0))
    max_fraction = max(0.0, _safe_float(config.get("max_daily_sell_fraction", 0.03), 0.03))

    if shares <= 0:
        result = empty_daily_sell(config)
        result.update(
            {
                "shares_remaining": 0.0,
                "near_high_gate": near_high_gate,
                "runup_gate": runup_gate,
                "weakening_gate": weakening_gate,
                "overextension_gate": overextension_gate,
                "prediction_gate": prediction_gate,
                "strong_climb_gate": strong_climb_gate,
                "user_sell_strength": user_sell_strength,
                "peak_score": peak_score,
                "predicted_downside_score": predicted_downside_score,
            }
        )
        return result

    if peak_score <= 0.0 or max_fraction <= 0.0:
        hold_reason = "not_near_high" if peak_score <= 0.0 else "max_sell_fraction_zero"
        return {
            "shares_to_sell": 0.0,
            "daily_sell_fraction": 0.0,
            "shares_remaining": shares,
            "target_daily_sell_fraction": 0.0,
            "sell_allowed": False,
            "hold_reason": hold_reason,
            "near_high_gate": near_high_gate,
            "runup_gate": runup_gate,
            "weakening_gate": weakening_gate,
            "overextension_gate": overextension_gate,
            "prediction_gate": prediction_gate,
            "strong_climb_gate": strong_climb_gate,
            "sell_strength": 0.0,
            "adjusted_sell_strength": 0.0,
            "user_sell_strength": user_sell_strength,
            "raw_sell_strength": 0.0,
            "peak_score": peak_score,
            "predicted_downside_score": predicted_downside_score,
        }

    raw_fraction = (
        float(g0)
        + float(peak_weight) * peak_score
        + float(predicted_downside_weight) * predicted_downside_score
        - float(volatility_weight) * volatility_value
    )
    raw_fraction = max(0.0, raw_fraction)
    target_fraction = min(raw_fraction * user_sell_strength, max_fraction)

    shares_to_sell = min(shares * target_fraction, shares)
    min_daily_sell_shares = _safe_float(config.get("min_daily_sell_shares", 0.0), 0.0)
    if 0.0 < shares_to_sell < min_daily_sell_shares:
        shares_to_sell = min(min_daily_sell_shares, shares)

    actual_fraction = shares_to_sell / shares if shares > 0 else 0.0
    sell_allowed = shares_to_sell > 0.0
    hold_reason = "sell_allowed" if sell_allowed else "sell_fraction_zero"
    sell_strength = target_fraction / max_fraction if max_fraction > 0 else 0.0
    raw_sell_strength = raw_fraction / max_fraction if max_fraction > 0 else 0.0

    return {
        "shares_to_sell": shares_to_sell,
        "daily_sell_fraction": actual_fraction,
        "shares_remaining": shares - shares_to_sell,
        "target_daily_sell_fraction": target_fraction,
        "sell_allowed": sell_allowed,
        "hold_reason": hold_reason,
        "sell_strength": max(0.0, min(1.0, sell_strength)),
        "adjusted_sell_strength": max(0.0, min(1.0, sell_strength)),
        "user_sell_strength": user_sell_strength,
        "raw_sell_strength": max(0.0, raw_sell_strength),
        "near_high_gate": near_high_gate,
        "runup_gate": runup_gate,
        "weakening_gate": weakening_gate,
        "overextension_gate": overextension_gate,
        "prediction_gate": prediction_gate,
        "strong_climb_gate": strong_climb_gate,
        "peak_score": peak_score,
        "predicted_downside_score": predicted_downside_score,
    }


def simulate_sell_window_for_theta(window_df, theta, initial_shares, config):
    shares = max(0.0, float(initial_shares))
    cash = 0.0

    if shares <= 0 or window_df.empty:
        return 0.0

    first_price = float(window_df["close"].iloc[0])
    penalty = 0.0
    prices = window_df["close"].to_numpy(dtype=float)
    rows = list(window_df.iterrows())

    for row_index, (_, row) in enumerate(rows):
        price = float(row["close"])
        if shares <= 0:
            break

        sale = compute_daily_sell(theta, shares, config, row)
        shares_to_sell = sale["shares_to_sell"]
        cash += shares_to_sell * price
        shares -= shares_to_sell

        if shares_to_sell > 0:
            penalty += float(config.get("sell_day_penalty", 0.0))
            if row_index + 1 < len(prices) and price > 0:
                next_return = prices[row_index + 1] / price - 1.0
                if next_return > 0:
                    penalty += (
                        float(config.get("sell_positive_next_day_penalty", 0.0))
                        * sale["daily_sell_fraction"]
                        * next_return
                    )

    final_price = float(window_df["close"].iloc[-1])
    terminal_value = cash + shares * final_price
    initial_value = initial_shares * first_price

    objective = -(terminal_value / initial_value) if initial_value > 0 else -terminal_value
    retention_penalty_weight = float(config.get("sell_share_retention_penalty", 0.0))
    retention_penalty = retention_penalty_weight * (shares / initial_shares) ** 2
    return objective + penalty + retention_penalty


def solve_sell_theta_for_day(history_df, initial_shares, config, workers=None):
    if workers is None:
        workers = config["optimizer_workers"]

    result = differential_evolution(
        SellThetaObjective(history_df, initial_shares, config),
        bounds=SELL_BOUNDS,
        seed=config["random_state"],
        polish=False,
        maxiter=config.get("optimizer_maxiter", 20),
        popsize=config.get("optimizer_popsize", 8),
        updating="deferred",
        workers=workers,
    )

    return result.x


def train_prediction_model(feature_df, config):
    target = f"future_return_{config['prediction_days']}d"
    training_df = feature_df.copy()
    training_df[target] = (
        training_df["close"].shift(-config["prediction_days"]) / training_df["close"] - 1
    )
    training_df = training_df.dropna(subset=[target]).copy()

    min_training_rows = int(config.get("min_training_rows", 30))
    if len(training_df) < min_training_rows:
        raise ValueError(
            f"Not enough usable training rows for this ticker/date range. "
            f"Need at least {min_training_rows}, found {len(training_df)}."
        )

    X = training_df[FEATURES]
    y = training_df[target]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=config["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model



def train_prediction_model_until(feature_df, current_date, config):
    """Train using only data strictly before current_date.

    This prevents look-ahead bias during backtesting. When predicting date t,
    the model can only use rows with dates earlier than t.
    """
    historical_df = feature_df.loc[feature_df.index < current_date].copy()
    return train_prediction_model(historical_df, config)


def build_prediction_chart(plot_df, ticker, prediction_days):
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Predicted Price",
                "x": plot_df["date"].tolist(),
                "y": plot_df["predicted_future_price"].round(4).tolist(),
                "line": {"color": "#00d9ff", "width": 3},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Actual Price",
                "x": plot_df["date"].tolist(),
                "y": plot_df["actual_future_price"].round(4).tolist(),
                "line": {"color": "#ffd700", "width": 3},
            },
        ],
        "layout": {
            "paper_bgcolor": "#151b2f",
            "plot_bgcolor": "#151b2f",
            "font": {"color": "#ffffff"},
            "xaxis": {"title": "Date", "gridcolor": "#2a3347"},
            "yaxis": {"title": "Price ($)", "gridcolor": "#2a3347"},
            "legend": {"orientation": "h"},
            "margin": {"l": 56, "r": 24, "t": 24, "b": 48},
        },
    }


def build_strategy_chart(result_df):
    metric_options = [
        ("portfolio_value", "Portfolio Value ($)"),
        ("stock_value", "Stock Value ($)"),
        ("avg_cost", "Average Cost ($)"),
        ("daily_investment", "Daily Investment ($)"),
        ("cash", "Remaining Cash ($)"),
        ("return_on_invested_capital", "Return on Invested Capital"),
    ]
    data = []

    for metric_index, (metric, _) in enumerate(metric_options):
        visible = metric_index == 0
        data.extend(
            [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Lucky Stock",
                    "x": result_df["date"].tolist(),
                    "y": result_df[f"tool_{metric}"].round(4).tolist(),
                    "line": {"color": "#00d9ff", "width": 3},
                    "visible": True if visible else False,
                    "legendgroup": "Lucky Stock",
                    "showlegend": visible,
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "DCA",
                    "x": result_df["date"].tolist(),
                    "y": result_df[f"dca_{metric}"].round(4).tolist(),
                    "line": {"color": "#ffd700", "width": 3},
                    "visible": True if visible else False,
                    "legendgroup": "DCA",
                    "showlegend": visible,
                },
            ]
        )

    return {
        "data": data,
        "layout": {
            "paper_bgcolor": "#151b2f",
            "plot_bgcolor": "#151b2f",
            "font": {"color": "#ffffff"},
            "xaxis": {"title": "Date", "gridcolor": "#2a3347"},
            "yaxis": {"title": "Portfolio Value ($)", "gridcolor": "#2a3347"},
            "legend": {"orientation": "h"},
            "margin": {"l": 56, "r": 24, "t": 24, "b": 48},
        },
    }


def build_sell_strategy_chart(result_df):
    metric_options = [
        ("cash", "Sale Total ($)"),
        ("stock_value", "Unsold Share Value ($)"),
        ("avg_sold_value", "Average Sold Price ($)"),
        ("daily_shares_sold", "Daily Shares Sold"),
        ("shares", "Remaining Shares"),
        ("portfolio_value", "Total Value ($)"),
    ]
    data = []

    for metric_index, (metric, _) in enumerate(metric_options):
        visible = metric_index == 0
        data.extend(
            [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Lucky Stock",
                    "x": result_df["date"].tolist(),
                    "y": result_df[f"tool_{metric}"].round(4).tolist(),
                    "line": {"color": "#00d9ff", "width": 3},
                    "visible": True if visible else False,
                    "legendgroup": "Lucky Stock",
                    "showlegend": visible,
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Linear Sell",
                    "x": result_df["date"].tolist(),
                    "y": result_df[f"linear_{metric}"].round(4).tolist(),
                    "line": {"color": "#ffd700", "width": 3},
                    "visible": True if visible else False,
                    "legendgroup": "Linear Sell",
                    "showlegend": visible,
                },
            ]
        )

    return {
        "data": data,
        "layout": {
            "paper_bgcolor": "#151b2f",
            "plot_bgcolor": "#151b2f",
            "font": {"color": "#ffffff"},
            "xaxis": {"title": "Date", "gridcolor": "#2a3347"},
            "yaxis": {"title": "Sale Total ($)", "gridcolor": "#2a3347"},
            "legend": {"orientation": "h"},
            "margin": {"l": 56, "r": 24, "t": 24, "b": 48},
        },
    }


def run_simulation(ticker, start_date, end_date, total_cash, data_dir="datasets"):
    from backtest import run_buy_simulation

    ticker = ticker.upper().strip()
    dataset_path = find_or_build_dataset(ticker, data_dir)
    feature_df = pd.read_csv(dataset_path, index_col=0, parse_dates=True).dropna(
        subset=FEATURES
    )

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    start_ts = pd.Timestamp(start_date)
    if start_ts not in feature_df.index:
        later_dates = feature_df.index[feature_df.index >= start_ts]
        if len(later_dates) == 0:
            raise ValueError("Start date is after the available market data.")
        start_ts = later_dates[0]

    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("End date must be after the start date.")

    sim_df = feature_df.loc[start_ts:end_ts]
    if sim_df.empty:
        raise ValueError("No market rows found for that simulation period.")
    if len(sim_df) < MIN_SIMULATION_TRADING_DAYS:
        raise ValueError("Simulation period must include at least 20 trading days.")
    if len(sim_df) > MAX_SIMULATION_TRADING_DAYS:
        raise ValueError("Simulation period must include 100 trading days or fewer.")

    result = run_buy_simulation(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        total_cash=total_cash,
        data_dir=data_dir,
        config_overrides={"buy_policy": "target_weight"},
    )

    return {
        "summary": result["summary"],
        "strategyChart": result["strategyChart"],
    }


def run_sell_simulation(
    ticker,
    start_date,
    end_date,
    initial_shares,
    data_dir="datasets",
    config_overrides=None,
):
    from backtest import run_sell_simulation as run_backtest_sell_simulation

    result = run_backtest_sell_simulation(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_shares=initial_shares,
        data_dir=data_dir,
        config_overrides=config_overrides,
    )

    rows_df = pd.DataFrame(result["rows"])
    if rows_df.empty:
        raise ValueError("No usable sell simulation rows were produced.")

    initial_shares = float(initial_shares)
    result_df = pd.DataFrame(
        {
            "date": rows_df["date"],
            "tool_cash": rows_df["lucky_cash"],
            "tool_stock_value": rows_df["lucky_stock_value"],
            "tool_daily_shares_sold": rows_df["shares_to_sell"],
            "tool_shares": rows_df["lucky_shares"],
            "tool_portfolio_value": rows_df["lucky_portfolio_value"],
            "linear_cash": rows_df["linear_cash"],
            "linear_stock_value": rows_df["linear_stock_value"],
            "linear_daily_shares_sold": rows_df["linear_shares_to_sell"],
            "linear_shares": rows_df["linear_shares"],
            "linear_portfolio_value": rows_df["linear_portfolio_value"],
        }
    )
    tool_sold = initial_shares - result_df["tool_shares"]
    linear_sold = initial_shares - result_df["linear_shares"]
    result_df["tool_avg_sold_value"] = (
        result_df["tool_cash"] / tool_sold.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    result_df["linear_avg_sold_value"] = (
        result_df["linear_cash"] / linear_sold.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    summary_source = result["summary"]
    tool_sale_total = float(result_df["tool_cash"].iloc[-1])
    linear_sale_total = float(result_df["linear_cash"].iloc[-1])
    tool_shares_sold = float(summary_source["sharesSold"])
    sell_history = result.get("sellHistory", [])
    avg_sell_fraction = (
        float(rows_df.loc[rows_df["shares_to_sell"] > 0, "daily_sell_fraction"].mean())
        if (rows_df["shares_to_sell"] > 0).any()
        else 0.0
    )
    max_sell_fraction = (
        float(rows_df["daily_sell_fraction"].max())
        if "daily_sell_fraction" in rows_df
        else 0.0
    )

    summary = {
        **summary_source,
        "startValue": summary_source.get("initialValue"),
        "toolFinalValue": round(tool_sale_total, 2),
        "linearSellFinalValue": round(linear_sale_total, 2),
        "toolVsLinearSell": round(tool_sale_total - linear_sale_total, 2),
        "toolCashRealized": round(tool_sale_total, 2),
        "toolSharesRemaining": round(float(result_df["tool_shares"].iloc[-1]), 6),
        "toolSharesSold": round(tool_shares_sold, 6),
        "avgSellFractionOnSellDays": round(avg_sell_fraction, 6),
        "maxSellFraction": round(max_sell_fraction, 6),
    }

    return {
        "summary": summary,
        "comprehensiveSummary": result.get("comprehensiveSummary"),
        "strategyChart": build_sell_strategy_chart(result_df),
        "sellHistory": sell_history,
        "predictionHistory": result.get("predictionHistory", []),
        "rows": result_df.to_dict("records"),
    }
