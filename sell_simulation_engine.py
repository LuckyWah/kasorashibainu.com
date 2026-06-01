import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from compute_investment import FEATURES, add_sell_signals
from data_builder import build_adaptive_dataset


CPU_COUNT = os.cpu_count() or 1

# Pure inverse of the buy model: 4 optimized parameters, no Kalman layer.
# Buy:  g0 + a * drawdown + b * predicted_return - c * volatility
# Sell: g0 + a * peak_score + b * predicted_downside - c * volatility
SELL_BOUNDS = [
    (0.0000, 0.0100),  # g0
    (0.0000, 0.0500),  # peak_score
    (0.0000, 0.3000),  # predicted_downside
    (0.0000, 0.0100),  # volatility
]

DEFAULT_CONFIG = {
    "prediction_days": 1,
    "max_daily_sell_fraction": 0.03,
    "min_daily_sell_shares": 0.0,

    # Local-peak feature construction.
    # These are not optimized parameters; they only normalize raw signals.
    "sell_min_near_high_score": 0.70,
    "sell_runup_scale": 0.20,
    "sell_overextension_scale": 0.08,
    "sell_new_high_scale": 0.15,
    "sell_downside_scale": 0.03,

    # Keep these for diagnostics only. They no longer hard-veto selling.
    "sell_min_runup_score": 0.03,
    "sell_min_weakening_score": 0.01,
    "sell_min_overextension_score": 0.0,
    "sell_max_predicted_return": 0.001,
    "sell_strong_climb_return": 0.003,

    # User-facing aggressiveness multiplier.
    "sell_strength": 1.0,

    # Objective penalties.
    "sell_day_penalty": 0.0005,
    "sell_positive_next_day_penalty": 0.25,

    # Pure inverse of buy cash-deployment penalty, but weaker by default.
    # Buy penalizes unused cash; sell penalizes never reducing shares.
    # Keep this small for the first baseline test so it does not force early liquidation.
    "sell_share_retention_penalty": 0.05,

    # Faster realistic backtest settings.
    # The model and theta are trained only on past data, but not refreshed every day.
    "rolling_opt_window": 120,
    "model_refresh_days": 20,
    "theta_refresh_days": 5,

    "optimizer_workers": 1,
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "min_training_rows": 30,
    "random_state": 42,
    "optimizer_maxiter": 8,
    "optimizer_popsize": 5,
}


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


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _clip01(value):
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


def find_or_build_dataset(ticker, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper().strip()
    dataset_path = data_dir / f"{ticker}.csv"

    if dataset_path.exists():
        return dataset_path

    return build_adaptive_dataset(ticker, output_dir=data_dir)


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
    """Train using only data strictly before current_date."""
    historical_df = feature_df.loc[feature_df.index < current_date].copy()
    return train_prediction_model(historical_df, config)


def add_inverse_sell_scores(feature_df, config):
    """Add inverse-buy sell features.

    The core sell signal is peak_score, the inverse of drawdown.
    It becomes large when price is near a local high, after a strong runup,
    overextended above the moving average, and/or making a higher high.
    """
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

    # Keep the tool as a local-peak seller, not a panic seller in decline.
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


def compute_daily_sell_inverse(theta, shares, config, row_signals):
    """Inverse of compute_daily_investment.

    Buy model:
        fraction = g0 + a * drawdown + b * predicted_return - c * volatility

    Sell model:
        fraction = g0 + a * peak_score + b * predicted_downside - c * volatility

    The only hard requirement is a nonzero local peak score. Prediction and
    momentum weakening do not veto selling.
    """
    shares = max(0.0, float(shares))
    g0, peak_weight, predicted_downside_weight, volatility_weight = theta

    peak_score = _safe_float(row_signals.get("peak_score", 0.0), 0.0)
    predicted_downside_score = _safe_float(row_signals.get("predicted_downside_score", 0.0), 0.0)
    volatility_value = _safe_float(row_signals.get("volatility_20d", 0.0), 0.0)
    predicted_return_value = _safe_float(row_signals.get("predicted_return", 0.0), 0.0)

    near_high_score = _safe_float(row_signals.get("near_high_score", 0.0), 0.0)
    runup_score = _safe_float(row_signals.get("runup_score", 0.0), 0.0)
    overextension_score = _safe_float(row_signals.get("overextension_score", 0.0), 0.0)
    momentum_weakening_score = _safe_float(row_signals.get("momentum_weakening_score", 0.0), 0.0)

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
        return {
            "shares_to_sell": 0.0,
            "daily_sell_fraction": 0.0,
            "shares_remaining": 0.0,
            "target_daily_sell_fraction": 0.0,
            "sell_allowed": False,
            "hold_reason": "no_shares",
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
        "near_high_gate": near_high_gate,
        "runup_gate": runup_gate,
        "weakening_gate": weakening_gate,
        "overextension_gate": overextension_gate,
        "prediction_gate": prediction_gate,
        "strong_climb_gate": strong_climb_gate,
        "sell_strength": max(0.0, min(1.0, sell_strength)),
        "adjusted_sell_strength": max(0.0, min(1.0, sell_strength)),
        "user_sell_strength": user_sell_strength,
        "raw_sell_strength": max(0.0, raw_sell_strength),
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

        sale = compute_daily_sell_inverse(theta, shares, config, row)
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

    # Inverse of buy model's cash-deployment penalty.
    # This prevents the optimizer from learning the trivial pure-hold solution
    # when the ticker rallies during the optimization window.
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


def build_sell_strategy_chart(result_df):
    metric_options = [
        ("portfolio_value", "Total Value ($)"),
        ("cash", "Realized Cash ($)"),
        ("shares", "Remaining Shares"),
        ("daily_shares_sold", "Shares Sold"),
        ("stock_value", "Unsold Share Value ($)"),
    ]
    data = []

    series = [
        ("tool", "Lucky Sell", "#ff7777"),
        ("hold", "Hold", "#00d9ff"),
        ("linear", "Linear Sell", "#ffd700"),
    ]

    for metric_index, (metric, _) in enumerate(metric_options):
        visible = metric_index == 0
        for prefix, name, color in series:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": name,
                    "x": result_df["date"].tolist(),
                    "y": result_df[f"{prefix}_{metric}"].round(4).tolist(),
                    "line": {"color": color, "width": 3},
                    "visible": True if visible else False,
                    "legendgroup": name,
                    "showlegend": visible,
                }
            )

    return {
        "data": data,
        "layout": {
            "paper_bgcolor": "#151b2f",
            "plot_bgcolor": "#151b2f",
            "font": {"color": "#ffffff"},
            "xaxis": {"title": "Date", "gridcolor": "#2a3347"},
            "yaxis": {"title": "Total Value ($)", "gridcolor": "#2a3347"},
            "legend": {"orientation": "h"},
            "margin": {"l": 56, "r": 24, "t": 24, "b": 48},
        },
    }


def build_sell_prediction_chart(prediction_df, ticker, prediction_days):
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Predicted Price",
                "x": prediction_df["date"].tolist(),
                "y": prediction_df["predicted_future_price"].round(4).tolist(),
                "line": {"color": "#00d9ff", "width": 3},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Actual Price",
                "x": prediction_df["date"].tolist(),
                "y": prediction_df["actual_future_price"].round(4).tolist(),
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
            "title": f"{ticker} {prediction_days}-day sell model price predictions",
        },
    }


def compute_sell_state(cash, shares, price):
    stock_value = shares * price
    portfolio_value = cash + stock_value
    return stock_value, portfolio_value


def max_drawdown(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = np.where(running_max > 0, values / running_max - 1.0, 0.0)
    return float(drawdowns.min())


def safe_round(value, digits=4):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, digits)


def series_stats(series, digits=6):
    if series is None or len(series) == 0:
        return {"min": None, "max": None, "mean": None, "median": None}

    clean = pd.Series(series).dropna()
    if clean.empty:
        return {"min": None, "max": None, "mean": None, "median": None}

    return {
        "min": safe_round(clean.min(), digits),
        "max": safe_round(clean.max(), digits),
        "mean": safe_round(clean.mean(), digits),
        "median": safe_round(clean.median(), digits),
    }


def run_sell_simulation(
    ticker,
    start_date,
    end_date,
    initial_shares,
    data_dir="datasets",
    config_overrides=None,
):
    config = DEFAULT_CONFIG.copy()
    config.update(config_overrides or {})
    ticker = ticker.upper().strip()
    initial_shares = float(initial_shares)

    if initial_shares <= 0:
        raise ValueError("initial_shares must be greater than 0.")

    dataset_path = find_or_build_dataset(ticker, data_dir)
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    prediction_days = int(config["prediction_days"])
    feature_df["actual_future_price"] = feature_df["close"].shift(-prediction_days)

    start_ts = pd.Timestamp(start_date)
    if start_ts not in feature_df.index:
        later_dates = feature_df.index[feature_df.index >= start_ts]
        if len(later_dates) == 0:
            raise ValueError("Start date is after the available market data.")
        start_ts = later_dates[0]

    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("End date must be after the start date.")

    sim_df = feature_df.loc[start_ts:end_ts].copy()

    if sim_df.empty:
        raise ValueError("No market rows found for that simulation period.")
    if len(sim_df) < 20:
        raise ValueError("Simulation period must include at least 20 trading days.")

    tool_cash = 0.0
    tool_shares = initial_shares
    theta = np.zeros(len(SELL_BOUNDS), dtype=float)
    theta_refresh_days = max(1, int(config["theta_refresh_days"]))

    prediction_rows = []
    result_rows = []
    theta_rows = []
    model = None
    model_signal_date = None
    model_refresh_days = max(1, int(config.get("model_refresh_days", 20)))

    for day_index, (current_date, row) in enumerate(sim_df.iterrows()):
        price = float(row["close"])

        signal_position = feature_df.index.get_loc(current_date) - 1
        if signal_position < 0:
            continue

        signal_date = feature_df.index[signal_position]
        signal_row = feature_df.iloc[signal_position]

        # Today's sale uses only the previous trading day's finalized features.
        if model is None or day_index % model_refresh_days == 0:
            model = train_prediction_model_until(feature_df, signal_date, config)
            model_signal_date = signal_date

        predicted_return = float(model.predict(signal_row[FEATURES].to_frame().T)[0])
        predicted_future_price = float(signal_row["close"]) * (1.0 + predicted_return)

        prediction_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "model_signal_date": model_signal_date.strftime("%Y-%m-%d"),
                "predicted_future_price": predicted_future_price,
                "actual_future_price": signal_row["actual_future_price"],
                "predicted_return": predicted_return,
            }
        )

        signal_sell_row = build_signal_sell_row(feature_df, signal_date, predicted_return, config)

        if tool_shares > 0 and day_index % theta_refresh_days == 0:
            theta_history_df = feature_df.loc[feature_df.index < signal_date].tail(
                config["rolling_opt_window"]
            ).copy()

            theta_source_rows = len(theta_history_df)
            theta_status = "insufficient_history"
            if len(theta_history_df) >= 30:
                theta_history_df["predicted_return"] = model.predict(theta_history_df[FEATURES])
                theta_history_df = add_inverse_sell_scores(theta_history_df, config)
                if len(theta_history_df) >= 30:
                    theta = solve_sell_theta_for_day(theta_history_df, initial_shares, config)
                    theta_status = "optimized"
                else:
                    theta = np.zeros(len(SELL_BOUNDS), dtype=float)
                    theta_status = "insufficient_sell_signals"
            else:
                theta = np.zeros(len(SELL_BOUNDS), dtype=float)

            theta_rows.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "status": theta_status,
                    "history_rows": int(theta_source_rows),
                    "theta_g0": float(theta[0]),
                    "theta_peak_score": float(theta[1]),
                    "theta_predicted_downside": float(theta[2]),
                    "theta_volatility": float(theta[3]),
                }
            )

        sale = compute_daily_sell_inverse(theta, tool_shares, config, signal_sell_row)
        tool_daily_shares_sold = sale["shares_to_sell"]
        tool_cash_before = tool_cash
        tool_shares_before = tool_shares
        tool_cash += tool_daily_shares_sold * price
        tool_shares -= tool_daily_shares_sold
        tool_stock_value, tool_portfolio_value = compute_sell_state(tool_cash, tool_shares, price)

        result_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "model_signal_date": model_signal_date.strftime("%Y-%m-%d"),
                "close": price,
                "signal_close": float(signal_row["close"]),
                "predicted_return": predicted_return,
                "peak_score": float(signal_sell_row.get("peak_score", 0.0)),
                "predicted_downside_score": float(signal_sell_row.get("predicted_downside_score", 0.0)),
                "new_high_score": float(signal_sell_row.get("new_high_score", 0.0)),
                "near_high_score": float(signal_sell_row.get("near_high_score", 0.0)),
                "overextension_score": float(signal_sell_row.get("overextension_score", 0.0)),
                "runup_score": float(signal_sell_row.get("runup_score", 0.0)),
                "momentum_weakening_score": float(signal_sell_row.get("momentum_weakening_score", 0.0)),
                "drawdown_score": float(signal_sell_row.get("drawdown_score", 0.0)),
                "volatility_20d": float(signal_sell_row.get("volatility_20d", 0.0)),
                "theta_g0": float(theta[0]),
                "theta_peak_score": float(theta[1]),
                "theta_predicted_downside": float(theta[2]),
                "theta_volatility": float(theta[3]),
                "tool_cash_before_sale": tool_cash_before,
                "tool_shares_before_sale": tool_shares_before,
                "tool_daily_shares_sold": tool_daily_shares_sold,
                "tool_daily_sell_value": tool_daily_shares_sold * price,
                "tool_daily_sell_fraction": sale["daily_sell_fraction"],
                "tool_target_daily_sell_fraction": sale["target_daily_sell_fraction"],
                "tool_sell_allowed": bool(sale["sell_allowed"]),
                "tool_hold_reason": sale["hold_reason"],
                "tool_sell_strength": sale["sell_strength"],
                "tool_adjusted_sell_strength": sale["adjusted_sell_strength"],
                "tool_user_sell_strength": sale["user_sell_strength"],
                "tool_raw_sell_strength": sale["raw_sell_strength"],
                "tool_near_high_gate": bool(sale["near_high_gate"]),
                "tool_runup_gate": bool(sale["runup_gate"]),
                "tool_weakening_gate": bool(sale["weakening_gate"]),
                "tool_overextension_gate": bool(sale["overextension_gate"]),
                "tool_prediction_gate": bool(sale["prediction_gate"]),
                "tool_strong_climb_gate": bool(sale["strong_climb_gate"]),
                "tool_cash": tool_cash,
                "tool_shares": tool_shares,
                "tool_stock_value": tool_stock_value,
                "tool_portfolio_value": tool_portfolio_value,
            }
        )

    result_df = pd.DataFrame(result_rows)
    if result_df.empty:
        raise ValueError("No usable simulation rows were produced.")

    linear_shares = initial_shares
    linear_cash = 0.0
    linear_daily_shares = initial_shares / len(result_df)
    linear_rows = []

    hold_shares = initial_shares
    hold_cash = 0.0

    for _, row in result_df.iterrows():
        price = float(row["close"])

        linear_sell = min(linear_daily_shares, linear_shares)
        linear_cash += linear_sell * price
        linear_shares -= linear_sell
        linear_stock_value, linear_portfolio_value = compute_sell_state(
            linear_cash,
            linear_shares,
            price,
        )

        hold_stock_value, hold_portfolio_value = compute_sell_state(
            hold_cash,
            hold_shares,
            price,
        )

        linear_rows.append(
            {
                "linear_daily_shares_sold": linear_sell,
                "linear_cash": linear_cash,
                "linear_shares": linear_shares,
                "linear_stock_value": linear_stock_value,
                "linear_portfolio_value": linear_portfolio_value,
                "hold_daily_shares_sold": 0.0,
                "hold_cash": hold_cash,
                "hold_shares": hold_shares,
                "hold_stock_value": hold_stock_value,
                "hold_portfolio_value": hold_portfolio_value,
            }
        )

    result_df = pd.concat([result_df.reset_index(drop=True), pd.DataFrame(linear_rows)], axis=1)
    prediction_df = pd.DataFrame(prediction_rows).dropna(
        subset=["actual_future_price", "predicted_future_price"]
    )
    theta_df = pd.DataFrame(theta_rows)

    sell_history_columns = [
        "date",
        "signal_date",
        "close",
        "signal_close",
        "predicted_return",
        "peak_score",
        "predicted_downside_score",
        "new_high_score",
        "near_high_score",
        "overextension_score",
        "runup_score",
        "momentum_weakening_score",
        "drawdown_score",
        "volatility_20d",
        "tool_shares_before_sale",
        "tool_daily_shares_sold",
        "tool_daily_sell_value",
        "tool_daily_sell_fraction",
        "tool_target_daily_sell_fraction",
        "tool_sell_allowed",
        "tool_hold_reason",
        "tool_sell_strength",
        "tool_adjusted_sell_strength",
        "tool_user_sell_strength",
        "tool_raw_sell_strength",
        "tool_near_high_gate",
        "tool_runup_gate",
        "tool_weakening_gate",
        "tool_overextension_gate",
        "tool_prediction_gate",
        "tool_strong_climb_gate",
        "tool_cash",
        "tool_shares",
        "tool_portfolio_value",
        "theta_g0",
        "theta_peak_score",
        "theta_predicted_downside",
        "theta_volatility",
    ]
    sell_history_df = result_df.loc[
        result_df["tool_daily_shares_sold"] > 0,
        sell_history_columns,
    ].copy()

    no_sell_days = int((result_df["tool_daily_shares_sold"] <= 0).sum())
    sell_days = int((result_df["tool_daily_shares_sold"] > 0).sum())
    avg_sell_fraction = (
        float(sell_history_df["tool_daily_sell_fraction"].mean())
        if not sell_history_df.empty else 0.0
    )
    max_sell_fraction = (
        float(sell_history_df["tool_daily_sell_fraction"].max())
        if not sell_history_df.empty else 0.0
    )
    max_sell_value = (
        float(sell_history_df["tool_daily_sell_value"].max())
        if not sell_history_df.empty else 0.0
    )
    avg_sell_strength = (
        float(sell_history_df["tool_sell_strength"].mean())
        if not sell_history_df.empty else 0.0
    )
    avg_adjusted_sell_strength = (
        float(sell_history_df["tool_adjusted_sell_strength"].mean())
        if not sell_history_df.empty else 0.0
    )
    max_sell_strength = (
        float(sell_history_df["tool_sell_strength"].max())
        if not sell_history_df.empty else 0.0
    )
    max_adjusted_sell_strength = (
        float(sell_history_df["tool_adjusted_sell_strength"].max())
        if not sell_history_df.empty else 0.0
    )
    eligible_days = int(result_df["tool_sell_allowed"].sum())
    hold_gated_days = int((~result_df["tool_sell_allowed"]).sum())
    hold_reason_counts = result_df["tool_hold_reason"].value_counts().to_dict()
    strong_climb_blocked_days = int(result_df["tool_strong_climb_gate"].sum())
    low_pullback_signal_days = int((~result_df["tool_prediction_gate"]).sum())

    start_value = initial_shares * float(result_df["close"].iloc[0])
    tool_final_value = float(result_df["tool_portfolio_value"].iloc[-1])
    hold_final_value = float(result_df["hold_portfolio_value"].iloc[-1])
    linear_final_value = float(result_df["linear_portfolio_value"].iloc[-1])

    summary = {
        "ticker": ticker,
        "startDate": result_df["date"].iloc[0],
        "endDate": result_df["date"].iloc[-1],
        "tradingDays": int(len(result_df)),
        "initialShares": round(initial_shares, 6),
        "startValue": round(start_value, 2),
        "toolFinalValue": round(tool_final_value, 2),
        "holdFinalValue": round(hold_final_value, 2),
        "linearSellFinalValue": round(linear_final_value, 2),
        "toolVsHold": round(tool_final_value - hold_final_value, 2),
        "toolVsLinearSell": round(tool_final_value - linear_final_value, 2),
        "toolCashRealized": round(float(result_df["tool_cash"].iloc[-1]), 2),
        "toolSharesRemaining": round(float(result_df["tool_shares"].iloc[-1]), 6),
        "toolSharesSold": round(initial_shares - float(result_df["tool_shares"].iloc[-1]), 6),
        "sellDays": sell_days,
        "noSellDays": no_sell_days,
        "avgSellFractionOnSellDays": round(avg_sell_fraction, 6),
        "maxSellFraction": round(max_sell_fraction, 6),
        "avgSellStrengthOnSellDays": round(avg_sell_strength, 6),
        "avgAdjustedSellStrengthOnSellDays": round(avg_adjusted_sell_strength, 6),
        "maxSellStrength": round(max_sell_strength, 6),
        "maxAdjustedSellStrength": round(max_adjusted_sell_strength, 6),
        "maxDailySellValue": round(max_sell_value, 2),
        "eligibleSellDays": eligible_days,
        "holdGatedDays": hold_gated_days,
        "strongClimbBlockedDays": strong_climb_blocked_days,
    }

    signal_columns = [
        "predicted_return",
        "peak_score",
        "predicted_downside_score",
        "new_high_score",
        "near_high_score",
        "overextension_score",
        "runup_score",
        "momentum_weakening_score",
        "drawdown_score",
        "volatility_20d",
    ]
    theta_columns = [
        "theta_g0",
        "theta_peak_score",
        "theta_predicted_downside",
        "theta_volatility",
    ]
    sell_dates = sell_history_df["date"].tolist()
    first_sale = sell_history_df.iloc[0].to_dict() if not sell_history_df.empty else None
    last_sale = sell_history_df.iloc[-1].to_dict() if not sell_history_df.empty else None
    largest_sale = (
        sell_history_df.sort_values("tool_daily_sell_value", ascending=False).iloc[0].to_dict()
        if not sell_history_df.empty else None
    )
    final_row = result_df.iloc[-1]

    comprehensive_summary = {
        "run": {
            "ticker": ticker,
            "startDate": result_df["date"].iloc[0],
            "endDate": result_df["date"].iloc[-1],
            "tradingDays": int(len(result_df)),
            "initialShares": round(initial_shares, 6),
            "startPrice": round(float(result_df["close"].iloc[0]), 2),
            "endPrice": round(float(result_df["close"].iloc[-1]), 2),
            "priceReturn": round(float(result_df["close"].iloc[-1] / result_df["close"].iloc[0] - 1.0), 6),
            "maxDailySellFraction": float(config["max_daily_sell_fraction"]),
            "userSellStrength": float(config.get("sell_strength", 1.0)),
            "sellShareRetentionPenalty": float(config.get("sell_share_retention_penalty", 0.0)),
        },
        "performance": {
            "startValue": round(start_value, 2),
            "toolFinalValue": round(tool_final_value, 2),
            "holdFinalValue": round(hold_final_value, 2),
            "linearSellFinalValue": round(linear_final_value, 2),
            "toolReturn": round(tool_final_value / start_value - 1.0, 6),
            "holdReturn": round(hold_final_value / start_value - 1.0, 6),
            "linearSellReturn": round(linear_final_value / start_value - 1.0, 6),
            "toolVsHold": round(tool_final_value - hold_final_value, 2),
            "toolVsLinearSell": round(tool_final_value - linear_final_value, 2),
            "toolMaxDrawdown": round(max_drawdown(result_df["tool_portfolio_value"]), 6),
            "holdMaxDrawdown": round(max_drawdown(result_df["hold_portfolio_value"]), 6),
            "linearSellMaxDrawdown": round(max_drawdown(result_df["linear_portfolio_value"]), 6),
        },
        "selling": {
            "sellDays": sell_days,
            "noSellDays": no_sell_days,
            "eligibleSellDays": eligible_days,
            "holdGatedDays": hold_gated_days,
            "holdReasonCounts": hold_reason_counts,
            "strongClimbBlockedDays": strong_climb_blocked_days,
            "predictionBlockedDays": low_pullback_signal_days,
            "sellDayRate": round(sell_days / len(result_df), 6),
            "eligibleSellDayRate": round(eligible_days / len(result_df), 6),
            "firstSellDate": sell_dates[0] if sell_dates else None,
            "lastSellDate": sell_dates[-1] if sell_dates else None,
            "sharesSold": round(initial_shares - float(final_row["tool_shares"]), 6),
            "sharesRemaining": round(float(final_row["tool_shares"]), 6),
            "percentSharesSold": round((initial_shares - float(final_row["tool_shares"])) / initial_shares, 6),
            "cashRealized": round(float(final_row["tool_cash"]), 2),
            "unsoldShareValue": round(float(final_row["tool_stock_value"]), 2),
            "avgSellFractionOnSellDays": round(avg_sell_fraction, 6),
            "maxSellFraction": round(max_sell_fraction, 6),
            "avgSellStrengthOnSellDays": round(avg_sell_strength, 6),
            "avgAdjustedSellStrengthOnSellDays": round(avg_adjusted_sell_strength, 6),
            "maxSellStrength": round(max_sell_strength, 6),
            "maxAdjustedSellStrength": round(max_adjusted_sell_strength, 6),
            "avgDailySellValueOnSellDays": safe_round(
                sell_history_df["tool_daily_sell_value"].mean() if not sell_history_df.empty else 0.0,
                2,
            ),
            "maxDailySellValue": round(max_sell_value, 2),
            "largestSale": {
                "date": largest_sale["date"],
                "shares": safe_round(largest_sale["tool_daily_shares_sold"], 6),
                "value": safe_round(largest_sale["tool_daily_sell_value"], 2),
                "fraction": safe_round(largest_sale["tool_daily_sell_fraction"], 6),
                "price": safe_round(largest_sale["close"], 2),
            } if largest_sale else None,
            "firstSale": {
                "date": first_sale["date"],
                "shares": safe_round(first_sale["tool_daily_shares_sold"], 6),
                "value": safe_round(first_sale["tool_daily_sell_value"], 2),
                "fraction": safe_round(first_sale["tool_daily_sell_fraction"], 6),
                "price": safe_round(first_sale["close"], 2),
            } if first_sale else None,
            "lastSale": {
                "date": last_sale["date"],
                "shares": safe_round(last_sale["tool_daily_shares_sold"], 6),
                "value": safe_round(last_sale["tool_daily_sell_value"], 2),
                "fraction": safe_round(last_sale["tool_daily_sell_fraction"], 6),
                "price": safe_round(last_sale["close"], 2),
            } if last_sale else None,
        },
        "signals": {column: series_stats(result_df[column]) for column in signal_columns},
        "signalsOnSellDays": {
            column: series_stats(sell_history_df[column]) for column in signal_columns
        },
        "theta": {
            "refreshCount": int(len(theta_df)),
            "optimizedCount": int((theta_df["status"] == "optimized").sum()) if not theta_df.empty else 0,
            "statuses": theta_df["status"].value_counts().to_dict() if not theta_df.empty else {},
            "latest": theta_df.iloc[-1].to_dict() if not theta_df.empty else None,
            "stats": {column: series_stats(theta_df[column]) for column in theta_columns} if not theta_df.empty else {},
        },
        "baselines": {
            "hold": {
                "finalValue": round(hold_final_value, 2),
                "cash": round(float(final_row["hold_cash"]), 2),
                "shares": round(float(final_row["hold_shares"]), 6),
                "stockValue": round(float(final_row["hold_stock_value"]), 2),
            },
            "linearSell": {
                "finalValue": round(linear_final_value, 2),
                "cash": round(float(final_row["linear_cash"]), 2),
                "shares": round(float(final_row["linear_shares"]), 6),
                "stockValue": round(float(final_row["linear_stock_value"]), 2),
                "dailySharesSold": round(float(result_df["linear_daily_shares_sold"].iloc[0]), 6),
            },
        },
    }

    return {
        "summary": summary,
        "comprehensiveSummary": comprehensive_summary,
        "predictionChart": build_sell_prediction_chart(prediction_df, ticker, prediction_days),
        "strategyChart": build_sell_strategy_chart(result_df),
        "sellHistory": sell_history_df.to_dict("records"),
        "thetaHistory": theta_df.to_dict("records"),
        "predictionHistory": prediction_df.to_dict("records"),
        "rows": result_df.to_dict("records"),
    }


if __name__ == "__main__":
    result = run_sell_simulation(
        ticker="QQQ",
        start_date="2026-02-01",
        end_date="2026-05-21",
        initial_shares=100,
    )
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 80)

    print("\n=== COMPACT SUMMARY ===")
    print(json.dumps(result["summary"], indent=2, default=str))

    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(json.dumps(result["comprehensiveSummary"], indent=2, default=str))

    print("\n=== DAILY BREAKDOWN ===")
    daily_df = pd.DataFrame(result["rows"])
    print(daily_df.to_string(index=False))

    print("\n=== SELL-ONLY HISTORY ===")
    sell_history_df = pd.DataFrame(result["sellHistory"])
    if sell_history_df.empty:
        print("No sell days.")
    else:
        print(sell_history_df.to_string(index=False))

    print("\n=== THETA HISTORY ===")
    theta_history_df = pd.DataFrame(result["thetaHistory"])
    if theta_history_df.empty:
        print("No theta refreshes.")
    else:
        print(theta_history_df.to_string(index=False))

    print("\n=== PREDICTION HISTORY ===")
    prediction_history_df = pd.DataFrame(result["predictionHistory"])
    if prediction_history_df.empty:
        print("No completed prediction rows.")
    else:
        print(prediction_history_df.to_string(index=False))
