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
    (0.0000, 0.0300),
    (0.0000, 0.0600),
    (0.0000, 0.0600),
    (0.0000, 8.0000),
    (0.0000, 0.0800),
    (0.0000, 0.0600),
    (0.0000, 0.0600),
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
    "sell_min_runup_score": 0.03,
    "sell_min_weakening_score": 0.01,
    "sell_min_overextension_score": 0.0,
    "sell_max_predicted_return": 0.001,
    "sell_strong_climb_return": 0.003,
    "sell_overextension_scale": 0.08,
    "sell_runup_scale": 0.20,
    "sell_weakening_scale": 0.10,
    "sell_downside_scale": 0.03,
    "sell_volatility_scale": 0.80,
    "sell_min_strength": 0.15,
    "sell_strength_power": 1.35,
    "sell_strength": 1.0,
    "sell_day_penalty": 0.0005,
    "sell_positive_next_day_penalty": 0.25,
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


def build_signal_sell_row(feature_df, signal_date, predicted_return):
    signal_context = feature_df.loc[feature_df.index <= signal_date].tail(80).copy()
    signal_context["predicted_return"] = 0.0
    signal_context.loc[signal_date, "predicted_return"] = predicted_return
    signal_sell_df = add_sell_signals(signal_context)

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
    ):
        signal_row[column] = 0.0
    return signal_row


def compute_sell_state(cash, shares, price):
    stock_value = shares * price
    portfolio_value = cash + stock_value
    return stock_value, portfolio_value


def _clip01(value):
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


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


def empty_daily_sell(config):
    return {
        "shares_to_sell": 0.0,
        "daily_sell_fraction": 0.0,
        "target_daily_sell_fraction": 0.0,
        "sell_allowed": False,
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
    }


def compute_daily_sell(theta, shares, config, row):
    shares = max(0.0, float(shares))
    if shares <= 0:
        return empty_daily_sell(config)

    predicted_return = float(row.get("predicted_return", 0.0))
    near_high_score = _clip01(float(row.get("near_high_score", 0.0)))
    runup_score = _clip01(float(row.get("runup_score", 0.0)))
    weakening_score = _clip01(float(row.get("momentum_weakening_score", 0.0)))
    overextension_score = _clip01(float(row.get("overextension_score", 0.0)))
    drawdown_score = _clip01(float(row.get("drawdown_score", 0.0)))
    volatility = max(0.0, float(row.get("volatility_20d", 0.0)))

    near_high_gate = near_high_score >= float(config.get("sell_min_near_high_score", 0.70))
    runup_gate = runup_score >= float(config.get("sell_min_runup_score", 0.03))
    weakening_gate = weakening_score >= float(config.get("sell_min_weakening_score", 0.01))
    overextension_gate = overextension_score >= float(config.get("sell_min_overextension_score", 0.0))
    prediction_gate = predicted_return <= float(config.get("sell_max_predicted_return", 0.001))
    strong_climb_gate = predicted_return >= float(config.get("sell_strong_climb_return", 0.003))
    sell_allowed = (
        near_high_gate
        and overextension_gate
        and prediction_gate
        and (runup_gate or weakening_gate or strong_climb_gate)
    )

    g0, near_high_w, runup_w, predicted_w, volatility_w, weakening_w, drawdown_w = theta
    raw_strength = (
        float(g0)
        + float(near_high_w) * near_high_score
        + float(runup_w) * runup_score
        + float(predicted_w) * max(0.0, -predicted_return)
        + float(volatility_w) * volatility
        + float(weakening_w) * weakening_score
        - float(drawdown_w) * drawdown_score
    )
    raw_strength = max(0.0, raw_strength)

    user_strength = max(0.0, float(config.get("sell_strength", 1.0)))
    sell_strength = raw_strength * user_strength
    adjusted_sell_strength = sell_strength ** float(config.get("sell_strength_power", 1.35))
    min_strength = float(config.get("sell_min_strength", 0.15))

    if not sell_allowed or adjusted_sell_strength < min_strength:
        daily_fraction = 0.0
    else:
        daily_fraction = min(
            adjusted_sell_strength,
            float(config.get("max_daily_sell_fraction", 0.03)),
        )

    min_shares = float(config.get("min_daily_sell_shares", 0.0))
    shares_to_sell = min(shares, shares * daily_fraction)
    if 0.0 < shares_to_sell < min_shares:
        shares_to_sell = min(shares, min_shares)
        daily_fraction = shares_to_sell / shares if shares > 0 else 0.0

    return {
        "shares_to_sell": shares_to_sell,
        "daily_sell_fraction": daily_fraction,
        "target_daily_sell_fraction": min(
            adjusted_sell_strength,
            float(config.get("max_daily_sell_fraction", 0.03)),
        ),
        "sell_allowed": sell_allowed,
        "sell_strength": sell_strength,
        "adjusted_sell_strength": adjusted_sell_strength,
        "user_sell_strength": user_strength,
        "raw_sell_strength": raw_strength,
        "near_high_gate": near_high_gate,
        "runup_gate": runup_gate,
        "weakening_gate": weakening_gate,
        "overextension_gate": overextension_gate,
        "prediction_gate": prediction_gate,
        "strong_climb_gate": strong_climb_gate,
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
    return objective + penalty


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
    config = DEFAULT_CONFIG.copy()
    config["initial_cash"] = float(total_cash)
    ticker = ticker.upper().strip()

    dataset_path = find_or_build_dataset(ticker, data_dir)
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    prediction_days = config["prediction_days"]
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

    tool_cash = float(total_cash)
    tool_shares = 0.0
    tool_rows = []
    theta = np.array([0.0, 0.0, 0.0, 0.0])
    theta_refresh_days = max(1, int(config["theta_refresh_days"]))

    prediction_rows = []
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

        # Realistic walk-forward logic:
        # today's investment uses only the previous trading day's finalized features.
        # To improve speed, the ML model is refreshed periodically, not every day.
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
            }
        )

        tool_invest = 0.0
        if tool_cash > 0:
            if day_index % theta_refresh_days == 0:
                theta_history_df = feature_df.loc[feature_df.index < signal_date].tail(
                    config["rolling_opt_window"]
                ).copy()

                if len(theta_history_df) >= 30:
                    theta_history_df["predicted_return"] = model.predict(theta_history_df[FEATURES])
                    theta = solve_theta_for_day(theta_history_df, float(total_cash), config)
                else:
                    theta = np.array([0.0, 0.0, 0.0, 0.0])

            g0, a, b, c = theta
            daily_fraction = (
                g0
                + a * float(signal_row["drawdown"])
                + b * predicted_return
                - c * float(signal_row["volatility_20d"])
            )
            daily_fraction = max(0.0, daily_fraction)
            daily_fraction = min(daily_fraction, config["max_daily_fraction"])
            tool_invest = min(
                max(tool_cash * daily_fraction, config["min_daily_investment"]),
                tool_cash,
            )

        tool_shares += tool_invest / price if price > 0 else 0.0
        tool_cash -= tool_invest
        invested_total, stock_value, portfolio_value, avg_cost, roic = compute_metrics(
            tool_cash,
            tool_shares,
            float(total_cash),
            price,
        )

        tool_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "model_signal_date": model_signal_date.strftime("%Y-%m-%d"),
                "close": price,
                "signal_close": float(signal_row["close"]),
                "predicted_return": predicted_return,
                "tool_daily_investment": tool_invest,
                "tool_cash": tool_cash,
                "tool_shares": tool_shares,
                "tool_invested_total": invested_total,
                "tool_stock_value": stock_value,
                "tool_portfolio_value": portfolio_value,
                "tool_avg_cost": avg_cost,
                "tool_return_on_invested_capital": roic,
            }
        )

    tool_df = pd.DataFrame(tool_rows)
    fair_investment_cash = float(tool_df["tool_invested_total"].iloc[-1])

    dca_cash = float(total_cash)
    dca_shares = 0.0
    dca_daily = fair_investment_cash / len(tool_df) if fair_investment_cash > 0 else 0.0
    dca_rows = []

    for _, row in tool_df.iterrows():
        price = float(row["close"])
        dca_invest = min(dca_daily, dca_cash)
        dca_shares += dca_invest / price if price > 0 else 0.0
        dca_cash -= dca_invest
        invested_total, stock_value, portfolio_value, avg_cost, roic = compute_metrics(
            dca_cash,
            dca_shares,
            float(total_cash),
            price,
        )
        dca_rows.append(
            {
                "dca_daily_investment": dca_invest,
                "dca_cash": dca_cash,
                "dca_shares": dca_shares,
                "dca_invested_total": invested_total,
                "dca_stock_value": stock_value,
                "dca_portfolio_value": portfolio_value,
                "dca_avg_cost": avg_cost,
                "dca_return_on_invested_capital": roic,
            }
        )

    result_df = pd.concat([tool_df.reset_index(drop=True), pd.DataFrame(dca_rows)], axis=1)
    prediction_df = pd.DataFrame(prediction_rows).dropna(
        subset=["actual_future_price", "predicted_future_price"]
    )

    summary = {
        "ticker": ticker,
        "startDate": result_df["date"].iloc[0],
        "endDate": result_df["date"].iloc[-1],
        "tradingDays": int(len(result_df)),
        "toolFinalValue": round(float(result_df["tool_portfolio_value"].iloc[-1]), 2),
        "dcaFinalValue": round(float(result_df["dca_portfolio_value"].iloc[-1]), 2),
        "toolInvested": round(float(result_df["tool_invested_total"].iloc[-1]), 2),
        "dcaInvested": round(float(result_df["dca_invested_total"].iloc[-1]), 2),
    }

    return {
        "summary": summary,
        "predictionChart": build_prediction_chart(prediction_df, ticker, prediction_days),
        "strategyChart": build_strategy_chart(result_df),
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
            }
        )

        signal_sell_row = build_signal_sell_row(feature_df, signal_date, predicted_return)

        if tool_shares > 0 and day_index % theta_refresh_days == 0:
            theta_history_df = feature_df.loc[feature_df.index < signal_date].tail(
                config["rolling_opt_window"]
            ).copy()

            theta_source_rows = len(theta_history_df)
            theta_status = "insufficient_history"
            if len(theta_history_df) >= 30:
                theta_history_df["predicted_return"] = model.predict(theta_history_df[FEATURES])
                theta_history_df = add_sell_signals(theta_history_df)
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
                    "theta_near_high": float(theta[1]),
                    "theta_runup": float(theta[2]),
                    "theta_predicted_return": float(theta[3]),
                    "theta_volatility": float(theta[4]),
                    "theta_momentum_weakening": float(theta[5]),
                    "theta_drawdown": float(theta[6]),
                }
            )

        sale = compute_daily_sell(theta, tool_shares, config, signal_sell_row)
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
                "near_high_score": float(signal_sell_row.get("near_high_score", 0.0)),
                "overextension_score": float(signal_sell_row.get("overextension_score", 0.0)),
                "runup_score": float(signal_sell_row.get("runup_score", 0.0)),
                "momentum_weakening_score": float(signal_sell_row.get("momentum_weakening_score", 0.0)),
                "drawdown_score": float(signal_sell_row.get("drawdown_score", 0.0)),
                "volatility_20d": float(signal_sell_row.get("volatility_20d", 0.0)),
                "theta_g0": float(theta[0]),
                "theta_near_high": float(theta[1]),
                "theta_runup": float(theta[2]),
                "theta_predicted_return": float(theta[3]),
                "theta_volatility": float(theta[4]),
                "theta_momentum_weakening": float(theta[5]),
                "theta_drawdown": float(theta[6]),
                "tool_cash_before_sale": tool_cash_before,
                "tool_shares_before_sale": tool_shares_before,
                "tool_daily_shares_sold": tool_daily_shares_sold,
                "tool_daily_sell_value": tool_daily_shares_sold * price,
                "tool_daily_sell_fraction": sale["daily_sell_fraction"],
                "tool_target_daily_sell_fraction": sale["target_daily_sell_fraction"],
                "tool_sell_allowed": bool(sale["sell_allowed"]),
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

    tool_shares_sold = initial_shares - float(result_df["tool_shares"].iloc[-1])
    result_df["tool_avg_sold_value"] = (
        result_df["tool_cash"] / (initial_shares - result_df["tool_shares"])
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    linear_shares = initial_shares
    linear_cash = 0.0
    linear_daily_shares = tool_shares_sold / len(result_df) if tool_shares_sold > 0 else 0.0
    linear_rows = []

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
        linear_shares_sold = initial_shares - linear_shares
        linear_rows.append(
            {
                "linear_daily_shares_sold": linear_sell,
                "linear_cash": linear_cash,
                "linear_shares": linear_shares,
                "linear_avg_sold_value": (
                    linear_cash / linear_shares_sold if linear_shares_sold > 0 else 0.0
                ),
                "linear_stock_value": linear_stock_value,
                "linear_portfolio_value": linear_portfolio_value,
            }
        )

    result_df = pd.concat([result_df.reset_index(drop=True), pd.DataFrame(linear_rows)], axis=1)
    prediction_df = pd.DataFrame(prediction_rows).dropna(
        subset=["actual_future_price", "predicted_future_price"]
    )
    theta_df = pd.DataFrame(theta_rows)
    sell_history_df = result_df.loc[
        result_df["tool_daily_shares_sold"] > 0,
        [
            "date",
            "signal_date",
            "close",
            "signal_close",
            "predicted_return",
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
            "theta_near_high",
            "theta_runup",
            "theta_predicted_return",
            "theta_volatility",
            "theta_momentum_weakening",
            "theta_drawdown",
        ],
    ].copy()

    sell_days = int((result_df["tool_daily_shares_sold"] > 0).sum())
    no_sell_days = int((result_df["tool_daily_shares_sold"] <= 0).sum())
    avg_sell_fraction = (
        float(sell_history_df["tool_daily_sell_fraction"].mean())
        if not sell_history_df.empty else 0.0
    )
    max_sell_fraction = (
        float(sell_history_df["tool_daily_sell_fraction"].max())
        if not sell_history_df.empty else 0.0
    )

    start_value = initial_shares * float(result_df["close"].iloc[0])
    tool_sale_total = float(result_df["tool_cash"].iloc[-1])
    linear_sale_total = float(result_df["linear_cash"].iloc[-1])

    summary = {
        "ticker": ticker,
        "startDate": result_df["date"].iloc[0],
        "endDate": result_df["date"].iloc[-1],
        "tradingDays": int(len(result_df)),
        "initialShares": round(initial_shares, 6),
        "startValue": round(start_value, 2),
        "toolFinalValue": round(tool_sale_total, 2),
        "linearSellFinalValue": round(linear_sale_total, 2),
        "toolVsLinearSell": round(tool_sale_total - linear_sale_total, 2),
        "toolCashRealized": round(tool_sale_total, 2),
        "toolSharesRemaining": round(float(result_df["tool_shares"].iloc[-1]), 6),
        "toolSharesSold": round(tool_shares_sold, 6),
        "sellDays": sell_days,
        "noSellDays": no_sell_days,
        "avgSellFractionOnSellDays": round(avg_sell_fraction, 6),
        "maxSellFraction": round(max_sell_fraction, 6),
    }

    return {
        "summary": summary,
        "predictionChart": build_prediction_chart(prediction_df, ticker, prediction_days),
        "strategyChart": build_sell_strategy_chart(result_df),
        "sellHistory": sell_history_df.to_dict("records"),
        "thetaHistory": theta_df.to_dict("records"),
        "predictionHistory": prediction_df.to_dict("records"),
        "rows": result_df.to_dict("records"),
    }
