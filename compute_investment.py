from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution


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
    (0.0000, 0.0200),  # g0
    (0.0000, 0.3000),  # a
    (0.0000, 5.0000),  # b
    (0.0000, 0.1000),  # c
]


def find_dataset(ticker, data_dirs=None):
    if data_dirs is None:
        data_dirs = [Path("datasets"), Path(".")]

    ticker = ticker.upper().strip()
    filename = f"{ticker}.csv"

    for folder in data_dirs:
        path = Path(folder) / filename
        if path.exists():
            return path

    return None


def classify_regime(drawdown, config):
    if drawdown >= config["bear_drawdown"]:
        return "bear_market", True

    if drawdown >= config["entry_drawdown"]:
        return "correction", True

    return "normal", False


def simulate_window_for_theta(window_df, theta, initial_cash, config):
    g0, a, b, c = theta

    cash = initial_cash
    shares = 0.0
    reference_high = window_df["close"].iloc[0]

    for _, row in window_df.iterrows():
        price = row["close"]

        if price > reference_high:
            reference_high = price

        drawdown = (reference_high - price) / reference_high
        _, in_buy_zone = classify_regime(drawdown, config)

        if not in_buy_zone or cash <= 0:
            invest = 0.0
        else:
            daily_fraction = (
                g0
                + a * drawdown
                + b * row["predicted_return"]
                - c * row["volatility_20d"]
            )

            daily_fraction = max(0.0, daily_fraction)
            daily_fraction = min(daily_fraction, config["max_daily_fraction"])

            invest = min(
                max(cash * daily_fraction, config["min_daily_investment"]),
                cash
            )

        shares += invest / price if price > 0 else 0.0
        cash -= invest

    invested_total = initial_cash - cash
    stock_value = shares * window_df["close"].iloc[-1]

    roic = stock_value / invested_total - 1 if invested_total > 0 else 0.0
    invested_ratio = invested_total / initial_cash
    cash_penalty = (1.0 - invested_ratio) ** 2

    return -roic + config["cash_deployment_penalty"] * cash_penalty


def solve_theta_for_day(history_df, initial_cash, config):
    result = differential_evolution(
        lambda theta: simulate_window_for_theta(
            history_df,
            theta,
            initial_cash,
            config,
        ),
        bounds=BOUNDS,
        seed=config["random_state"],
        polish=False,
        maxiter=20,
        popsize=8,
    )

    return result.x


def compute_today_investment(ticker, config, data_dirs=None):
    dataset_path = find_dataset(ticker, data_dirs=data_dirs)

    if dataset_path is None:
        raise FileNotFoundError(f"No dataset found for {ticker}.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable rows.")

    target = f"future_return_{config['prediction_days']}d"
    training_df = feature_df.copy()
    training_df[target] = training_df["close"].shift(-config["prediction_days"]) / training_df["close"] - 1
    training_df = training_df.dropna(subset=[target]).copy()

    if training_df.empty:
        raise ValueError("Dataset has no usable training rows.")

    X = training_df[FEATURES]
    y = training_df[target]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=config["random_state"],
    )

    model.fit(X_train, y_train)

    feature_df["predicted_return"] = model.predict(feature_df[FEATURES])

    today_row = feature_df.iloc[-1]
    today_date = feature_df.index[-1]
    price = today_row["close"]

    drawdown = today_row["drawdown"]
    regime, in_buy_zone = classify_regime(drawdown, config)

    history_df = feature_df.iloc[-config["rolling_opt_window"]:].copy()

    if len(history_df) >= 30:
        theta_t = solve_theta_for_day(
            history_df,
            config["initial_cash"],
            config,
        )
    else:
        theta_t = np.array([0.0, 0.0, 0.0, 0.0])

    cash = config["initial_cash"]
    daily_fraction = 0.0
    daily_investment = 0.0

    if in_buy_zone and cash > 0:
        g0, a, b, c = theta_t

        daily_fraction = (
            g0
            + a * drawdown
            + b * today_row["predicted_return"]
            - c * today_row["volatility_20d"]
        )

        daily_fraction = max(0.0, daily_fraction)
        daily_fraction = min(daily_fraction, config["max_daily_fraction"])

        daily_investment = min(
            max(cash * daily_fraction, config["min_daily_investment"]),
            cash,
        )

    return {
        "date": today_date,
        "ticker": ticker,
        "price": price,
        "daily_investment": daily_investment,
        "daily_fraction": daily_fraction,
        "cash_remaining": cash - daily_investment,
        "shares": daily_investment / price if price > 0 else 0.0,
        "drawdown": drawdown,
        "reference_high": today_row["rolling_max"],
        "regime": regime,
        "in_buy_zone": in_buy_zone,
        "predicted_return": today_row["predicted_return"],
        "theta_g0": theta_t[0],
        "theta_a": theta_t[1],
        "theta_b": theta_t[2],
        "theta_c": theta_t[3],
    }
