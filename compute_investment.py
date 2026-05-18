from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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

DEFAULT_CONFIG = {
    "prediction_days": 1,
    "max_daily_fraction": 0.05,
    "min_daily_investment": 0.0,
    "cash_deployment_penalty": 0.20,
    "rolling_opt_window": 120,
    "optimizer_workers": 1,
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
    "optimizer_maxiter": 8,
    "optimizer_popsize": 5,
}


def normalize_config(config):
    merged = DEFAULT_CONFIG.copy()
    merged.update(config or {})
    merged["initial_cash"] = float(merged["initial_cash"])
    return merged


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


def compute_daily_investment(
    theta,
    cash,
    config,
    drawdown,
    predicted_return,
    volatility,
):
    g0, a, b, c = theta
    daily_fraction = g0 + a * drawdown + b * predicted_return - c * volatility
    daily_fraction = max(0.0, daily_fraction)
    daily_fraction = min(daily_fraction, config["max_daily_fraction"])
    invest = min(max(cash * daily_fraction, config["min_daily_investment"]), cash)

    return {
        "daily_investment": invest,
        "target_daily_fraction": daily_fraction,
    }


def simulate_window_for_theta(window_df, theta, initial_cash, config):
    cash = initial_cash
    shares = 0.0
    prices = window_df["close"].to_numpy(dtype=float)

    for _, row in window_df.iterrows():
        price = float(row["close"])
        if cash <= 0:
            invest = 0.0
        else:
            invest = compute_daily_investment(
                theta,
                cash,
                config,
                float(row["drawdown"]),
                float(row["predicted_return"]),
                float(row["volatility_20d"]),
            )["daily_investment"]

        shares += invest / price if price > 0 else 0.0
        cash -= invest

    stock_value = shares * prices[-1]
    portfolio_value = stock_value + cash
    portfolio_growth = portfolio_value / initial_cash - 1.0
    cash_penalty = (cash / initial_cash) ** 2

    return -portfolio_growth + config["cash_deployment_penalty"] * cash_penalty


def solve_theta_for_day(history_df, initial_cash, config, workers=None):
    if workers is None:
        workers = config["optimizer_workers"]

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

    if len(training_df) < 80:
        raise ValueError("Not enough usable training rows for this ticker/date range.")

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
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


def compute_today_investment(ticker, config, data_dirs=None):
    config = normalize_config(config)
    dataset_path = find_dataset(ticker, data_dirs=data_dirs)

    if dataset_path is None:
        raise FileNotFoundError(f"No dataset found for {ticker}.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    model = train_prediction_model(feature_df, config)
    feature_df["predicted_return"] = model.predict(feature_df[FEATURES])

    today_row = feature_df.iloc[-1]
    today_date = feature_df.index[-1]
    price = float(today_row["close"])

    history_df = feature_df.iloc[-int(config["rolling_opt_window"]):].copy()

    if len(history_df) >= 30:
        theta_t = solve_theta_for_day(
            history_df,
            config["initial_cash"],
            config,
        )
    else:
        theta_t = np.zeros(len(BOUNDS), dtype=float)

    cash = config["initial_cash"]
    investment = compute_daily_investment(
        theta_t,
        cash,
        config,
        float(today_row["drawdown"]),
        float(today_row["predicted_return"]),
        float(today_row["volatility_20d"]),
    )
    daily_investment = investment["daily_investment"]

    return {
        "date": today_date,
        "ticker": ticker.upper().strip(),
        "price": price,
        "daily_investment": daily_investment,
        "daily_fraction": daily_investment / cash if cash > 0 else 0.0,
        "cash_remaining": cash - daily_investment,
        "shares": daily_investment / price if price > 0 else 0.0,
        "drawdown": float(today_row["drawdown"]),
        "reference_high": float(today_row["rolling_max"]),
        "predicted_return": float(today_row["predicted_return"]),
        **investment,
        "theta_g0": theta_t[0],
        "theta_drawdown": theta_t[1],
        "theta_predicted_return": theta_t[2],
        "theta_volatility": theta_t[3],
    }
