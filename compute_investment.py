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
    (0.0000, 10.0000),  # alpha_1: bull trend coefficient
    (0.0000, 10.0000),  # alpha_2: bull acceleration coefficient
    (0.0000, 10.0000),  # alpha_3: bull predicted-return coefficient
    (0.0000, 10.0000),  # beta_1: bear drawdown coefficient
    (0.0000, 10.0000),  # beta_2: bear volatility penalty coefficient
    (0.0000, 10.0000),  # eta_1: sideways short-dip coefficient
    (0.0000, 10.0000),  # eta_2: sideways trend penalty coefficient
    (0.1000, 10.0000),  # kappa: cash-reserve sensitivity
]

DEFAULT_CONFIG = {
    "prediction_days": 1,
    "max_daily_fraction": 0.05,
    "min_daily_investment": 0.0,
    "cash_deployment_penalty": 0.20,
    "aggressiveness": 1.0,
    "cash_aggressiveness": 0.50,
    "regime_temperature": 1.0,
    "drawdown_power": 1.5,
    "short_drawdown_window": 20,
    "min_base_fraction": 0.25,
    "rolling_opt_window": 120,
    "investment_days": 120,
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


def add_regime_features(feature_df, config):
    feature_df = feature_df.copy()

    ma20 = feature_df["close"].rolling(20, min_periods=5).mean()
    ma60 = feature_df["close"].rolling(60, min_periods=20).mean()

    feature_df["regime_trend"] = (ma20 - ma60) / ma60
    feature_df["regime_acceleration"] = feature_df["regime_trend"].diff()

    short_window = int(config.get("short_drawdown_window", 20))
    short_high = feature_df["close"].rolling(short_window, min_periods=5).max()
    feature_df["short_drawdown"] = (short_high - feature_df["close"]) / short_high

    regime_cols = ["regime_trend", "regime_acceleration", "short_drawdown"]
    feature_df[regime_cols] = feature_df[regime_cols].replace([np.inf, -np.inf], np.nan)
    feature_df[regime_cols] = feature_df[regime_cols].fillna(0.0)

    return feature_df


def softmax3(z1, z2, z3, temperature):
    temperature = max(temperature, 1e-8)
    z1 = z1 / temperature
    z2 = z2 / temperature
    z3 = z3 / temperature

    zmax = max(z1, z2, z3)
    e1 = np.exp(z1 - zmax)
    e2 = np.exp(z2 - zmax)
    e3 = np.exp(z3 - zmax)
    total = e1 + e2 + e3

    return e1 / total, e2 / total, e3 / total


def compute_regime_investment(
    theta,
    cash,
    initial_cash,
    day_index,
    n_days,
    config,
    drawdown,
    predicted_return,
    volatility,
    trend_strength,
    trend_acceleration,
    short_drawdown,
):
    (
        alpha_trend,
        alpha_accel,
        alpha_pred,
        beta_drawdown,
        beta_volatility,
        eta_short_dip,
        eta_trend_penalty,
        kappa_cash,
    ) = theta

    remaining_days = max(1, n_days - day_index)
    base_invest = cash / remaining_days

    c_dca = initial_cash * remaining_days / n_days
    if c_dca <= 0:
        cash_signal = 0.0
    else:
        cash_deviation = cash / c_dca - 1.0
        cash_signal = np.tanh(kappa_cash * cash_deviation)

    z_bull = trend_strength + trend_acceleration + predicted_return
    z_bear = drawdown + volatility - trend_strength
    z_side = (1.0 - abs(trend_strength)) + (1.0 - drawdown)
    w_bull, w_bear, w_side = softmax3(
        z_bull,
        z_bear,
        z_side,
        config["regime_temperature"],
    )

    s_bull = np.tanh(
        alpha_trend * trend_strength
        + alpha_accel * trend_acceleration
        + alpha_pred * predicted_return
    )
    s_bear = np.tanh(
        beta_drawdown * (drawdown ** config["drawdown_power"])
        - beta_volatility * volatility
    )
    s_side = np.tanh(
        eta_short_dip * short_drawdown
        - eta_trend_penalty * trend_strength
    )

    market_signal = w_bull * s_bull + w_bear * s_bear + w_side * s_side
    multiplier = (
        1.0
        + config["aggressiveness"] * market_signal
        + config["cash_aggressiveness"] * cash_signal
    )
    multiplier = max(config["min_base_fraction"], multiplier)

    invest = base_invest * multiplier
    max_invest = max(config["max_daily_fraction"] * cash, base_invest)
    invest = min(invest, max_invest)
    invest = min(max(invest, config["min_daily_investment"]), cash)

    return {
        "daily_investment": invest,
        "base_investment": base_invest,
        "investment_multiplier": multiplier,
        "weight_bull": w_bull,
        "weight_bear": w_bear,
        "weight_side": w_side,
        "signal_bull": s_bull,
        "signal_bear": s_bear,
        "signal_side": s_side,
        "cash_signal": cash_signal,
        "market_signal": market_signal,
    }


def simulate_window_for_theta(window_df, theta, initial_cash, config):
    cash = initial_cash
    shares = 0.0
    prices = window_df["close"].to_numpy(dtype=float)

    for day_index, (_, row) in enumerate(window_df.iterrows()):
        price = float(row["close"])
        if cash <= 0:
            invest = 0.0
        else:
            invest = compute_regime_investment(
                theta,
                cash,
                initial_cash,
                day_index,
                len(prices),
                config,
                float(row["drawdown"]),
                float(row["predicted_return"]),
                float(row["volatility_20d"]),
                float(row["regime_trend"]),
                float(row["regime_acceleration"]),
                float(row["short_drawdown"]),
            )["daily_investment"]

        shares += invest / price if price > 0 else 0.0
        cash -= invest

    stock_value = shares * prices[-1]
    portfolio_value = stock_value + cash
    portfolio_growth = portfolio_value / initial_cash - 1.0
    cash_penalty = (cash / initial_cash) ** 2

    return -portfolio_growth + config["cash_deployment_penalty"] * cash_penalty


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
        maxiter=config.get("optimizer_maxiter", 20),
        popsize=config.get("optimizer_popsize", 8),
    )

    return result.x


def train_prediction_model(feature_df, config):
    target = f"future_return_{config['prediction_days']}d"
    training_df = feature_df.copy()
    training_df[target] = (
        training_df["close"].shift(-config["prediction_days"]) / training_df["close"] - 1
    )
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
    feature_df = add_regime_features(feature_df, config)

    if feature_df.empty:
        raise ValueError("Dataset has no usable rows.")

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
    n_days = max(1, int(config.get("investment_days", config["rolling_opt_window"])))
    investment = compute_regime_investment(
        theta_t,
        cash,
        config["initial_cash"],
        0,
        n_days,
        config,
        float(today_row["drawdown"]),
        float(today_row["predicted_return"]),
        float(today_row["volatility_20d"]),
        float(today_row["regime_trend"]),
        float(today_row["regime_acceleration"]),
        float(today_row["short_drawdown"]),
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
        "aggressiveness": config["aggressiveness"],
        **investment,
        "theta_alpha_trend": theta_t[0],
        "theta_alpha_accel": theta_t[1],
        "theta_alpha_pred": theta_t[2],
        "theta_beta_drawdown": theta_t[3],
        "theta_beta_volatility": theta_t[4],
        "theta_eta_short_dip": theta_t[5],
        "theta_eta_trend_penalty": theta_t[6],
        "theta_kappa_cash": theta_t[7],
    }
