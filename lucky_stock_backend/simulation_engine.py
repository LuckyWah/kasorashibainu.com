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

    # Regime-weighted adaptive DCA settings.
    # aggressiveness = 0 gives pure DCA; larger values allow stronger deviation from DCA.
    "aggressiveness": 1.0,
    "cash_aggressiveness": 0.50,
    "regime_temperature": 1.0,
    "drawdown_power": 1.5,
    "short_drawdown_window": 20,
    "min_base_fraction": 0.25,
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
    "random_state": 42,

    # Differential evolution speed settings.
    "optimizer_maxiter": 8,
    "optimizer_popsize": 5,
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
        object.__setattr__(
            self,
            "trend_strengths",
            self.window_df["regime_trend"].to_numpy(dtype=float),
        )
        object.__setattr__(
            self,
            "trend_accelerations",
            self.window_df["regime_acceleration"].to_numpy(dtype=float),
        )
        object.__setattr__(
            self,
            "short_drawdowns",
            self.window_df["short_drawdown"].to_numpy(dtype=float),
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
            self.trend_strengths,
            self.trend_accelerations,
            self.short_drawdowns,
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


def add_regime_features(feature_df, config):
    """Add regime features shared by simulation and theta optimization."""
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


def simulate_arrays_for_theta(
    theta,
    initial_cash,
    config,
    prices,
    drawdowns,
    predicted_returns,
    volatilities,
    trend_strengths,
    trend_accelerations,
    short_drawdowns,
):
    return _simulate_arrays_for_theta(
        theta,
        initial_cash,
        config["max_daily_fraction"],
        config["min_daily_investment"],
        config["cash_deployment_penalty"],
        config["aggressiveness"],
        config["cash_aggressiveness"],
        config["regime_temperature"],
        config["drawdown_power"],
        config["min_base_fraction"],
        prices,
        drawdowns,
        predicted_returns,
        volatilities,
        trend_strengths,
        trend_accelerations,
        short_drawdowns,
    )


def _softmax3(z1, z2, z3, temperature):
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


def _compute_regime_investment(
    theta,
    cash,
    initial_cash,
    price,
    day_index,
    n_days,
    max_daily_fraction,
    min_daily_investment,
    aggressiveness,
    cash_aggressiveness,
    regime_temperature,
    drawdown_power,
    min_base_fraction,
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

    # Regime scores. These are hand-designed, while opportunity intensities are optimized.
    z_bull = trend_strength + trend_acceleration + predicted_return
    z_bear = drawdown + volatility - trend_strength
    z_side = (1.0 - abs(trend_strength)) + (1.0 - drawdown)

    w_bull, w_bear, w_side = _softmax3(z_bull, z_bear, z_side, regime_temperature)

    # Regime-specific opportunity signals.
    s_bull = np.tanh(
        alpha_trend * trend_strength
        + alpha_accel * trend_acceleration
        + alpha_pred * predicted_return
    )

    s_bear = np.tanh(
        beta_drawdown * (drawdown ** drawdown_power)
        - beta_volatility * volatility
    )

    s_side = np.tanh(
        eta_short_dip * short_drawdown
        - eta_trend_penalty * trend_strength
    )

    market_signal = w_bull * s_bull + w_bear * s_bear + w_side * s_side

    multiplier = 1.0 + aggressiveness * market_signal + cash_aggressiveness * cash_signal
    multiplier = max(min_base_fraction, multiplier)

    invest = base_invest * multiplier

    # The daily cap cannot block the baseline, otherwise the strategy may fail to deploy all cash.
    max_invest = max(max_daily_fraction * cash, base_invest)
    invest = min(invest, max_invest)
    invest = min(max(invest, min_daily_investment), cash)

    return (
        invest,
        base_invest,
        multiplier,
        w_bull,
        w_bear,
        w_side,
        s_bull,
        s_bear,
        s_side,
        cash_signal,
        market_signal,
    )


def _simulate_arrays_for_theta_python(
    theta,
    initial_cash,
    max_daily_fraction,
    min_daily_investment,
    cash_deployment_penalty,
    aggressiveness,
    cash_aggressiveness,
    regime_temperature,
    drawdown_power,
    min_base_fraction,
    prices,
    drawdowns,
    predicted_returns,
    volatilities,
    trend_strengths,
    trend_accelerations,
    short_drawdowns,
):
    cash = initial_cash
    shares = 0.0
    n_days = len(prices)

    for i in range(n_days):
        price = prices[i]

        if cash <= 0:
            invest = 0.0
        else:
            invest = _compute_regime_investment(
                theta,
                cash,
                initial_cash,
                price,
                i,
                n_days,
                max_daily_fraction,
                min_daily_investment,
                aggressiveness,
                cash_aggressiveness,
                regime_temperature,
                drawdown_power,
                min_base_fraction,
                drawdowns[i],
                predicted_returns[i],
                volatilities[i],
                trend_strengths[i],
                trend_accelerations[i],
                short_drawdowns[i],
            )[0]

        shares += invest / price if price > 0 else 0.0
        cash -= invest

    stock_value = shares * prices[-1]
    portfolio_value = stock_value + cash
    portfolio_growth = portfolio_value / initial_cash - 1.0

    cash_ratio = cash / initial_cash
    cash_penalty = cash_ratio ** 2

    return -portfolio_growth + cash_deployment_penalty * cash_penalty


if njit is not None:
    _softmax3 = njit(cache=True)(_softmax3)
    _compute_regime_investment = njit(cache=True)(_compute_regime_investment)
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
        window_df["regime_trend"].to_numpy(dtype=float),
        window_df["regime_acceleration"].to_numpy(dtype=float),
        window_df["short_drawdown"].to_numpy(dtype=float),
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


def run_simulation(
    ticker,
    start_date,
    end_date,
    total_cash,
    data_dir="datasets",
    aggressiveness=None,
):
    config = DEFAULT_CONFIG.copy()
    config["initial_cash"] = float(total_cash)
    if aggressiveness is not None:
        config["aggressiveness"] = float(aggressiveness)
    ticker = ticker.upper().strip()

    dataset_path = find_or_build_dataset(ticker, data_dir)
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()
    feature_df = add_regime_features(feature_df, config)

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
    theta = np.zeros(len(BOUNDS), dtype=float)
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
                "actual_future_price": price,
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
                    theta = np.zeros(len(BOUNDS), dtype=float)

            (
                tool_invest,
                base_invest,
                investment_multiplier,
                weight_bull,
                weight_bear,
                weight_side,
                signal_bull,
                signal_bear,
                signal_side,
                cash_signal,
                market_signal,
            ) = _compute_regime_investment(
                theta,
                tool_cash,
                float(total_cash),
                price,
                day_index,
                len(sim_df),
                config["max_daily_fraction"],
                config["min_daily_investment"],
                config["aggressiveness"],
                config["cash_aggressiveness"],
                config["regime_temperature"],
                config["drawdown_power"],
                config["min_base_fraction"],
                float(signal_row["drawdown"]),
                predicted_return,
                float(signal_row["volatility_20d"]),
                float(signal_row["regime_trend"]),
                float(signal_row["regime_acceleration"]),
                float(signal_row["short_drawdown"]),
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
                "base_investment": base_invest if tool_cash > 0 else 0.0,
                "investment_multiplier": investment_multiplier if tool_cash > 0 else 0.0,
                "weight_bull": weight_bull if tool_cash > 0 else 0.0,
                "weight_bear": weight_bear if tool_cash > 0 else 0.0,
                "weight_side": weight_side if tool_cash > 0 else 0.0,
                "signal_bull": signal_bull if tool_cash > 0 else 0.0,
                "signal_bear": signal_bear if tool_cash > 0 else 0.0,
                "signal_side": signal_side if tool_cash > 0 else 0.0,
                "cash_signal": cash_signal if tool_cash > 0 else 0.0,
                "market_signal": market_signal if tool_cash > 0 else 0.0,
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
