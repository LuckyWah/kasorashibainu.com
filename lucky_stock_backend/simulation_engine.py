import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_builder import build_adaptive_dataset


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

DEFAULT_CONFIG = {
    "prediction_days": 1,
    "entry_drawdown": 0.10,
    "exit_drawdown": 0.03,
    "bear_drawdown": 0.20,
    "max_daily_fraction": 0.05,
    "min_daily_investment": 0.0,
    "cash_deployment_penalty": 0.20,
    "rolling_opt_window": 120,
    "theta_smoothing": 0.20,
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
}


@dataclass(frozen=True)
class ThetaObjective:
    window_df: pd.DataFrame
    initial_cash: float
    config: dict

    def __call__(self, theta):
        return simulate_window_for_theta(self.window_df, theta, self.initial_cash, self.config)


def find_or_build_dataset(ticker, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper().strip()
    dataset_path = data_dir / f"{ticker}.csv"

    if dataset_path.exists():
        return dataset_path

    return build_adaptive_dataset(ticker, output_dir=data_dir)


def classify_regime(drawdown, config):
    if drawdown >= config["bear_drawdown"]:
        return "bear_market", True

    if drawdown >= config["entry_drawdown"]:
        return "correction", True

    return "normal", False


def compute_metrics(cash, shares, initial_cash, price):
    invested_total = initial_cash - cash
    stock_value = shares * price
    portfolio_value = stock_value + cash
    avg_cost = invested_total / shares if shares > 0 else 0.0
    roic = stock_value / invested_total - 1 if invested_total > 0 else 0.0

    return invested_total, stock_value, portfolio_value, avg_cost, roic


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
            invest = min(max(cash * daily_fraction, config["min_daily_investment"]), cash)

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
        ThetaObjective(history_df, initial_cash, config),
        bounds=BOUNDS,
        seed=config["random_state"],
        polish=False,
        maxiter=20,
        popsize=8,
        updating="deferred",
        workers=-1,
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


def build_prediction_chart(plot_df, ticker, prediction_days):
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"Actual {prediction_days}-Day Future Price",
                "x": plot_df["date"].tolist(),
                "y": plot_df["actual_future_price"].round(4).tolist(),
                "line": {"color": "#00d9ff", "width": 3},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"Predicted {prediction_days}-Day Future Price",
                "x": plot_df["date"].tolist(),
                "y": plot_df["predicted_future_price"].round(4).tolist(),
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

    buttons = []
    for metric_index, (_, y_title) in enumerate(metric_options):
        visible = [False] * (len(metric_options) * 2)
        visible[metric_index * 2] = True
        visible[metric_index * 2 + 1] = True
        buttons.append(
            {
                "label": y_title.replace(" ($)", ""),
                "method": "update",
                "args": [
                    {"visible": visible, "showlegend": [i == metric_index * 2 or i == metric_index * 2 + 1 for i in range(len(visible))]},
                    {"yaxis": {"title": y_title, "gridcolor": "#2a3347"}},
                ],
            }
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
            "updatemenus": [
                {
                    "buttons": buttons,
                    "direction": "right",
                    "showactive": True,
                    "x": 0,
                    "xanchor": "left",
                    "y": 1.18,
                    "yanchor": "top",
                    "bgcolor": "#0a0e27",
                    "bordercolor": "#2a3347",
                    "font": {"color": "#ffffff"},
                }
            ],
            "margin": {"l": 56, "r": 24, "t": 84, "b": 48},
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

    model = train_prediction_model(feature_df, config)
    prediction_days = config["prediction_days"]
    feature_df["predicted_return"] = model.predict(feature_df[FEATURES])
    feature_df["predicted_future_price"] = feature_df["close"] * (1 + feature_df["predicted_return"])
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

    if len(sim_df) < 2:
        raise ValueError("Simulation period is too short.")

    tool_cash = float(total_cash)
    tool_shares = 0.0
    previous_theta = None
    in_correction_mode = False
    reference_high = sim_df["close"].iloc[0]
    tool_rows = []

    for i, (current_date, row) in enumerate(sim_df.iterrows()):
        price = float(row["close"])

        if not in_correction_mode and price > reference_high:
            reference_high = price

        drawdown = (reference_high - price) / reference_high

        if not in_correction_mode and drawdown >= config["entry_drawdown"]:
            in_correction_mode = True

        if in_correction_mode and drawdown <= config["exit_drawdown"]:
            in_correction_mode = False

        history_start = max(0, i - config["rolling_opt_window"])
        history_df = sim_df.iloc[history_start:i].copy()

        if len(history_df) >= 30:
            theta_raw = solve_theta_for_day(history_df, float(total_cash), config)
            if previous_theta is None:
                theta = theta_raw
            else:
                smoothing = config["theta_smoothing"]
                theta = (1.0 - smoothing) * previous_theta + smoothing * theta_raw
            previous_theta = theta
        elif previous_theta is not None:
            theta = previous_theta
        else:
            theta = np.array([0.0, 0.0, 0.0, 0.0])

        tool_invest = 0.0
        if in_correction_mode and tool_cash > 0:
            g0, a, b, c = theta
            daily_fraction = (
                g0
                + a * drawdown
                + b * float(row["predicted_return"])
                - c * float(row["volatility_20d"])
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
                "close": price,
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
    tool_df["tool_cash"] = (fair_investment_cash - tool_df["tool_invested_total"]).clip(lower=0.0)
    tool_df["tool_portfolio_value"] = tool_df["tool_stock_value"] + tool_df["tool_cash"]

    dca_cash = fair_investment_cash
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
            fair_investment_cash,
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
    prediction_df = sim_df.dropna(subset=["actual_future_price", "predicted_future_price"]).copy()
    prediction_df["date"] = prediction_df.index.strftime("%Y-%m-%d")

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
