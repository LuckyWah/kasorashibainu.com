from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from compute_investment import (
    BOUNDS,
    DEFAULT_CONFIG,
    FEATURES,
    compute_target_weight_buy,
    optimize_policy_theta,
    solve_theta_for_day,
    train_prediction_model,
)
from compute_sell import add_peak_signals, compute_sell_decision, normalize_config as normalize_sell_config
from data_builder import build_adaptive_dataset
from prediction_model import (
    UPGRADED_FEATURES,
    build_upgraded_feature_frame,
    generate_point_in_time_predictions,
    prepare_prediction_frame,
)


_UPGRADED_FEATURE_FRAME_CACHE = {}
_POINT_IN_TIME_LEDGER_CACHE = {}
_BACKTEST_CACHE_DIR = Path("runtime") / "backtest-cache"
_BACKTEST_CACHE_VERSION = "v3-shared-prediction-model"


def _dataset_cache_key(dataset_path):
    path = Path(dataset_path)
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _hashable_config_value(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_config_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_config_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_hashable_config_value(item) for item in value))
    return value


def _config_cache_key(config):
    return tuple(
        sorted((key, _hashable_config_value(value)) for key, value in config.items())
    )


def _cache_path(cache_name, cache_key):
    digest = hashlib.sha256(repr((_BACKTEST_CACHE_VERSION, cache_key)).encode("utf-8")).hexdigest()
    return _BACKTEST_CACHE_DIR / f"{cache_name}-{digest}.pkl"


def _read_cached_frame(cache_name, cache_key):
    path = _cache_path(cache_name, cache_key)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _write_cached_frame(cache_name, cache_key, frame):
    try:
        _BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(_cache_path(cache_name, cache_key))
    except Exception:
        pass


def _get_upgraded_feature_frame(feature_df, dataset_path):
    cache_key = _dataset_cache_key(dataset_path)
    cached = _UPGRADED_FEATURE_FRAME_CACHE.get(cache_key)
    if cached is None:
        cached = _read_cached_frame("upgraded-features", cache_key)
    if cached is None:
        cached = build_upgraded_feature_frame(feature_df)
        _write_cached_frame("upgraded-features", cache_key, cached)
        _UPGRADED_FEATURE_FRAME_CACHE[cache_key] = cached
    else:
        _UPGRADED_FEATURE_FRAME_CACHE[cache_key] = cached
    return cached.copy()


def _get_point_in_time_ledger(feature_df, config, dataset_path):
    max_horizon = int(config.get("downside_horizon", 60))
    model_kind = config.get("forecast_model_kind", "rf")
    retrain_every = int(config.get("ledger_retrain_every", 20))
    cache_key = (
        _dataset_cache_key(dataset_path),
        tuple(UPGRADED_FEATURES),
        _config_cache_key(config),
        max_horizon,
        model_kind,
        retrain_every,
        len(feature_df),
        feature_df.index[0] if not feature_df.empty else None,
        feature_df.index[-1] if not feature_df.empty else None,
    )
    cached = _POINT_IN_TIME_LEDGER_CACHE.get(cache_key)
    if cached is None:
        cached = _read_cached_frame("forecast-ledger", cache_key)
    if cached is None:
        cached = generate_point_in_time_predictions(
            feature_df,
            UPGRADED_FEATURES,
            config=config,
            max_horizon=max_horizon,
            model_kind=model_kind,
            retrain_every=retrain_every,
        )
        _write_cached_frame("forecast-ledger", cache_key, cached)
        _POINT_IN_TIME_LEDGER_CACHE[cache_key] = cached
    else:
        _POINT_IN_TIME_LEDGER_CACHE[cache_key] = cached
    return cached.copy()


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


def train_prediction_model_until(feature_df, current_date, config):
    historical_df = feature_df.loc[feature_df.index < current_date].copy()
    return train_prediction_model(historical_df, config)


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
    series = [
        ("lucky_portfolio_value", "Lucky Sell", "#00d9ff"),
        ("hold_portfolio_value", "Hold", "#ffd700"),
        ("linear_portfolio_value", "Matched Linear Sell", "#ff8a65"),
    ]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "x": result_df["date"].tolist(),
                "y": result_df[column].round(4).tolist(),
                "line": {"color": color, "width": 3},
            }
            for column, name, color in series
        ],
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


def safe_round(value, digits=4):
    if value is None or not np.isfinite(float(value)):
        return None
    return round(float(value), digits)


def max_drawdown(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = np.where(running_max > 0, values / running_max - 1.0, 0.0)
    return float(drawdowns.min())


def series_stats(series, digits=6):
    clean = pd.Series(series).dropna()
    if clean.empty:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": safe_round(clean.min(), digits),
        "max": safe_round(clean.max(), digits),
        "mean": safe_round(clean.mean(), digits),
        "median": safe_round(clean.median(), digits),
    }


EDGE_SIGNED_COMPONENT_COLUMNS = [
    "edge_upside_signed",
    "edge_downside_penalty_signed",
    "edge_disagreement_penalty_signed",
    "edge_correction_bonus_signed",
    "edge_rally_bonus_signed",
    "edge_riskoff_penalty_signed",
    "edge_volatility_penalty_signed",
]


def edge_component_summary(df):
    if df.empty:
        return {
            "rows": 0,
            "approvedRate": None,
            "calibratedEdge": {},
            "meanCalibratedEdge": None,
            "meanSumSignedComponents": None,
            "meanReconciliationError": None,
            "signedComponents": {},
            "largestNegativeSignedComponent": None,
        }

    signed_means = {
        column: safe_round(df[column].mean(), 6)
        for column in EDGE_SIGNED_COMPONENT_COLUMNS
        if column in df
    }
    largest_negative = None
    negative_components = {
        column: value for column, value in signed_means.items()
        if value is not None and value < 0.0
    }
    if negative_components:
        largest_negative = min(negative_components, key=negative_components.get)

    calibrated_edge_mean = safe_round(df["calibrated_edge"].mean(), 6) if "calibrated_edge" in df else None
    signed_sum_mean = safe_round(df["edge_sum_signed_components"].mean(), 6) if "edge_sum_signed_components" in df else None
    reconciliation_mean = (
        safe_round(df["edge_reconciliation_error"].mean(), 10)
        if "edge_reconciliation_error" in df
        else None
    )

    return {
        "rows": int(len(df)),
        "approvedRate": safe_round(df["buy_approved"].mean(), 6) if "buy_approved" in df else None,
        "calibratedEdge": series_stats(df["calibrated_edge"]) if "calibrated_edge" in df else {},
        "meanCalibratedEdge": calibrated_edge_mean,
        "meanSumSignedComponents": signed_sum_mean,
        "meanReconciliationError": reconciliation_mean,
        "signedComponents": signed_means,
        "largestNegativeSignedComponent": largest_negative,
    }


def forecast_evaluation_summary(prediction_df):
    if prediction_df.empty:
        return {}

    evaluation = {}
    horizon_specs = [
        ("20d", "predicted_return_20d", "actual_future_price_20d", "raw_base_return_20d", "kalman_adjusted_return_20d"),
        ("60d", "predicted_return_60d", "actual_future_price_60d", "raw_base_return_60d", "kalman_adjusted_return_60d"),
    ]
    for label, predicted_col, actual_price_col, raw_col, kalman_col in horizon_specs:
        if predicted_col not in prediction_df or actual_price_col not in prediction_df:
            continue
        frame = prediction_df.dropna(subset=[predicted_col, actual_price_col, "predicted_future_price"]).copy()
        if frame.empty:
            continue
        signal_price = frame["predicted_future_price"] / (1.0 + frame["predicted_return"])
        actual_return = frame[actual_price_col] / signal_price - 1.0
        predicted_return = frame[predicted_col]
        error = predicted_return - actual_return
        direction_hit = np.sign(predicted_return) == np.sign(actual_return)
        item = {
            "rows": int(len(frame)),
            "predictedReturn": series_stats(predicted_return),
            "actualReturn": series_stats(actual_return),
            "meanError": safe_round(error.mean(), 6),
            "meanAbsoluteError": safe_round(error.abs().mean(), 6),
            "directionHitRate": safe_round(direction_hit.mean(), 6),
        }
        if raw_col in frame and kalman_col in frame:
            raw = frame[raw_col]
            kalman = frame[kalman_col]
            raw_error = raw - actual_return
            kalman_error = kalman - actual_return
            item["rawBase"] = {
                "predictedReturn": series_stats(raw),
                "meanError": safe_round(raw_error.mean(), 6),
                "meanAbsoluteError": safe_round(raw_error.abs().mean(), 6),
                "directionHitRate": safe_round((np.sign(raw) == np.sign(actual_return)).mean(), 6),
            }
            item["kalmanAdjusted"] = {
                "predictedReturn": series_stats(kalman),
                "meanError": safe_round(kalman_error.mean(), 6),
                "meanAbsoluteError": safe_round(kalman_error.abs().mean(), 6),
                "directionHitRate": safe_round((np.sign(kalman) == np.sign(actual_return)).mean(), 6),
                "meanBias": safe_round(frame.get(f"kalman_bias_{label}", pd.Series(dtype=float)).mean(), 6),
                "meanGain": safe_round(frame.get(f"kalman_gain_{label}", pd.Series(dtype=float)).mean(), 6),
                "maxUpdateCount": safe_round(frame.get(f"kalman_update_count_{label}", pd.Series(dtype=float)).max(), 0),
            }
            item["kalmanMaeImprovement"] = safe_round(
                raw_error.abs().mean() - kalman_error.abs().mean(),
                6,
            )
        evaluation[label] = item
    return evaluation


def no_buy_decision(cash, shares, price, max_daily_buy=0.0, policy_status="hold"):
    stock_value = max(0.0, float(shares)) * max(0.0, float(price))
    portfolio_value = max(0.0, float(cash)) + stock_value
    current_weight = stock_value / portfolio_value if portfolio_value > 0 else 0.0
    return {
        "daily_investment": 0.0,
        "target_daily_fraction": 0.0,
        "policy_target_weight": 0.0,
        "target_weight": 0.0,
        "risk_ceiling": 0.0,
        "risk_ceiling_score": 0.0,
        "forecast_disagreement": 0.0,
        "staleness_scale": 1.0,
        "desired_stock_value": stock_value,
        "current_stock_weight": current_weight,
        "current_stock_value": stock_value,
        "portfolio_value": portfolio_value,
        "raw_buy_needed": 0.0,
        "exposure_cap_weight": 0.0,
        "exposure_target_value": stock_value,
        "exposure_headroom": 0.0,
        "portfolio_daily_cap": 0.0,
        "deployment_budget": 0.0,
        "elapsed_trading_days": 0,
        "nominal_deployment_days": 0,
        "max_extension_days": 0,
        "model_confidence": 0.0,
        "raw_model_multiplier": 1.0,
        "risk_acceleration_cap": 1.0,
        "signal_model_multiplier": 1.0,
        "effective_model_multiplier": 1.0,
        "final_multiplier": 1.0,
        "invested_total_before_buy": max(0.0, portfolio_value - cash),
        "ramp_days": 0.0,
        "model_buy": 0.0,
        "base_buy": 0.0,
        "calibrated_edge": 0.0,
        "entry_threshold": 0.0,
        "strong_edge": 0.0,
        "buy_approved": False,
        "buy_strength": 0.0,
        "buy_intensity_gamma": 0.0,
        "edge_upside_component": 0.0,
        "edge_downside_component": 0.0,
        "edge_disagreement_component": 0.0,
        "edge_correction_timing_component": 0.0,
        "edge_rally_momentum_component": 0.0,
        "edge_riskoff_component": 0.0,
        "edge_volatility_component": 0.0,
        "edge_upside_signed": 0.0,
        "edge_downside_penalty_signed": 0.0,
        "edge_disagreement_penalty_signed": 0.0,
        "edge_correction_bonus_signed": 0.0,
        "edge_rally_bonus_signed": 0.0,
        "edge_riskoff_penalty_signed": 0.0,
        "edge_volatility_penalty_signed": 0.0,
        "edge_sum_signed_components": 0.0,
        "edge_reconciliation_error": 0.0,
        "soft_target_progress": 0.0,
        "actual_progress": 0.0,
        "pace_gap": 0.0,
        "time_scale": 1.0,
        "max_daily_buy": float(max_daily_buy),
        "buy_limited_by_cap": False,
        "policy_status": policy_status,
        "contribution_intercept": 0.0,
        "contribution_mu20": 0.0,
        "contribution_mu60": 0.0,
        "contribution_downside": 0.0,
        "contribution_drawdown": 0.0,
        "contribution_drawdown_correction": 0.0,
        "contribution_drawdown_riskoff": 0.0,
        "contribution_momentum_rally": 0.0,
        "raw_score": 0.0,
    }


def _delete_backtest_dataset(path):
    path = Path(path)
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass


def run_buy_simulation(
    ticker,
    start_date,
    end_date,
    total_cash,
    data_dir="datasets",
    config_overrides=None,
    cleanup_data=False,
):
    data_dir_path = Path(data_dir)
    ticker = ticker.upper().strip()
    expected_dataset_path = data_dir_path / f"{ticker}.csv"
    dataset_existed_before = expected_dataset_path.exists()
    dataset_path = None

    try:
        dataset_path = find_or_build_dataset(ticker, data_dir_path)
        return _run_buy_simulation_core(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_cash=total_cash,
            data_dir=data_dir_path,
            dataset_path=dataset_path,
            config_overrides=config_overrides,
        )
    finally:
        if cleanup_data and dataset_path is not None and not dataset_existed_before:
            _delete_backtest_dataset(dataset_path)


def _run_buy_simulation_core(
    ticker,
    start_date,
    end_date,
    total_cash,
    data_dir="datasets",
    dataset_path=None,
    config_overrides=None,
):
    user_config_overrides = config_overrides or {}
    config = DEFAULT_CONFIG.copy()
    config.update(user_config_overrides)
    config.setdefault("model_refresh_days", 20)
    config.setdefault("theta_refresh_days", 20)
    config["initial_cash"] = float(total_cash)
    if config.get("deployment_budget") is None:
        config["deployment_budget"] = float(total_cash)
    ticker = ticker.upper().strip()

    if dataset_path is None:
        dataset_path = find_or_build_dataset(ticker, data_dir)
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    upgraded_policy = config.get("buy_policy") == "target_weight"
    if upgraded_policy:
        feature_df = _get_upgraded_feature_frame(feature_df, dataset_path)
        ledger_df = _get_point_in_time_ledger(feature_df, config, dataset_path)
        if ledger_df.empty:
            raise ValueError("Not enough history to build an out-of-sample forecast ledger.")
        ledger_df = ledger_df.set_index("date")
        ledger_columns = [
            "predicted_return_20d",
            "predicted_return_60d",
            "predicted_downside_60d",
            "raw_base_return_20d",
            "raw_base_return_60d",
            "kalman_adjusted_return_20d",
            "kalman_adjusted_return_60d",
            "kalman_bias_20d",
            "kalman_bias_60d",
            "kalman_gain_20d",
            "kalman_gain_60d",
            "kalman_update_count_20d",
            "kalman_update_count_60d",
            "model_trained_through",
            "maturity_date",
            "maturity_date_20d",
            "maturity_date_60d",
            "is_mature",
        ]
        feature_df = feature_df.join(ledger_df[ledger_columns], how="left")
        feature_df = feature_df.dropna(
            subset=["predicted_return_20d", "predicted_return_60d", "predicted_downside_60d"]
        )

    prediction_days = config["prediction_days"]
    feature_df["actual_future_price"] = feature_df["close"].shift(-prediction_days)
    feature_df["actual_future_price_20d"] = feature_df["close"].shift(-20)
    feature_df["actual_future_price_60d"] = feature_df["close"].shift(-60)

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

    if upgraded_policy:
        if "deployment_budget" not in user_config_overrides or config.get("deployment_budget") is None:
            config["deployment_budget"] = float(total_cash)
        if "max_extension_days" not in user_config_overrides:
            config["max_extension_days"] = max(0, min(int(config.get("max_extension_days", 25)), len(sim_df) - 1))
        if "nominal_deployment_days" not in user_config_overrides:
            config["nominal_deployment_days"] = max(1, len(sim_df) - int(config.get("max_extension_days", 0)))
        if "soft_horizon_days" not in user_config_overrides:
            config["soft_horizon_days"] = int(config["nominal_deployment_days"])

    tool_cash = float(total_cash)
    tool_shares = 0.0
    tool_rows = []
    prediction_rows = []
    theta_rows = []
    theta = np.zeros(8 if upgraded_policy else len(BOUNDS), dtype=float)
    theta_is_optimized = False
    theta_refresh_days = max(1, int(config["theta_refresh_days"]))
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

        if not upgraded_policy and (model is None or day_index % model_refresh_days == 0):
            model = train_prediction_model_until(feature_df, signal_date, config)
            model_signal_date = signal_date

        if upgraded_policy:
            model_signal_date = signal_row.get("model_trained_through", signal_date)
            predicted_return = float(signal_row["predicted_return_60d"])
            actual_future_price_for_chart = signal_row["actual_future_price_60d"]
        else:
            predicted_return = float(model.predict(signal_row[FEATURES].to_frame().T)[0])
            actual_future_price_for_chart = signal_row["actual_future_price"]
        predicted_future_price = float(signal_row["close"]) * (1.0 + predicted_return)
        forecast_age_days = (
            int((signal_date - pd.Timestamp(model_signal_date)).days)
            if upgraded_policy and pd.notna(model_signal_date)
            else 0
        )

        prediction_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "model_signal_date": model_signal_date.strftime("%Y-%m-%d"),
                "predicted_future_price": predicted_future_price,
                "actual_future_price": actual_future_price_for_chart,
                "actual_future_price_20d": signal_row["actual_future_price_20d"],
                "actual_future_price_60d": signal_row["actual_future_price_60d"],
                "predicted_return": predicted_return,
                "predicted_return_20d": float(signal_row.get("predicted_return_20d", predicted_return)),
                "predicted_return_60d": float(signal_row.get("predicted_return_60d", predicted_return)),
                "predicted_downside_60d": float(signal_row.get("predicted_downside_60d", 0.0)),
                "raw_base_return_20d": float(signal_row.get("raw_base_return_20d", predicted_return)),
                "raw_base_return_60d": float(signal_row.get("raw_base_return_60d", predicted_return)),
                "kalman_adjusted_return_20d": float(signal_row.get("kalman_adjusted_return_20d", predicted_return)),
                "kalman_adjusted_return_60d": float(signal_row.get("kalman_adjusted_return_60d", predicted_return)),
                "kalman_bias_20d": float(signal_row.get("kalman_bias_20d", 0.0)),
                "kalman_bias_60d": float(signal_row.get("kalman_bias_60d", 0.0)),
                "kalman_gain_20d": float(signal_row.get("kalman_gain_20d", 0.0)),
                "kalman_gain_60d": float(signal_row.get("kalman_gain_60d", 0.0)),
                "kalman_update_count_20d": int(signal_row.get("kalman_update_count_20d", 0)),
                "kalman_update_count_60d": int(signal_row.get("kalman_update_count_60d", 0)),
            }
        )

        tool_invest = 0.0
        theta_status = "cash_depleted" if tool_cash <= 0 else "not_refreshed"
        theta_source_rows = 0
        theta_optimizer = {
            "optimizer_fun": None,
            "optimizer_nfev": 0,
            "optimizer_success": False,
            "optimizer_message": "",
        }

        if tool_cash > 0:
            if day_index % theta_refresh_days == 0:
                theta_history_df = feature_df.loc[feature_df.index < signal_date].copy()
                if upgraded_policy:
                    maturity_dates = pd.to_datetime(theta_history_df["maturity_date"])
                    theta_history_df = theta_history_df.loc[
                        theta_history_df["is_mature"].astype(bool)
                        & maturity_dates.notna()
                        & (maturity_dates <= signal_date)
                    ].tail(config["rolling_opt_window"])
                    assert (
                        pd.to_datetime(theta_history_df["maturity_date"]) <= signal_date
                    ).all()
                else:
                    theta_history_df = theta_history_df.tail(config["rolling_opt_window"])
                theta_source_rows = len(theta_history_df)

                if len(theta_history_df) >= 30:
                    if upgraded_policy:
                        theta, theta_optimizer = optimize_policy_theta(
                            theta_history_df,
                            float(total_cash),
                            config,
                            return_diagnostics=True,
                        )
                        theta_is_optimized = True
                        theta_status = (
                            "optimized"
                            if theta_optimizer["optimizer_success"]
                            else "partial_optimization"
                        )
                    else:
                        theta_history_df["predicted_return"] = model.predict(theta_history_df[FEATURES])
                        theta = solve_theta_for_day(theta_history_df, float(total_cash), config)
                        theta_status = "optimized"
                else:
                    theta = np.zeros(8 if upgraded_policy else len(BOUNDS), dtype=float)
                    theta_is_optimized = False
                    theta_status = "insufficient_history"
                    theta_optimizer["optimizer_message"] = "insufficient_history"

                theta_rows.append(
                    {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "signal_date": signal_date.strftime("%Y-%m-%d"),
                        "policy": "target_weight" if upgraded_policy else "baseline_theta",
                        "status": theta_status,
                        "history_rows": int(theta_source_rows),
                        "training_start": (
                            theta_history_df.index[0].strftime("%Y-%m-%d")
                            if upgraded_policy and not theta_history_df.empty
                            else None
                        ),
                        "training_end": (
                            theta_history_df.index[-1].strftime("%Y-%m-%d")
                            if upgraded_policy and not theta_history_df.empty
                            else None
                        ),
                        "mature_label_count": int(theta_source_rows) if upgraded_policy else 0,
                        "optimizer_fun": theta_optimizer["optimizer_fun"],
                        "optimizer_nfev": int(theta_optimizer["optimizer_nfev"]),
                        "optimizer_success": bool(theta_optimizer["optimizer_success"]),
                        "optimizer_message": theta_optimizer["optimizer_message"],
                        "theta_g0": float(theta[0]),
                        "theta_drawdown": float(theta[4]) if upgraded_policy else float(theta[1]),
                        "theta_predicted_return": float(theta[2]),
                        "theta_volatility": 0.0 if upgraded_policy else float(theta[3]),
                        "theta_mu20": float(theta[1]) if upgraded_policy else 0.0,
                        "theta_mu60": float(theta[2]) if upgraded_policy else 0.0,
                        "theta_downside": float(theta[3]) if upgraded_policy else 0.0,
                        "theta_drawdown_correction": float(theta[5]) if upgraded_policy else 0.0,
                        "theta_drawdown_riskoff": float(theta[6]) if upgraded_policy else 0.0,
                        "theta_momentum_rally": float(theta[7]) if upgraded_policy else 0.0,
                    }
                )

            if upgraded_policy:
                if not theta_is_optimized:
                    buy = no_buy_decision(
                        tool_cash,
                        tool_shares,
                        price,
                        policy_status="warming_up",
                    )
                else:
                    effective_config = config.copy()
                    allocation_signals = signal_row.copy()
                    allocation_signals["forecast_age_days"] = forecast_age_days
                    allocation_signals["elapsed_trading_days"] = day_index + 1
                    allocation_signals["deployment_budget"] = float(config.get("deployment_budget", total_cash))
                    buy = compute_target_weight_buy(
                        theta,
                        tool_cash,
                        tool_shares,
                        price,
                        effective_config,
                        allocation_signals,
                    )
                    buy["policy_status"] = "optimized"
                daily_fraction = buy["target_daily_fraction"]
                tool_invest = buy["daily_investment"]
            else:
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
        else:
            daily_fraction = 0.0
            buy = no_buy_decision(tool_cash, tool_shares, price, policy_status="cash_depleted")

        if not upgraded_policy:
            buy = no_buy_decision(
                tool_cash,
                tool_shares,
                price,
                max_daily_buy=float(total_cash) * float(config["max_daily_fraction"]),
                policy_status="baseline_theta",
            )

        tool_cash_before = tool_cash
        tool_shares_before = tool_shares
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
                "predicted_return_20d": float(signal_row.get("predicted_return_20d", predicted_return)),
                "predicted_return_60d": float(signal_row.get("predicted_return_60d", predicted_return)),
                "predicted_downside_60d": float(signal_row.get("predicted_downside_60d", 0.0)),
                "raw_base_return_20d": float(signal_row.get("raw_base_return_20d", predicted_return)),
                "raw_base_return_60d": float(signal_row.get("raw_base_return_60d", predicted_return)),
                "kalman_adjusted_return_20d": float(signal_row.get("kalman_adjusted_return_20d", predicted_return)),
                "kalman_adjusted_return_60d": float(signal_row.get("kalman_adjusted_return_60d", predicted_return)),
                "kalman_bias_20d": float(signal_row.get("kalman_bias_20d", 0.0)),
                "kalman_bias_60d": float(signal_row.get("kalman_bias_60d", 0.0)),
                "kalman_gain_20d": float(signal_row.get("kalman_gain_20d", 0.0)),
                "kalman_gain_60d": float(signal_row.get("kalman_gain_60d", 0.0)),
                "kalman_update_count_20d": int(signal_row.get("kalman_update_count_20d", 0)),
                "kalman_update_count_60d": int(signal_row.get("kalman_update_count_60d", 0)),
                "p_rally": float(signal_row.get("p_rally", 0.0)),
                "p_correction": float(signal_row.get("p_correction", 0.0)),
                "p_riskoff": float(signal_row.get("p_riskoff", 0.0)),
                "regime_probability_sum": (
                    float(signal_row.get("p_rally", 0.0))
                    + float(signal_row.get("p_correction", 0.0))
                    + float(signal_row.get("p_riskoff", 0.0))
                ),
                "drawdown": float(signal_row["drawdown"]),
                "volatility_20d": float(signal_row["volatility_20d"]),
                "current_stock_weight": float(buy["current_stock_weight"]),
                "target_stock_weight": float(buy["target_weight"]),
                "policy_target_weight": float(buy.get("policy_target_weight", buy["target_weight"])),
                "risk_ceiling": float(buy.get("risk_ceiling", 0.0)),
                "risk_ceiling_score": float(buy.get("risk_ceiling_score", 0.0)),
                "forecast_disagreement": float(buy.get("forecast_disagreement", 0.0)),
                "desired_stock_value": float(buy["desired_stock_value"]),
                "exposure_cap_weight": float(buy.get("exposure_cap_weight", buy["target_weight"])),
                "exposure_target_value": float(buy.get("exposure_target_value", buy["desired_stock_value"])),
                "exposure_headroom": float(buy.get("exposure_headroom", 0.0)),
                "current_stock_value_before_buy": float(buy["current_stock_value"]),
                "portfolio_value_before_buy": float(buy["portfolio_value"]),
                "raw_buy_needed": float(buy["raw_buy_needed"]),
                "portfolio_daily_cap": float(buy.get("portfolio_daily_cap", 0.0)),
                "deployment_budget": float(buy.get("deployment_budget", 0.0)),
                "elapsed_trading_days": int(buy.get("elapsed_trading_days", 0)),
                "nominal_deployment_days": int(buy.get("nominal_deployment_days", 0)),
                "max_extension_days": int(buy.get("max_extension_days", 0)),
                "model_confidence": float(buy.get("model_confidence", 0.0)),
                "raw_model_multiplier": float(buy.get("raw_model_multiplier", 1.0)),
                "risk_acceleration_cap": float(buy.get("risk_acceleration_cap", 1.0)),
                "signal_model_multiplier": float(buy.get("signal_model_multiplier", 1.0)),
                "effective_model_multiplier": float(buy.get("effective_model_multiplier", 1.0)),
                "final_multiplier": float(buy.get("final_multiplier", 1.0)),
                "invested_total_before_buy": float(buy.get("invested_total_before_buy", 0.0)),
                "ramp_days": float(buy.get("ramp_days", 0.0)),
                "model_buy": float(buy.get("model_buy", 0.0)),
                "base_buy": float(buy.get("base_buy", 0.0)),
                "calibrated_edge": float(buy.get("calibrated_edge", 0.0)),
                "entry_threshold": float(buy.get("entry_threshold", 0.0)),
                "strong_edge": float(buy.get("strong_edge", 0.0)),
                "buy_approved": bool(buy.get("buy_approved", False)),
                "buy_strength": float(buy.get("buy_strength", 0.0)),
                "buy_intensity_gamma": float(buy.get("buy_intensity_gamma", 0.0)),
                "edge_upside_component": float(buy.get("edge_upside_component", 0.0)),
                "edge_downside_component": float(buy.get("edge_downside_component", 0.0)),
                "edge_disagreement_component": float(buy.get("edge_disagreement_component", 0.0)),
                "edge_correction_timing_component": float(
                    buy.get("edge_correction_timing_component", 0.0)
                ),
                "edge_rally_momentum_component": float(
                    buy.get("edge_rally_momentum_component", 0.0)
                ),
                "edge_riskoff_component": float(buy.get("edge_riskoff_component", 0.0)),
                "edge_volatility_component": float(buy.get("edge_volatility_component", 0.0)),
                "edge_upside_signed": float(buy.get("edge_upside_signed", 0.0)),
                "edge_downside_penalty_signed": float(
                    buy.get("edge_downside_penalty_signed", 0.0)
                ),
                "edge_disagreement_penalty_signed": float(
                    buy.get("edge_disagreement_penalty_signed", 0.0)
                ),
                "edge_correction_bonus_signed": float(
                    buy.get("edge_correction_bonus_signed", 0.0)
                ),
                "edge_rally_bonus_signed": float(buy.get("edge_rally_bonus_signed", 0.0)),
                "edge_riskoff_penalty_signed": float(
                    buy.get("edge_riskoff_penalty_signed", 0.0)
                ),
                "edge_volatility_penalty_signed": float(
                    buy.get("edge_volatility_penalty_signed", 0.0)
                ),
                "edge_sum_signed_components": float(
                    buy.get("edge_sum_signed_components", 0.0)
                ),
                "edge_reconciliation_error": float(
                    buy.get("edge_reconciliation_error", 0.0)
                ),
                "soft_target_progress": float(buy.get("soft_target_progress", 0.0)),
                "actual_progress": float(buy.get("actual_progress", 0.0)),
                "pace_gap": float(buy.get("pace_gap", 0.0)),
                "time_scale": float(buy.get("time_scale", 1.0)),
                "max_daily_buy": float(buy["max_daily_buy"]),
                "buy_limited_by_cap": bool(buy["buy_limited_by_cap"]),
                "policy_status": buy.get("policy_status", theta_status),
                "forecast_age_days": forecast_age_days,
                "staleness_scale": float(buy.get("staleness_scale", 1.0)),
                "maturity_date": (
                    pd.Timestamp(signal_row["maturity_date"]).strftime("%Y-%m-%d")
                    if upgraded_policy and pd.notna(signal_row.get("maturity_date"))
                    else None
                ),
                "contribution_intercept": float(buy.get("contribution_intercept", 0.0)),
                "contribution_mu20": float(buy.get("contribution_mu20", 0.0)),
                "contribution_mu60": float(buy.get("contribution_mu60", 0.0)),
                "contribution_downside": float(buy.get("contribution_downside", 0.0)),
                "contribution_drawdown": float(buy.get("contribution_drawdown", 0.0)),
                "contribution_drawdown_correction": float(
                    buy.get("contribution_drawdown_correction", 0.0)
                ),
                "contribution_drawdown_riskoff": float(
                    buy.get("contribution_drawdown_riskoff", 0.0)
                ),
                "contribution_momentum_rally": float(
                    buy.get("contribution_momentum_rally", 0.0)
                ),
                "raw_score": float(buy.get("raw_score", 0.0)),
                "theta_g0": float(theta[0]),
                "theta_drawdown": float(theta[4]) if upgraded_policy else float(theta[1]),
                "theta_predicted_return": float(theta[2]),
                "theta_volatility": 0.0 if upgraded_policy else float(theta[3]),
                "theta_mu20": float(theta[1]) if upgraded_policy else 0.0,
                "theta_mu60": float(theta[2]) if upgraded_policy else 0.0,
                "theta_downside": float(theta[3]) if upgraded_policy else 0.0,
                "theta_drawdown_correction": float(theta[5]) if upgraded_policy else 0.0,
                "theta_drawdown_riskoff": float(theta[6]) if upgraded_policy else 0.0,
                "theta_momentum_rally": float(theta[7]) if upgraded_policy else 0.0,
                "tool_cash_before_investment": tool_cash_before,
                "tool_shares_before_investment": tool_shares_before,
                "tool_daily_fraction": daily_fraction,
                "tool_daily_fraction_denominator": "cash_before_investment",
                "tool_portfolio_daily_fraction": (
                    tool_invest / float(buy.get("portfolio_value", 0.0))
                    if float(buy.get("portfolio_value", 0.0)) > 0
                    else 0.0
                ),
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
    if tool_df.empty:
        raise ValueError("No usable simulation rows were produced.")

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
    theta_df = pd.DataFrame(theta_rows)
    investment_history_df = result_df.loc[
        result_df["tool_daily_investment"] > 0,
        [
            "date",
            "signal_date",
            "close",
            "signal_close",
            "predicted_return",
            "predicted_return_20d",
            "predicted_return_60d",
            "predicted_downside_60d",
            "raw_base_return_20d",
            "raw_base_return_60d",
            "kalman_adjusted_return_20d",
            "kalman_adjusted_return_60d",
            "kalman_bias_20d",
            "kalman_bias_60d",
            "kalman_gain_20d",
            "kalman_gain_60d",
            "kalman_update_count_20d",
            "kalman_update_count_60d",
            "p_rally",
            "p_correction",
            "p_riskoff",
            "regime_probability_sum",
            "drawdown",
            "volatility_20d",
            "current_stock_weight",
            "target_stock_weight",
            "policy_target_weight",
            "risk_ceiling",
            "risk_ceiling_score",
            "forecast_disagreement",
            "desired_stock_value",
            "exposure_cap_weight",
            "exposure_target_value",
            "exposure_headroom",
            "current_stock_value_before_buy",
            "portfolio_value_before_buy",
            "raw_buy_needed",
            "portfolio_daily_cap",
            "deployment_budget",
            "elapsed_trading_days",
            "nominal_deployment_days",
            "max_extension_days",
            "model_confidence",
            "raw_model_multiplier",
            "risk_acceleration_cap",
            "signal_model_multiplier",
            "effective_model_multiplier",
            "final_multiplier",
            "invested_total_before_buy",
            "ramp_days",
            "model_buy",
            "base_buy",
            "calibrated_edge",
            "entry_threshold",
            "strong_edge",
            "buy_approved",
            "buy_strength",
            "buy_intensity_gamma",
            "edge_upside_component",
            "edge_downside_component",
            "edge_disagreement_component",
            "edge_correction_timing_component",
            "edge_rally_momentum_component",
            "edge_riskoff_component",
            "edge_volatility_component",
            "edge_upside_signed",
            "edge_downside_penalty_signed",
            "edge_disagreement_penalty_signed",
            "edge_correction_bonus_signed",
            "edge_rally_bonus_signed",
            "edge_riskoff_penalty_signed",
            "edge_volatility_penalty_signed",
            "edge_sum_signed_components",
            "edge_reconciliation_error",
            "soft_target_progress",
            "actual_progress",
            "pace_gap",
            "time_scale",
            "max_daily_buy",
            "buy_limited_by_cap",
            "policy_status",
            "forecast_age_days",
            "staleness_scale",
            "maturity_date",
            "contribution_intercept",
            "contribution_mu20",
            "contribution_mu60",
            "contribution_downside",
            "contribution_drawdown",
            "contribution_drawdown_correction",
            "contribution_drawdown_riskoff",
            "contribution_momentum_rally",
            "raw_score",
            "tool_cash_before_investment",
            "tool_daily_fraction",
            "tool_daily_fraction_denominator",
            "tool_portfolio_daily_fraction",
            "tool_daily_investment",
            "tool_cash",
            "tool_shares",
            "tool_invested_total",
            "tool_stock_value",
            "tool_portfolio_value",
            "tool_avg_cost",
            "tool_return_on_invested_capital",
            "theta_g0",
            "theta_drawdown",
            "theta_predicted_return",
            "theta_volatility",
            "theta_mu20",
            "theta_mu60",
            "theta_downside",
            "theta_drawdown_correction",
            "theta_drawdown_riskoff",
            "theta_momentum_rally",
        ],
    ].copy()

    summary = {
        "ticker": ticker,
        "startDate": result_df["date"].iloc[0],
        "endDate": result_df["date"].iloc[-1],
        "tradingDays": int(len(result_df)),
        "initialCash": round(float(total_cash), 2),
        "buyPolicy": "target_weight" if upgraded_policy else "baseline_theta",
        "useKalmanForecastAdjustment": bool(config.get("use_kalman_forecast_adjustment", True)),
        "maxForecastAgeDays": int(config.get("max_forecast_age_days", 0)),
        "deploymentBudget": round(float(config.get("deployment_budget", total_cash)), 2),
        "nominalDeploymentDays": int(config.get("nominal_deployment_days", 0)),
        "maxExtensionDays": int(config.get("max_extension_days", 0)),
        "toolFinalValue": round(float(result_df["tool_portfolio_value"].iloc[-1]), 2),
        "dcaFinalValue": round(float(result_df["dca_portfolio_value"].iloc[-1]), 2),
        "toolVsDca": round(
            float(result_df["tool_portfolio_value"].iloc[-1])
            - float(result_df["dca_portfolio_value"].iloc[-1]),
            2,
        ),
        "toolInvested": round(float(result_df["tool_invested_total"].iloc[-1]), 2),
        "dcaInvested": round(float(result_df["dca_invested_total"].iloc[-1]), 2),
        "toolAvgCost": round(float(result_df["tool_avg_cost"].iloc[-1]), 4),
        "dcaAvgCost": round(float(result_df["dca_avg_cost"].iloc[-1]), 4),
        "toolCashRemaining": round(float(result_df["tool_cash"].iloc[-1]), 2),
        "toolShares": round(float(result_df["tool_shares"].iloc[-1]), 6),
        "investmentDays": int((result_df["tool_daily_investment"] > 0).sum()),
        "noInvestmentDays": int((result_df["tool_daily_investment"] <= 0).sum()),
    }

    comprehensive_summary = {
        "run": {
            "ticker": ticker,
            "startDate": result_df["date"].iloc[0],
            "endDate": result_df["date"].iloc[-1],
            "tradingDays": int(len(result_df)),
            "initialCash": round(float(total_cash), 2),
            "startPrice": round(float(result_df["close"].iloc[0]), 2),
            "endPrice": round(float(result_df["close"].iloc[-1]), 2),
            "priceReturn": round(float(result_df["close"].iloc[-1] / result_df["close"].iloc[0] - 1.0), 6),
            "maxDailyFraction": float(config["max_daily_fraction"]),
            "buyPolicy": "target_weight" if upgraded_policy else "baseline_theta",
            "useKalmanForecastAdjustment": bool(config.get("use_kalman_forecast_adjustment", True)),
            "maxForecastAgeDays": int(config.get("max_forecast_age_days", 0)),
            "deploymentBudget": round(float(config.get("deployment_budget", total_cash)), 2),
            "nominalDeploymentDays": int(config.get("nominal_deployment_days", 0)),
            "maxExtensionDays": int(config.get("max_extension_days", 0)),
        },
        "performance": {
            "toolFinalValue": round(float(result_df["tool_portfolio_value"].iloc[-1]), 2),
            "dcaFinalValue": round(float(result_df["dca_portfolio_value"].iloc[-1]), 2),
            "toolReturn": round(float(result_df["tool_portfolio_value"].iloc[-1]) / float(total_cash) - 1.0, 6),
            "dcaReturn": round(float(result_df["dca_portfolio_value"].iloc[-1]) / float(total_cash) - 1.0, 6),
            "toolVsDca": summary["toolVsDca"],
            "toolMaxDrawdown": round(max_drawdown(result_df["tool_portfolio_value"]), 6),
            "dcaMaxDrawdown": round(max_drawdown(result_df["dca_portfolio_value"]), 6),
        },
        "buying": {
            "investmentDays": summary["investmentDays"],
            "noInvestmentDays": summary["noInvestmentDays"],
            "investmentDayRate": round(summary["investmentDays"] / len(result_df), 6),
            "cashInvested": round(float(result_df["tool_invested_total"].iloc[-1]), 2),
            "cashRemaining": round(float(result_df["tool_cash"].iloc[-1]), 2),
            "sharesBought": round(float(result_df["tool_shares"].iloc[-1]), 6),
            "avgCost": round(float(result_df["tool_avg_cost"].iloc[-1]), 4),
            "avgDailyInvestmentOnBuyDays": safe_round(
                investment_history_df["tool_daily_investment"].mean()
                if not investment_history_df.empty else 0.0,
                2,
            ),
            "maxDailyInvestment": safe_round(
                investment_history_df["tool_daily_investment"].max()
                if not investment_history_df.empty else 0.0,
                2,
            ),
        },
        "signals": {
            "predicted_return": series_stats(result_df["predicted_return"]),
            "drawdown": series_stats(result_df["drawdown"]),
            "volatility_20d": series_stats(result_df["volatility_20d"]),
            "tool_daily_fraction": series_stats(result_df["tool_daily_fraction"]),
            "tool_daily_fraction_denominator": "cash_before_investment",
            "tool_portfolio_daily_fraction": series_stats(result_df["tool_portfolio_daily_fraction"]),
            "predicted_return_20d": series_stats(result_df["predicted_return_20d"]),
            "predicted_return_60d": series_stats(result_df["predicted_return_60d"]),
            "predicted_downside_60d": series_stats(result_df["predicted_downside_60d"]),
            "raw_base_return_20d": series_stats(result_df["raw_base_return_20d"]),
            "raw_base_return_60d": series_stats(result_df["raw_base_return_60d"]),
            "kalman_adjusted_return_20d": series_stats(result_df["kalman_adjusted_return_20d"]),
            "kalman_adjusted_return_60d": series_stats(result_df["kalman_adjusted_return_60d"]),
            "kalman_bias_20d": series_stats(result_df["kalman_bias_20d"]),
            "kalman_bias_60d": series_stats(result_df["kalman_bias_60d"]),
            "kalman_gain_20d": series_stats(result_df["kalman_gain_20d"]),
            "kalman_gain_60d": series_stats(result_df["kalman_gain_60d"]),
            "calibrated_edge": series_stats(result_df["calibrated_edge"]),
            "buy_strength": series_stats(result_df["buy_strength"]),
            "base_buy": series_stats(result_df["base_buy"]),
            "exposure_headroom": series_stats(result_df["exposure_headroom"]),
        },
        "signalsOnBuyDays": {
            "predicted_return": series_stats(investment_history_df["predicted_return"]),
            "drawdown": series_stats(investment_history_df["drawdown"]),
            "volatility_20d": series_stats(investment_history_df["volatility_20d"]),
            "tool_daily_fraction": series_stats(investment_history_df["tool_daily_fraction"]),
            "tool_daily_fraction_denominator": "cash_before_investment",
            "tool_portfolio_daily_fraction": series_stats(
                investment_history_df["tool_portfolio_daily_fraction"]
            ),
            "calibrated_edge": series_stats(investment_history_df["calibrated_edge"]),
            "buy_strength": series_stats(investment_history_df["buy_strength"]),
            "base_buy": series_stats(investment_history_df["base_buy"]),
            "exposure_headroom": series_stats(investment_history_df["exposure_headroom"]),
        },
        "edgeAttribution": {
            "allDays": edge_component_summary(result_df),
            "buyDays": edge_component_summary(investment_history_df),
            "negativeEdgeDays": edge_component_summary(
                result_df.loc[result_df["calibrated_edge"] < 0].copy()
            ),
            "approvedButCappedDays": edge_component_summary(
                result_df.loc[
                    result_df["buy_approved"].astype(bool)
                    & (result_df["base_buy"] > 0)
                    & (result_df["exposure_headroom"] <= 0)
                ].copy()
            ),
        },
        "forecastEvaluation": forecast_evaluation_summary(prediction_df),
        "theta": {
            "refreshCount": int(len(theta_df)),
            "optimizedCount": int((theta_df["status"] == "optimized").sum()) if not theta_df.empty else 0,
            "statuses": theta_df["status"].value_counts().to_dict() if not theta_df.empty else {},
            "latest": theta_df.iloc[-1].to_dict() if not theta_df.empty else None,
            "stats": {
                "theta_g0": series_stats(theta_df["theta_g0"]),
                "theta_drawdown": series_stats(theta_df["theta_drawdown"]),
                "theta_predicted_return": series_stats(theta_df["theta_predicted_return"]),
                "theta_volatility": series_stats(theta_df["theta_volatility"]),
                "theta_mu20": series_stats(theta_df["theta_mu20"]),
                "theta_mu60": series_stats(theta_df["theta_mu60"]),
                "theta_downside": series_stats(theta_df["theta_downside"]),
            } if not theta_df.empty else {},
        },
    }

    return {
        "summary": summary,
        "comprehensiveSummary": comprehensive_summary,
        "strategyChart": build_strategy_chart(result_df),
        "investmentHistory": investment_history_df.to_dict("records"),
        "thetaHistory": theta_df.to_dict("records"),
        "predictionHistory": prediction_df.to_dict("records"),
        "rows": result_df.to_dict("records"),
    }


def run_sell_simulation(
    ticker,
    start_date,
    end_date,
    initial_shares,
    data_dir="datasets",
    config_overrides=None,
):
    user_config_overrides = config_overrides or {}
    config = DEFAULT_CONFIG.copy()
    config.update(user_config_overrides)
    config = normalize_sell_config(config)
    config["initial_shares"] = float(initial_shares)
    ticker = ticker.upper().strip()

    dataset_path = find_or_build_dataset(ticker, data_dir)
    prediction_frame = prepare_prediction_frame(
        ticker,
        config,
        data_dirs=[Path(dataset_path).parent],
        live=False,
    )
    feature_df = add_peak_signals(prediction_frame, config)
    if feature_df.empty:
        raise ValueError("Dataset has no usable sell signal rows.")

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
        raise ValueError("Simulation period must include at least 2 trading days.")

    cash = 0.0
    shares = float(initial_shares)
    rows = []
    prediction_rows = []

    for _, (current_date, row) in enumerate(sim_df.iterrows()):
        signal_position = feature_df.index.get_loc(current_date) - 1
        if signal_position < 0:
            continue

        signal_date = feature_df.index[signal_position]
        signal_row = feature_df.iloc[signal_position]
        decision_price = float(row["close"])
        shares_before = shares
        cash_before = cash

        sale = compute_sell_decision(
            shares=shares,
            price=decision_price,
            signals=signal_row,
            config=config,
        )

        shares_to_sell = float(sale["shares_to_sell"])
        cash += shares_to_sell * decision_price
        shares -= shares_to_sell
        lucky_value = cash + shares * decision_price
        hold_value = float(initial_shares) * decision_price

        prediction_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "model_signal_date": pd.Timestamp(
                    signal_row.get("model_trained_through", signal_date)
                ).strftime("%Y-%m-%d"),
                "predicted_return_20d": float(signal_row["predicted_return_20d"]),
                "predicted_return_60d": float(signal_row["predicted_return_60d"]),
                "predicted_downside_60d": float(signal_row["predicted_downside_60d"]),
                "raw_base_return_20d": float(signal_row.get("raw_base_return_20d", signal_row["predicted_return_20d"])),
                "raw_base_return_60d": float(signal_row.get("raw_base_return_60d", signal_row["predicted_return_60d"])),
                "kalman_adjusted_return_20d": float(signal_row.get("kalman_adjusted_return_20d", signal_row["predicted_return_20d"])),
                "kalman_adjusted_return_60d": float(signal_row.get("kalman_adjusted_return_60d", signal_row["predicted_return_60d"])),
                "predicted_return": float(signal_row["predicted_return_60d"]),
                "p_rally": float(signal_row.get("p_rally", 0.0)),
                "p_correction": float(signal_row.get("p_correction", 0.0)),
                "p_riskoff": float(signal_row.get("p_riskoff", 0.0)),
            }
        )

        rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "decision_price": decision_price,
                "signal_close": float(signal_row["close"]),
                "shares_before": float(shares_before),
                "shares_to_sell": shares_to_sell,
                "shares_after": float(shares),
                "cash_before": float(cash_before),
                "cash": float(cash),
                "sell_value": float(sale["sell_value"]),
                "daily_sell_fraction": float(sale["daily_sell_fraction"]),
                "sell_edge": float(sale["sell_edge"]),
                "sell_approved": bool(sale["sell_approved"]),
                "sell_allowed": bool(sale["sell_allowed"]),
                "sell_intensity": float(sale["sell_intensity"]),
                "hold_reason": sale["hold_reason"],
                "peak_score": float(sale["peak_score"]),
                "near_high_score": float(sale["near_high_score"]),
                "overextension_score": float(sale["overextension_score"]),
                "runup_score": float(sale["runup_score"]),
                "momentum_weakening_score": float(sale["momentum_weakening_score"]),
                "predicted_return_20d": float(sale["predicted_return_20d"]),
                "predicted_return_60d": float(sale["predicted_return_60d"]),
                "predicted_downside_60d": float(sale["predicted_downside_60d"]),
                "p_rally": float(sale["p_rally"]),
                "p_correction": float(sale["p_correction"]),
                "p_riskoff": float(sale["p_riskoff"]),
                "lucky_cash": float(cash),
                "lucky_shares": float(shares),
                "lucky_stock_value": float(shares * decision_price),
                "lucky_portfolio_value": float(lucky_value),
                "hold_shares": float(initial_shares),
                "hold_portfolio_value": float(hold_value),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No usable sell simulation rows were produced.")

    total_lucky_sold = float(result_df["shares_to_sell"].sum())
    linear_daily_sale = total_lucky_sold / len(result_df) if total_lucky_sold > 0 else 0.0
    linear_cash = 0.0
    linear_shares = float(initial_shares)
    linear_rows = []
    for _, row in result_df.iterrows():
        decision_price = float(row["decision_price"])
        linear_sale = min(linear_daily_sale, linear_shares)
        linear_cash += linear_sale * decision_price
        linear_shares -= linear_sale
        linear_rows.append(
            {
                "linear_shares_to_sell": float(linear_sale),
                "linear_cash": float(linear_cash),
                "linear_shares": float(linear_shares),
                "linear_stock_value": float(linear_shares * decision_price),
                "linear_portfolio_value": float(linear_cash + linear_shares * decision_price),
            }
        )
    result_df = pd.concat([result_df.reset_index(drop=True), pd.DataFrame(linear_rows)], axis=1)

    sell_history_df = result_df.loc[
        result_df["shares_to_sell"] > 0,
        [
            "date",
            "signal_date",
            "decision_price",
            "shares_before",
            "shares_to_sell",
            "shares_after",
            "cash",
            "sell_edge",
            "sell_approved",
            "sell_intensity",
            "peak_score",
            "near_high_score",
            "momentum_weakening_score",
            "predicted_return_20d",
            "predicted_return_60d",
            "predicted_downside_60d",
            "p_rally",
            "p_correction",
            "p_riskoff",
        ],
    ].copy()
    prediction_df = pd.DataFrame(prediction_rows)

    initial_value = float(initial_shares) * float(result_df["decision_price"].iloc[0])
    summary = {
        "ticker": ticker,
        "startDate": result_df["date"].iloc[0],
        "endDate": result_df["date"].iloc[-1],
        "tradingDays": int(len(result_df)),
        "initialShares": round(float(initial_shares), 6),
        "initialValue": round(initial_value, 2),
        "luckyFinalValue": round(float(result_df["lucky_portfolio_value"].iloc[-1]), 2),
        "holdFinalValue": round(float(result_df["hold_portfolio_value"].iloc[-1]), 2),
        "linearFinalValue": round(float(result_df["linear_portfolio_value"].iloc[-1]), 2),
        "luckyVsHold": round(
            float(result_df["lucky_portfolio_value"].iloc[-1])
            - float(result_df["hold_portfolio_value"].iloc[-1]),
            2,
        ),
        "luckyVsLinear": round(
            float(result_df["lucky_portfolio_value"].iloc[-1])
            - float(result_df["linear_portfolio_value"].iloc[-1]),
            2,
        ),
        "sharesSold": round(total_lucky_sold, 6),
        "sharesRemaining": round(float(result_df["lucky_shares"].iloc[-1]), 6),
        "sellDays": int((result_df["shares_to_sell"] > 0).sum()),
        "noSellDays": int((result_df["shares_to_sell"] <= 0).sum()),
    }

    comprehensive_summary = {
        "run": {
            "ticker": ticker,
            "startDate": summary["startDate"],
            "endDate": summary["endDate"],
            "tradingDays": summary["tradingDays"],
            "initialShares": summary["initialShares"],
            "startPrice": round(float(result_df["decision_price"].iloc[0]), 2),
            "endPrice": round(float(result_df["decision_price"].iloc[-1]), 2),
            "maxDailySellFraction": float(config["max_daily_sell_fraction"]),
        },
        "performance": {
            "luckyFinalValue": summary["luckyFinalValue"],
            "holdFinalValue": summary["holdFinalValue"],
            "linearFinalValue": summary["linearFinalValue"],
            "luckyVsHold": summary["luckyVsHold"],
            "luckyVsLinear": summary["luckyVsLinear"],
            "luckyMaxDrawdown": round(max_drawdown(result_df["lucky_portfolio_value"]), 6),
            "holdMaxDrawdown": round(max_drawdown(result_df["hold_portfolio_value"]), 6),
            "linearMaxDrawdown": round(max_drawdown(result_df["linear_portfolio_value"]), 6),
        },
        "selling": {
            "sellDays": summary["sellDays"],
            "noSellDays": summary["noSellDays"],
            "sellDayRate": round(summary["sellDays"] / len(result_df), 6),
            "sharesSold": summary["sharesSold"],
            "sharesRemaining": summary["sharesRemaining"],
            "avgDailySharesSoldOnSellDays": safe_round(
                sell_history_df["shares_to_sell"].mean() if not sell_history_df.empty else 0.0,
                6,
            ),
            "maxDailySharesSold": safe_round(
                sell_history_df["shares_to_sell"].max() if not sell_history_df.empty else 0.0,
                6,
            ),
        },
        "signals": {
            "sell_edge": series_stats(result_df["sell_edge"]),
            "sell_intensity": series_stats(result_df["sell_intensity"]),
            "peak_score": series_stats(result_df["peak_score"]),
            "near_high_score": series_stats(result_df["near_high_score"]),
            "momentum_weakening_score": series_stats(result_df["momentum_weakening_score"]),
            "predicted_return_20d": series_stats(result_df["predicted_return_20d"]),
            "predicted_return_60d": series_stats(result_df["predicted_return_60d"]),
            "predicted_downside_60d": series_stats(result_df["predicted_downside_60d"]),
        },
    }

    return {
        "summary": summary,
        "comprehensiveSummary": comprehensive_summary,
        "rows": result_df.to_dict("records"),
        "sellHistory": sell_history_df.to_dict("records"),
        "predictionHistory": prediction_df.to_dict("records"),
        "strategyChart": build_sell_strategy_chart(result_df),
    }


def compare_baseline_upgraded_dca(
    tickers,
    start_dates,
    end_date,
    total_cash,
    data_dir="datasets",
    config_overrides=None,
    cleanup_data=False,
):
    config_overrides = config_overrides or {}
    rows = []
    for ticker in tickers:
        for start_date in start_dates:
            baseline = run_buy_simulation(
                ticker,
                start_date,
                end_date,
                total_cash,
                data_dir=data_dir,
                config_overrides={**config_overrides, "buy_policy": "baseline_theta"},
                cleanup_data=cleanup_data,
            )
            upgraded = run_buy_simulation(
                ticker,
                start_date,
                end_date,
                total_cash,
                data_dir=data_dir,
                config_overrides={**config_overrides, "buy_policy": "target_weight"},
                cleanup_data=cleanup_data,
            )

            baseline_summary = baseline["comprehensiveSummary"]
            upgraded_summary = upgraded["comprehensiveSummary"]
            rows.append(
                {
                    "ticker": ticker.upper().strip(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "baseline_final_value": baseline_summary["performance"]["toolFinalValue"],
                    "upgraded_final_value": upgraded_summary["performance"]["toolFinalValue"],
                    "dca_final_value": baseline_summary["performance"]["dcaFinalValue"],
                    "baseline_vs_dca": baseline_summary["performance"]["toolVsDca"],
                    "upgraded_vs_dca": upgraded_summary["performance"]["toolVsDca"],
                    "upgraded_vs_baseline": round(
                        upgraded_summary["performance"]["toolFinalValue"]
                        - baseline_summary["performance"]["toolFinalValue"],
                        2,
                    ),
                    "baseline_avg_entry_price": baseline_summary["buying"]["avgCost"],
                    "upgraded_avg_entry_price": upgraded_summary["buying"]["avgCost"],
                    "baseline_max_drawdown": baseline_summary["performance"]["toolMaxDrawdown"],
                    "upgraded_max_drawdown": upgraded_summary["performance"]["toolMaxDrawdown"],
                    "baseline_cash_remaining": baseline_summary["buying"]["cashRemaining"],
                    "upgraded_cash_remaining": upgraded_summary["buying"]["cashRemaining"],
                }
            )
    return pd.DataFrame(rows)
