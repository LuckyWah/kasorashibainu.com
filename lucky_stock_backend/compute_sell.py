import numpy as np
import pandas as pd

from prediction_model import prepare_prediction_frame


DEFAULT_CONFIG = {
    "max_daily_fraction": 0.05,
    "max_daily_sell_fraction": 0.03,
    "min_stock_weight": 0.0,
    "max_target_weight": 1.0,
    "sell_min_near_high_score": 0.70,
    "sell_runup_scale": 0.20,
    "sell_overextension_scale": 0.08,
    "sell_new_high_scale": 0.15,
    "sell_momentum_weakening_scale": 0.08,
    "sell_mu20_scale": 0.08,
    "sell_mu60_scale": 0.20,
    "sell_downside_scale": 0.20,
    "sell_disagreement_scale": 0.20,
    "sell_momentum_scale": 0.08,
    "correction20_weight": 0.45,
    "correction60_weight": 0.55,
    "downside_weight": 0.35,
    "peak_correction_weight": 0.30,
    "weakening_weight": 0.25,
    "peak_riskoff_weight": 0.25,
    "disagreement_weight": 0.20,
    "rally_continuation_weight": 0.35,
    "sell_entry_threshold": 0.15,
    "sell_strong_edge": 0.70,
    "sell_intensity_gamma": 0.75,
}


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


def normalize_config(config):
    source = config or {}
    merged = DEFAULT_CONFIG.copy()
    merged.update(source)
    if "max_daily_fraction" not in source and "max_daily_sell_fraction" in source:
        merged["max_daily_fraction"] = merged.get("max_daily_sell_fraction", 0.05)
    if "initial_shares" in merged:
        merged["initial_shares"] = float(merged["initial_shares"])
    if "user_sell_strength" in merged:
        merged["user_sell_strength"] = _clip01(merged["user_sell_strength"])
    return merged


def sell_target_weight(sell_strength, config=None):
    config = normalize_config(config)
    min_stock_weight = _clip01(config.get("min_stock_weight", 0.0))
    max_stock_weight = _clip01(config.get("max_target_weight", 1.0))
    if max_stock_weight < min_stock_weight:
        max_stock_weight = min_stock_weight
    strength = _clip01(sell_strength)
    return float(max_stock_weight - (max_stock_weight - min_stock_weight) * strength)


def _clip_series(series, lower=0.0, upper=None):
    clipped = series.clip(lower=lower)
    if upper is not None:
        clipped = clipped.clip(upper=upper)
    return clipped


def add_peak_signals(frame, config):
    """Add peak-side policy signals using only data available at each row."""
    config = normalize_config(config)
    sell_df = frame.copy()
    close = sell_df["close"].astype(float)

    recent_high = close.rolling(60, min_periods=5).max()
    rolling_max = pd.to_numeric(sell_df.get("rolling_max", close.cummax()), errors="coerce")
    recent_high = recent_high.fillna(rolling_max)
    recent_high = recent_high.replace(0, np.nan)
    prior_high = close.rolling(60, min_periods=5).max().shift(1).replace(0, np.nan)

    ma_50 = pd.to_numeric(
        sell_df.get("ma_50", close.rolling(50, min_periods=5).mean()),
        errors="coerce",
    )
    ma_50 = ma_50.replace(0, np.nan)

    momentum_20d = sell_df["momentum_20d"].astype(float)
    momentum_60d = sell_df["momentum_60d"].astype(float)
    prior_momentum_20d = momentum_20d.shift(5).fillna(momentum_20d)

    runup_scale = max(_safe_float(config.get("sell_runup_scale", 0.20), 0.20), 1e-9)
    overextension_scale = max(
        _safe_float(config.get("sell_overextension_scale", 0.08), 0.08),
        1e-9,
    )
    new_high_scale = max(_safe_float(config.get("sell_new_high_scale", 0.15), 0.15), 1e-9)
    weakening_scale = max(
        _safe_float(config.get("sell_momentum_weakening_scale", 0.08), 0.08),
        1e-9,
    )
    min_near_high = _safe_float(config.get("sell_min_near_high_score", 0.70), 0.70)

    sell_df["near_high_score"] = _clip_series((close / recent_high - 0.90) / 0.10, 0.0, 1.0)
    sell_df["overextension_score"] = _clip_series(close / ma_50 - 1.0, 0.0)
    sell_df["runup_score"] = _clip_series(momentum_20d, 0.0)
    sell_df["momentum_weakening_score"] = _clip_series(
        np.maximum(momentum_60d - momentum_20d, prior_momentum_20d - momentum_20d),
        0.0,
    )
    sell_df["drawdown_score"] = _clip_series(sell_df["drawdown"].astype(float), 0.0)
    sell_df["new_high_score"] = ((close / prior_high - 1.0) / new_high_scale).clip(
        lower=0.0,
        upper=1.0,
    ).fillna(0.0)

    sell_df["peak_near_high_component"] = sell_df["near_high_score"].apply(_clip01)
    sell_df["peak_runup_component"] = (
        sell_df["runup_score"].astype(float) / runup_scale
    ).clip(lower=0.0, upper=1.0)
    sell_df["peak_overextension_component"] = (
        sell_df["overextension_score"].astype(float) / overextension_scale
    ).clip(lower=0.0, upper=1.0)
    sell_df["peak_weakening_component"] = (
        sell_df["momentum_weakening_score"].astype(float) / weakening_scale
    ).clip(lower=0.0, upper=1.0)

    peak_score = (
        0.35 * sell_df["peak_near_high_component"]
        + 0.25 * sell_df["peak_runup_component"]
        + 0.20 * sell_df["peak_overextension_component"]
        + 0.10 * sell_df["new_high_score"]
        + 0.10 * sell_df["peak_weakening_component"]
    )
    sell_df["peak_score"] = peak_score.where(
        sell_df["near_high_score"] >= min_near_high,
        0.0,
    ).clip(lower=0.0, upper=1.0)

    required = [
        "near_high_score",
        "overextension_score",
        "runup_score",
        "momentum_weakening_score",
        "peak_score",
    ]
    sell_df[required] = sell_df[required].apply(pd.to_numeric, errors="coerce")
    sell_df[required] = sell_df[required].replace([np.inf, -np.inf], np.nan)
    return sell_df.dropna(subset=required).copy()


def compute_sell_decision(cash, shares, price, signals, config):
    config = normalize_config(config)
    cash = max(0.0, _safe_float(cash, 0.0))
    shares = max(0.0, _safe_float(shares, 0.0))
    price = max(0.0, _safe_float(price, 0.0))

    max_daily_fraction = max(
        0.0,
        _safe_float(config.get("max_daily_fraction", 0.05), 0.05),
    )

    mu20 = _safe_float(signals.get("predicted_return_20d", signals.get("predicted_return", 0.0)))
    mu60 = _safe_float(signals.get("predicted_return_60d", signals.get("predicted_return", 0.0)))
    predicted_downside_60d = _safe_float(signals.get("predicted_downside_60d", 0.0))
    peak_score = _clip01(signals.get("peak_score", 0.0))
    momentum_weakening_score = _clip01(signals.get("momentum_weakening_score", 0.0))
    momentum_20d = _safe_float(signals.get("momentum_20d", 0.0))
    p_rally = _clip01(signals.get("p_rally", 1.0 / 3.0))
    p_correction = _clip01(signals.get("p_correction", 1.0 / 3.0))
    p_riskoff = _clip01(signals.get("p_riskoff", 1.0 / 3.0))

    sell_mu20_scale = max(
        1e-9,
        _safe_float(config.get("edge_mu20_scale", config.get("sell_mu20_scale", 0.08)), 0.08),
    )
    sell_mu60_scale = max(
        1e-9,
        _safe_float(config.get("edge_mu60_scale", config.get("sell_mu60_scale", 0.20)), 0.20),
    )
    sell_downside_scale = max(
        1e-9,
        _safe_float(config.get("sell_downside_scale", 0.20), 0.20),
    )
    sell_disagreement_scale = max(
        1e-9,
        _safe_float(config.get("sell_disagreement_scale", 0.20), 0.20),
    )
    sell_momentum_scale = max(
        1e-9,
        _safe_float(config.get("sell_momentum_scale", 0.08), 0.08),
    )

    upside_component = (
        _safe_float(config.get("edge_mu20_weight", config.get("correction20_weight", 0.55)), 0.55)
        * np.tanh(mu20 / sell_mu20_scale)
        + _safe_float(config.get("edge_mu60_weight", config.get("correction60_weight", 0.45)), 0.45)
        * np.tanh(mu60 / sell_mu60_scale)
    )
    downside_component = (
        _safe_float(config.get("downside_edge_weight", config.get("downside_weight", 0.35)), 0.35)
        * np.tanh(abs(predicted_downside_60d) / sell_downside_scale)
    )
    disagreement_component = (
        _safe_float(config.get("disagreement_edge_weight", config.get("disagreement_weight", 0.20)), 0.20)
        * np.tanh(abs(mu20 - mu60) / sell_disagreement_scale)
    )
    peak_correction_component = (
        _safe_float(config.get("peak_correction_weight", 0.30), 0.30)
        * peak_score
        * p_correction
    )
    weakening_component = (
        _safe_float(config.get("weakening_weight", 0.25), 0.25)
        * peak_score
        * momentum_weakening_score
    )
    healthy_rally_component = (
        _safe_float(config.get("rally_continuation_weight", 0.35), 0.35)
        * peak_score
        * p_rally
        * np.tanh(max(momentum_20d, 0.0) / sell_momentum_scale)
    )
    riskoff_component = (
        _safe_float(config.get("peak_riskoff_weight", 0.25), 0.25)
        * peak_score
        * p_riskoff
    )
    volatility_component = (
        _safe_float(config.get("volatility_edge_weight", 0.0), 0.0)
        * np.tanh(max(0.0, _safe_float(signals.get("volatility_20d", 0.0))) / max(1e-9, _safe_float(config.get("volatility_edge_scale", 0.50), 0.50)))
    )

    sell_edge = (
        -upside_component
        + downside_component
        - disagreement_component
        + peak_correction_component
        + weakening_component
        - healthy_rally_component
        + riskoff_component
        + volatility_component
    )
    sell_edge = float(sell_edge)

    sell_entry_threshold = _safe_float(config.get("sell_entry_threshold", 0.15), 0.15)
    sell_strong_edge = _safe_float(config.get("sell_strong_edge", 0.70), 0.70)
    if sell_strong_edge <= sell_entry_threshold:
        sell_strong_edge = sell_entry_threshold + 1e-9

    sell_approved = bool(sell_edge > sell_entry_threshold and peak_score > 0.0)
    sell_intensity = float(
        np.clip(
            (sell_edge - sell_entry_threshold) / (sell_strong_edge - sell_entry_threshold),
            0.0,
            1.0,
        )
    )
    sell_intensity_gamma = max(
        1e-9,
        _safe_float(config.get("sell_intensity_gamma", 0.75), 0.75),
    )

    if shares <= 0:
        hold_reason = "no_shares"
        sell_approved = False
    elif price <= 0:
        hold_reason = "invalid_price"
        sell_approved = False
    current_stock_value = shares * price
    portfolio_value = cash + current_stock_value
    current_stock_weight = current_stock_value / portfolio_value if portfolio_value > 0 else 0.0
    user_sell_strength = _clip01(config.get("user_sell_strength", 1.0))
    effective_sell_strength = _clip01(sell_intensity * user_sell_strength)
    target_stock_weight = sell_target_weight(effective_sell_strength, config)
    desired_stock_value = target_stock_weight * portfolio_value
    raw_sell_needed = max(0.0, current_stock_value - desired_stock_value)
    max_daily_sell = max(0.0, min(portfolio_value * max_daily_fraction, current_stock_value))
    base_sell = (
        max_daily_sell * effective_sell_strength ** sell_intensity_gamma
        if sell_approved
        else 0.0
    )
    sell_value = min(raw_sell_needed, base_sell, max_daily_sell, current_stock_value)

    if shares <= 0:
        hold_reason = "no_shares"
        sell_approved = False
    elif price <= 0:
        hold_reason = "invalid_price"
        sell_approved = False
    elif portfolio_value <= 0:
        hold_reason = "empty_portfolio"
        sell_approved = False
    elif max_daily_fraction <= 0:
        hold_reason = "max_sell_fraction_zero"
        sell_approved = False
    elif peak_score <= 0.0:
        hold_reason = "not_near_peak"
        sell_approved = False
    elif sell_edge <= sell_entry_threshold:
        hold_reason = "sell_edge_below_threshold"
        sell_approved = False
    else:
        hold_reason = "sell_approved"

    if not sell_approved:
        sell_value = 0.0

    shares_to_sell = sell_value / price if price > 0 else 0.0

    daily_sell_fraction = shares_to_sell / shares if shares > 0 else 0.0
    return {
        "shares_to_sell": float(shares_to_sell),
        "sell_value": float(shares_to_sell * price),
        "daily_sell_fraction": float(daily_sell_fraction),
        "target_daily_sell_fraction": float(daily_sell_fraction),
        "shares_remaining": float(shares - shares_to_sell),
        "cash": float(cash + sell_value),
        "cash_before": float(cash),
        "portfolio_value": float(portfolio_value),
        "current_stock_value": float(current_stock_value),
        "current_stock_weight": float(current_stock_weight),
        "target_stock_weight": float(target_stock_weight),
        "desired_stock_value": float(desired_stock_value),
        "raw_sell_needed": float(raw_sell_needed),
        "max_daily_sell": float(max_daily_sell),
        "portfolio_daily_cap": float(portfolio_value * max_daily_fraction),
        "base_sell": float(base_sell),
        "sell_limited_by_cap": bool(base_sell < raw_sell_needed) if sell_approved else False,
        "sell_approved": bool(sell_approved),
        "sell_allowed": bool(sell_approved),
        "hold_reason": hold_reason,
        "sell_edge": sell_edge,
        "sell_intensity": sell_intensity if sell_approved else 0.0,
        "adjusted_sell_intensity": effective_sell_strength if sell_approved else 0.0,
        "raw_sell_intensity": sell_intensity,
        "user_sell_strength": float(user_sell_strength),
        "sell_strength": float(effective_sell_strength if sell_approved else 0.0),
        "raw_sell_strength": float(sell_intensity),
        "sell_entry_threshold": float(sell_entry_threshold),
        "sell_strong_edge": float(sell_strong_edge),
        "sell_intensity_gamma": float(sell_intensity_gamma),
        "peak_score": peak_score,
        "near_high_score": _safe_float(signals.get("near_high_score", 0.0)),
        "overextension_score": _safe_float(signals.get("overextension_score", 0.0)),
        "runup_score": _safe_float(signals.get("runup_score", 0.0)),
        "momentum_weakening_score": momentum_weakening_score,
        "predicted_return": mu60,
        "predicted_return_20d": mu20,
        "predicted_return_60d": mu60,
        "predicted_downside_60d": predicted_downside_60d,
        "p_rally": p_rally,
        "p_correction": p_correction,
        "p_riskoff": p_riskoff,
        "sell_edge_upside_component": float(upside_component),
        "sell_edge_downside_component": float(downside_component),
        "sell_edge_disagreement_component": float(disagreement_component),
        "sell_edge_peak_correction_component": float(peak_correction_component),
        "sell_edge_weakening_component": float(weakening_component),
        "sell_edge_healthy_rally_component": float(healthy_rally_component),
        "sell_edge_riskoff_component": float(riskoff_component),
        "sell_edge_volatility_component": float(volatility_component),
    }


def compute_today_sell(ticker, cash, shares=None, config=None, data_dirs=None):
    if isinstance(cash, dict) and config is None:
        config = cash
        cash = config.get("cash_balance", 0.0)
        shares = config.get("current_shares", config.get("initial_shares", 0.0))
    config = normalize_config(config)
    cash = max(0.0, _safe_float(cash, 0.0))
    shares = max(0.0, _safe_float(shares, config.get("initial_shares", 0.0)))
    initial_shares = max(0.0, float(config.get("initial_shares", shares)))
    prediction_frame = prepare_prediction_frame(
        ticker,
        config,
        data_dirs=data_dirs,
        live=True,
    )
    signal_frame = add_peak_signals(prediction_frame, config)
    if signal_frame.empty:
        raise ValueError("Dataset has no usable sell signal rows.")

    today_row = signal_frame.iloc[-1]
    today_date = today_row.name
    price = float(today_row["close"])
    sale = compute_sell_decision(cash, shares, price, today_row, config)

    return {
        "date": today_date,
        "ticker": ticker.upper().strip(),
        "price": price,
        "initial_shares": initial_shares,
        "current_shares": shares,
        "cash_balance": cash,
        "reference_high": float(today_row.get("rolling_max", price)),
        "drawdown_score": float(today_row.get("drawdown", 0.0)),
        "volatility_20d": float(today_row.get("volatility_20d", 0.0)),
        "new_high_score": float(today_row.get("new_high_score", 0.0)),
        "raw_base_return_20d": float(today_row.get("raw_base_return_20d", today_row["predicted_return_20d"])),
        "raw_base_return_60d": float(today_row.get("raw_base_return_60d", today_row["predicted_return_60d"])),
        "kalman_adjusted_return_20d": float(today_row.get("kalman_adjusted_return_20d", today_row["predicted_return_20d"])),
        "kalman_adjusted_return_60d": float(today_row.get("kalman_adjusted_return_60d", today_row["predicted_return_60d"])),
        "kalman_bias_20d": float(today_row.get("kalman_bias_20d", 0.0)),
        "kalman_bias_60d": float(today_row.get("kalman_bias_60d", 0.0)),
        "kalman_gain_20d": float(today_row.get("kalman_gain_20d", 0.0)),
        "kalman_gain_60d": float(today_row.get("kalman_gain_60d", 0.0)),
        "kalman_update_count_20d": int(today_row.get("kalman_update_count_20d", 0)),
        "kalman_update_count_60d": int(today_row.get("kalman_update_count_60d", 0)),
        "theta_g0": 0.0,
        "theta_near_high": 0.0,
        "theta_runup": 0.0,
        "theta_predicted_return": 0.0,
        "theta_volatility": 0.0,
        "theta_momentum_weakening": 0.0,
        "theta_drawdown": 0.0,
        **sale,
    }
