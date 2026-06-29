from prediction_model import (
    get_live_prediction_snapshot,
    prepare_prediction_frame,
)

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


TARGET_THETA_BOUNDS = [
    (-2.5, 2.5),   # theta_0
    (0.0, 15.0),   # mu_20
    (0.0, 15.0),   # mu_60
    (0.0, 15.0),   # downside penalty
    (0.0, 1e-9),   # raw drawdown disabled; use regime-conditioned terms
    (0.0, 8.0),    # drawdown * correction
    (0.0, 8.0),    # risk-off drawdown restraint
    (0.0, 8.0),    # momentum * rally
]


def sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def target_weight(theta, signals, max_weight=1.0):
    mu20 = float(signals.get("predicted_return_20d", signals.get("predicted_return", 0.0)))
    mu60 = float(signals.get("predicted_return_60d", signals.get("predicted_return", 0.0)))
    downside = abs(float(signals.get("predicted_downside_60d", 0.0)))
    drawdown = float(signals.get("drawdown", 0.0))
    p_correction = float(signals.get("p_correction", 1.0 / 3.0))
    p_riskoff = float(signals.get("p_riskoff", 1.0 / 3.0))
    p_rally = float(signals.get("p_rally", 1.0 / 3.0))
    momentum = float(signals.get("momentum_20d", 0.0))

    score = (
        float(theta[0])
        + float(theta[1]) * mu20
        + float(theta[2]) * mu60
        - float(theta[3]) * downside
        + float(theta[4]) * drawdown
        + float(theta[5]) * drawdown * p_correction
        - float(theta[6]) * drawdown * p_riskoff
        + float(theta[7]) * momentum * p_rally
    )
    return max(0.0, min(float(max_weight), float(max_weight) * sigmoid(score)))


def target_weight_details(theta, signals, max_weight=1.0):
    mu20 = float(signals.get("predicted_return_20d", signals.get("predicted_return", 0.0)))
    mu60 = float(signals.get("predicted_return_60d", signals.get("predicted_return", 0.0)))
    downside = abs(float(signals.get("predicted_downside_60d", 0.0)))
    drawdown = float(signals.get("drawdown", 0.0))
    p_correction = float(signals.get("p_correction", 1.0 / 3.0))
    p_riskoff = float(signals.get("p_riskoff", 1.0 / 3.0))
    p_rally = float(signals.get("p_rally", 1.0 / 3.0))
    momentum = float(signals.get("momentum_20d", 0.0))

    contributions = {
        "contribution_intercept": float(theta[0]),
        "contribution_mu20": float(theta[1]) * mu20,
        "contribution_mu60": float(theta[2]) * mu60,
        "contribution_downside": -float(theta[3]) * downside,
        "contribution_drawdown": float(theta[4]) * drawdown,
        "contribution_drawdown_correction": float(theta[5]) * drawdown * p_correction,
        "contribution_drawdown_riskoff": -float(theta[6]) * drawdown * p_riskoff,
        "contribution_momentum_rally": float(theta[7]) * momentum * p_rally,
    }
    raw_score = sum(contributions.values())
    weight = max(0.0, min(float(max_weight), float(max_weight) * sigmoid(raw_score)))
    return {
        **contributions,
        "raw_score": raw_score,
        "target_weight": weight,
    }


def staleness_scale_for_age(forecast_age_days, config):
    forecast_age_days = max(0.0, _safe_float(forecast_age_days, 0.0))
    max_age = max(0.0, _safe_float(config.get("max_forecast_age_days", 90.0), 90.0))
    if forecast_age_days <= max_age:
        return 1.0

    half_life = max(
        1.0,
        _safe_float(config.get("forecast_staleness_half_life_days", 56.0), 56.0),
    )
    excess_age = forecast_age_days - max_age
    floor = max(0.0, _safe_float(config.get("stale_forecast_weight_scale", 0.0), 0.0))
    decayed = float(np.exp(-np.log(2.0) * excess_age / half_life))
    return max(floor, min(1.0, decayed))


def risk_ceiling(signals, config, max_weight):
    mu20 = _safe_float(signals.get("predicted_return_20d", signals.get("predicted_return", 0.0)))
    mu60 = _safe_float(signals.get("predicted_return_60d", signals.get("predicted_return", 0.0)))
    downside_loss = abs(_safe_float(signals.get("predicted_downside_60d", 0.0)))
    disagreement = abs(mu20 - mu60)
    p_riskoff = _safe_float(signals.get("p_riskoff", 1.0 / 3.0), 1.0 / 3.0)

    score = (
        _safe_float(config.get("risk_ceiling_intercept", 1.25), 1.25)
        + _safe_float(config.get("risk_ceiling_mu20_weight", 6.0), 6.0) * mu20
        + _safe_float(config.get("risk_ceiling_mu60_weight", 4.0), 4.0) * mu60
        - _safe_float(config.get("risk_ceiling_downside_weight", 8.0), 8.0) * downside_loss
        - _safe_float(config.get("risk_ceiling_disagreement_weight", 4.0), 4.0) * disagreement
        - _safe_float(config.get("risk_ceiling_riskoff_weight", 1.25), 1.25) * p_riskoff
    )
    ceiling = max(0.0, min(float(max_weight), float(max_weight) * sigmoid(score)))
    return ceiling, score, disagreement


def model_pace_multiplier(details, max_weight, config):
    """Convert the learned policy target into a relative buying pace."""
    min_multiplier = _safe_float(config.get("min_model_multiplier", 0.50), 0.50)
    max_multiplier = _safe_float(config.get("max_model_multiplier", 1.75), 1.75)
    if max_multiplier < min_multiplier:
        max_multiplier = min_multiplier

    confidence = np.clip(
        details["target_weight"] / max(float(max_weight), 1e-9),
        0.0,
        1.0,
    )
    edge = 2.0 * confidence - 1.0

    if edge >= 0.0:
        raw_multiplier = 1.0 + edge * (max_multiplier - 1.0)
    else:
        raw_multiplier = 1.0 + edge * (1.0 - min_multiplier)

    return float(raw_multiplier), float(confidence)


def apply_risk_and_staleness_to_multiplier(
    raw_multiplier,
    ceiling,
    max_weight,
    forecast_age_days,
    config,
):
    """Keep risk as a diagnostic and use staleness only as a bounded clamp."""
    max_multiplier = _safe_float(config.get("max_model_multiplier", 1.75), 1.75)
    risk_acceleration_cap = 1.0 + (max_multiplier - 1.0) * np.clip(
        ceiling / max(float(max_weight), 1e-9),
        0.0,
        1.0,
    )
    signal_multiplier = raw_multiplier

    max_age = _safe_float(config.get("max_forecast_age_days", 90.0), 90.0)
    if forecast_age_days > max_age:
        stale_min = _safe_float(config.get("stale_min_multiplier", 0.75), 0.75)
        stale_max = _safe_float(config.get("stale_max_multiplier", 1.25), 1.25)
        if stale_max < stale_min:
            stale_max = stale_min
        effective_multiplier = float(np.clip(signal_multiplier, stale_min, stale_max))
    else:
        effective_multiplier = float(signal_multiplier)

    return float(effective_multiplier), float(signal_multiplier), float(risk_acceleration_cap)


def tactical_ramp_days(signals, config):
    downside_loss = abs(_safe_float(signals.get("predicted_downside_60d", 0.0)))
    disagreement = abs(
        _safe_float(signals.get("predicted_return_20d", 0.0))
        - _safe_float(signals.get("predicted_return_60d", 0.0))
    )
    volatility = max(0.0, _safe_float(signals.get("volatility_20d", 0.0)))
    ramp_days = (
        _safe_float(config.get("base_ramp_days", 12.0), 12.0)
        + downside_loss * _safe_float(config.get("downside_ramp_scale", 20.0), 20.0)
        + disagreement * _safe_float(config.get("disagreement_ramp_scale", 20.0), 20.0)
        + volatility * _safe_float(config.get("volatility_ramp_scale", 1.0), 1.0)
    )
    min_days = max(1.0, _safe_float(config.get("min_ramp_days", 4.0), 4.0))
    max_days = max(min_days, _safe_float(config.get("max_ramp_days", 30.0), 30.0))
    return float(np.clip(ramp_days, min_days, max_days))


def calibrated_tactical_edge(signals, config, staleness_scale=1.0):
    mu20 = _safe_float(signals.get("predicted_return_20d", signals.get("predicted_return", 0.0)))
    mu60 = _safe_float(signals.get("predicted_return_60d", signals.get("predicted_return", 0.0)))
    downside_loss = abs(_safe_float(signals.get("predicted_downside_60d", 0.0)))
    disagreement = abs(mu20 - mu60)
    drawdown = max(0.0, _safe_float(signals.get("drawdown", 0.0)))
    momentum = _safe_float(signals.get("momentum_20d", 0.0))
    volatility = max(0.0, _safe_float(signals.get("volatility_20d", 0.0)))
    p_correction = np.clip(_safe_float(signals.get("p_correction", 1.0 / 3.0), 1.0 / 3.0), 0.0, 1.0)
    p_rally = np.clip(_safe_float(signals.get("p_rally", 1.0 / 3.0), 1.0 / 3.0), 0.0, 1.0)
    p_riskoff = np.clip(_safe_float(signals.get("p_riskoff", 1.0 / 3.0), 1.0 / 3.0), 0.0, 1.0)

    mu20_scale = max(1e-9, _safe_float(config.get("edge_mu20_scale", 0.08), 0.08))
    mu60_scale = max(1e-9, _safe_float(config.get("edge_mu60_scale", 0.20), 0.20))
    downside_scale = max(1e-9, _safe_float(config.get("downside_edge_scale", 0.20), 0.20))
    disagreement_scale = max(1e-9, _safe_float(config.get("disagreement_edge_scale", 0.20), 0.20))
    drawdown_scale = max(1e-9, _safe_float(config.get("drawdown_timing_scale", 0.15), 0.15))
    momentum_scale = max(1e-9, _safe_float(config.get("rally_momentum_scale", 0.08), 0.08))
    volatility_scale = max(1e-9, _safe_float(config.get("volatility_edge_scale", 0.50), 0.50))

    upside_component = (
        _safe_float(config.get("edge_mu20_weight", 0.55), 0.55) * np.tanh(mu20 / mu20_scale)
        + _safe_float(config.get("edge_mu60_weight", 0.45), 0.45) * np.tanh(mu60 / mu60_scale)
    )
    downside_component = (
        _safe_float(config.get("downside_edge_weight", 0.35), 0.35)
        * np.tanh(downside_loss / downside_scale)
    )
    disagreement_component = (
        _safe_float(config.get("disagreement_edge_weight", 0.20), 0.20)
        * np.tanh(disagreement / disagreement_scale)
    )
    correction_timing_component = (
        _safe_float(config.get("correction_timing_weight", 0.20), 0.20)
        * p_correction
        * np.tanh(drawdown / drawdown_scale)
    )
    rally_momentum_component = (
        _safe_float(config.get("rally_momentum_weight", 0.15), 0.15)
        * p_rally
        * np.tanh(max(0.0, momentum) / momentum_scale)
    )
    riskoff_component = _safe_float(config.get("riskoff_edge_weight", 0.25), 0.25) * p_riskoff
    volatility_component = (
        _safe_float(config.get("volatility_edge_weight", 0.15), 0.15)
        * np.tanh(volatility / volatility_scale)
    )

    forecast_edge = float(staleness_scale) * (
        upside_component - downside_component - disagreement_component
    )
    timing_edge = correction_timing_component + rally_momentum_component
    risk_edge = riskoff_component + volatility_component
    edge = forecast_edge + timing_edge - risk_edge
    signed_components = {
        "edge_upside_signed": float(staleness_scale) * upside_component,
        "edge_downside_penalty_signed": -float(staleness_scale) * downside_component,
        "edge_disagreement_penalty_signed": -float(staleness_scale) * disagreement_component,
        "edge_correction_bonus_signed": correction_timing_component,
        "edge_rally_bonus_signed": rally_momentum_component,
        "edge_riskoff_penalty_signed": -riskoff_component,
        "edge_volatility_penalty_signed": -volatility_component,
    }
    signed_sum = float(sum(signed_components.values()))

    return {
        "calibrated_edge": float(edge),
        **signed_components,
        "edge_sum_signed_components": signed_sum,
        "edge_reconciliation_error": float(edge - signed_sum),
        "edge_upside_component": float(upside_component),
        "edge_downside_component": float(downside_component),
        "edge_disagreement_component": float(disagreement_component),
        "edge_correction_timing_component": float(correction_timing_component),
        "edge_rally_momentum_component": float(rally_momentum_component),
        "edge_riskoff_component": float(riskoff_component),
        "edge_volatility_component": float(volatility_component),
    }


def tactical_buy_intensity(edge, config):
    entry_threshold = _safe_float(config.get("entry_threshold", 0.05), 0.05)
    strong_edge = _safe_float(config.get("strong_edge", 0.55), 0.55)
    if strong_edge <= entry_threshold:
        strong_edge = entry_threshold + 1e-9
    approved = edge > entry_threshold
    strength = 0.0
    if approved:
        strength = np.clip((edge - entry_threshold) / (strong_edge - entry_threshold), 0.0, 1.0)
    gamma = max(1e-9, _safe_float(config.get("buy_intensity_gamma", 0.75), 0.75))
    return {
        "entry_threshold": float(entry_threshold),
        "strong_edge": float(strong_edge),
        "buy_approved": bool(approved),
        "buy_strength": float(strength),
        "buy_intensity_gamma": float(gamma),
    }


def soft_time_scale(elapsed_days, invested_total, deployment_budget, config):
    soft_horizon = max(1.0, _safe_float(config.get("soft_horizon_days", 76), 76))
    target_fraction = np.clip(
        _safe_float(config.get("soft_target_fraction", 0.80), 0.80),
        0.0,
        1.0,
    )
    target_progress = target_fraction * min(max(0.0, elapsed_days) / soft_horizon, 1.0)
    actual_progress = (
        max(0.0, invested_total) / max(float(deployment_budget), 1e-9)
    )
    pace_gap = target_progress - actual_progress
    scale = np.exp(_safe_float(config.get("time_strength", 1.5), 1.5) * pace_gap)
    min_scale = _safe_float(config.get("min_time_scale", 0.50), 0.50)
    max_scale = _safe_float(config.get("max_time_scale", 2.00), 2.00)
    if max_scale < min_scale:
        max_scale = min_scale
    return {
        "soft_target_progress": float(target_progress),
        "actual_progress": float(actual_progress),
        "pace_gap": float(pace_gap),
        "time_scale": float(np.clip(scale, min_scale, max_scale)),
    }


def build_decision_rows(feature_df):
    decision_df = feature_df.copy()
    decision_df["decision_price"] = decision_df["close"].shift(-1)
    decision_df["decision_date"] = pd.Series(decision_df.index, index=decision_df.index).shift(-1)
    return decision_df.dropna(subset=["decision_price"]).copy()


def compute_target_weight_buy(theta, cash, shares, price, config, signals):
    cash = max(0.0, float(cash))
    shares = max(0.0, float(shares))
    price = max(0.0, float(price))
    current_stock_value = shares * price
    portfolio_value = cash + current_stock_value
    current_weight = current_stock_value / portfolio_value if portfolio_value > 0 else 0.0
    if portfolio_value <= 0 or cash <= 0 or price <= 0:
        return {
            "daily_investment": 0.0,
            "target_weight": 0.0,
            "desired_stock_value": current_stock_value,
            "current_stock_weight": current_weight,
            "current_stock_value": current_stock_value,
            "portfolio_value": portfolio_value,
            "raw_buy_needed": 0.0,
            "exposure_cap_weight": 0.0,
            "exposure_headroom": 0.0,
            "calibrated_edge": 0.0,
            "entry_threshold": _safe_float(config.get("entry_threshold", 0.05), 0.05),
            "strong_edge": _safe_float(config.get("strong_edge", 0.55), 0.55),
            "buy_approved": False,
            "buy_strength": 0.0,
            "base_buy": 0.0,
            "max_daily_buy": 0.0,
            "buy_limited_by_cap": False,
            "target_daily_fraction": 0.0,
        }

    max_weight = float(config.get("max_target_weight", 1.0))
    details = target_weight_details(theta, signals, max_weight=max_weight)
    forecast_age_days = _safe_float(signals.get("forecast_age_days", 0.0))
    staleness_scale = staleness_scale_for_age(forecast_age_days, config)
    ceiling, risk_ceiling_score, disagreement = risk_ceiling(signals, config, max_weight)
    raw_multiplier, model_confidence = model_pace_multiplier(
        details,
        max_weight,
        config,
    )
    effective_multiplier, signal_multiplier, risk_acceleration_cap = (
        apply_risk_and_staleness_to_multiplier(
            raw_multiplier,
            ceiling,
            max_weight,
            forecast_age_days,
            config,
        )
    )
    weight = details["target_weight"]
    desired_stock_value = weight * portfolio_value
    raw_buy_needed = max(0.0, desired_stock_value - current_stock_value)
    portfolio_cap = portfolio_value * _safe_float(config.get("max_daily_fraction", 0.05), 0.05)
    configured_max_daily_buy = config.get("max_daily_buy")
    absolute_cap = (
        _safe_float(configured_max_daily_buy)
        if configured_max_daily_buy is not None
        else float("inf")
    )

    budget = _safe_float(
        signals.get(
            "deployment_budget",
            config.get("deployment_budget", config.get("initial_cash", cash)),
        )
    )
    if budget <= 0:
        budget = cash + current_stock_value

    elapsed_days = int(_safe_float(signals.get("elapsed_trading_days", 1), 1))
    max_daily_buy = max(0.0, min(portfolio_cap, absolute_cap, cash))
    invested_total = max(0.0, budget - cash)
    edge_details = calibrated_tactical_edge(signals, config, staleness_scale=staleness_scale)
    intensity = tactical_buy_intensity(edge_details["calibrated_edge"], config)
    strength = intensity["buy_strength"]
    base_buy = (
        max_daily_buy * (strength ** intensity["buy_intensity_gamma"])
        if intensity["buy_approved"]
        else 0.0
    )
    min_approved_weight = _safe_float(config.get("min_approved_exposure_weight", 0.35), 0.35)
    approved_floor = min_approved_weight * strength if intensity["buy_approved"] else 0.0
    exposure_cap_weight = float(np.clip(max(weight, approved_floor), 0.0, max_weight))
    exposure_target_value = exposure_cap_weight * portfolio_value
    exposure_headroom = max(0.0, exposure_target_value - current_stock_value)
    model_buy = min(base_buy, exposure_headroom, max_daily_buy, cash)
    time_details = soft_time_scale(elapsed_days, invested_total, budget, config)
    desired_buy = base_buy * time_details["time_scale"]
    buy_amount = min(desired_buy, exposure_headroom, max_daily_buy, cash)

    min_daily = float(config.get("min_daily_investment", 0.0))
    if base_buy > 0.0 and 0.0 < buy_amount < min_daily:
        buy_amount = min(min_daily, cash, max_daily_buy, exposure_headroom)

    return {
        "daily_investment": buy_amount,
        **details,
        "policy_target_weight": details["target_weight"],
        "target_weight": weight,
        "risk_ceiling": ceiling,
        "risk_ceiling_score": risk_ceiling_score,
        "forecast_disagreement": disagreement,
        "staleness_scale": staleness_scale,
        "desired_stock_value": desired_stock_value,
        "exposure_cap_weight": exposure_cap_weight,
        "exposure_target_value": exposure_target_value,
        "exposure_headroom": exposure_headroom,
        "current_stock_weight": current_weight,
        "current_stock_value": current_stock_value,
        "portfolio_value": portfolio_value,
        "raw_buy_needed": raw_buy_needed,
        "portfolio_daily_cap": portfolio_cap,
        "deployment_budget": budget,
        "elapsed_trading_days": elapsed_days,
        "model_confidence": model_confidence,
        "raw_model_multiplier": raw_multiplier,
        "risk_acceleration_cap": risk_acceleration_cap,
        "signal_model_multiplier": signal_multiplier,
        "effective_model_multiplier": effective_multiplier,
        "final_multiplier": effective_multiplier,
        "invested_total_before_buy": invested_total,
        "ramp_days": 0.0,
        "model_buy": model_buy,
        "base_buy": base_buy,
        **edge_details,
        **intensity,
        **time_details,
        "max_daily_buy": max_daily_buy,
        "buy_limited_by_cap": desired_buy > buy_amount,
        "target_daily_fraction": buy_amount / cash if cash > 0 else 0.0,
    }


def _simulate_policy(window_df, theta, initial_cash, config):
    cash = float(initial_cash)
    shares = 0.0
    first_cash = float(initial_cash)
    decision_df = build_decision_rows(window_df)
    if decision_df.empty:
        return 0.0

    soft_penalty_sum = 0.0
    for elapsed_days, (signal_date, row) in enumerate(decision_df.iterrows(), start=1):
        row = row.copy()
        row["elapsed_trading_days"] = elapsed_days
        row["deployment_budget"] = initial_cash
        trained_through = row.get("model_trained_through")
        if pd.notna(trained_through):
            row["forecast_age_days"] = max(
                0,
                int((pd.Timestamp(signal_date) - pd.Timestamp(trained_through)).days),
            )
        elif "forecast_age_days" not in row:
            row["forecast_age_days"] = 0.0
        price = float(row["decision_price"])
        buy = compute_target_weight_buy(theta, cash, shares, price, config, row)
        invest = buy["daily_investment"]
        shares += invest / price if price > 0 else 0.0
        cash -= invest
        target_progress = float(buy.get("soft_target_progress", 0.0))
        actual_progress = (initial_cash - cash) / max(float(initial_cash), 1e-9)
        lag = max(0.0, target_progress - actual_progress)
        soft_penalty_sum += lag * lag

    final_price = float(decision_df["decision_price"].iloc[-1])
    portfolio_value = cash + shares * final_price
    cash_penalty = (cash / first_cash) ** 2 if first_cash > 0 else 0.0
    soft_penalty = soft_penalty_sum / max(1, len(decision_df))
    return (
        -(portfolio_value / first_cash - 1.0)
        + float(config.get("cash_deployment_penalty", 0.0)) * cash_penalty
        + float(config.get("soft_deployment_penalty", 0.02)) * soft_penalty
    )


def optimize_policy_theta(history_df, initial_cash, config, workers=None, return_diagnostics=False):
    workers = int(config.get("optimizer_workers", 1) if workers is None else workers)
    if len(history_df) < int(config.get("min_training_rows", 30)):
        theta = np.zeros(len(TARGET_THETA_BOUNDS), dtype=float)
        diagnostics = {
            "optimizer_fun": None,
            "optimizer_nfev": 0,
            "optimizer_success": False,
            "optimizer_message": "insufficient_history",
        }
        return (theta, diagnostics) if return_diagnostics else theta

    result = differential_evolution(
        lambda theta: _simulate_policy(history_df, theta, initial_cash, config),
        bounds=TARGET_THETA_BOUNDS,
        seed=int(config.get("random_state", 42)),
        polish=False,
        maxiter=int(config.get("optimizer_maxiter", 8)),
        popsize=int(config.get("optimizer_popsize", 5)),
        updating="deferred",
        workers=workers,
    )
    result_fun = getattr(result, "fun", None)
    diagnostics = {
        "optimizer_fun": float(result_fun) if result_fun is not None and np.isfinite(float(result_fun)) else None,
        "optimizer_nfev": int(getattr(result, "nfev", 0)),
        "optimizer_success": bool(getattr(result, "success", False)),
        "optimizer_message": str(getattr(result, "message", "")),
    }
    return (result.x, diagnostics) if return_diagnostics else result.x


# ===== Core daily investment model =====
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
    "buy_policy": "target_weight",
    "forecast_horizons": [20, 60],
    "downside_horizon": 60,
    "max_target_weight": 1.0,
    "max_daily_buy": None,
    "deployment_budget": None,
    "ledger_retrain_every": 1000,
    "live_ledger_min_rows": 250,
    "live_ledger_max_rows": 300,
    "live_ledger_warmup_rows": 40,
    "max_forecast_age_days": 999999,
    "stale_forecast_weight_scale": 0.05,
    "forecast_staleness_half_life_days": 56,
    "risk_ceiling_intercept": 1.25,
    "risk_ceiling_mu20_weight": 6.0,
    "risk_ceiling_mu60_weight": 4.0,
    "risk_ceiling_downside_weight": 8.0,
    "risk_ceiling_disagreement_weight": 4.0,
    "risk_ceiling_riskoff_weight": 1.25,
    "entry_threshold": 0.05,
    "strong_edge": 0.55,
    "buy_intensity_gamma": 0.75,
    "edge_mu20_weight": 0.55,
    "edge_mu60_weight": 0.45,
    "edge_mu20_scale": 0.08,
    "edge_mu60_scale": 0.20,
    "downside_edge_weight": 0.35,
    "downside_edge_scale": 0.20,
    "disagreement_edge_weight": 0.20,
    "disagreement_edge_scale": 0.20,
    "correction_timing_weight": 0.20,
    "drawdown_timing_scale": 0.15,
    "rally_momentum_weight": 0.15,
    "rally_momentum_scale": 0.08,
    "riskoff_edge_weight": 0.25,
    "volatility_edge_weight": 0.15,
    "volatility_edge_scale": 0.50,
    "min_approved_exposure_weight": 0.35,
    "use_kalman_forecast_adjustment": True,
    "kalman_initial_uncertainty": 0.25,
    "kalman_bias_process_variance": 1e-4,
    "kalman_bias_observation_variance": 0.05,
    "min_daily_investment": 0.0,
    "cash_deployment_penalty": 0.0,
    "soft_horizon_days": 76,
    "soft_target_fraction": 0.80,
    "time_strength": 1.5,
    "min_time_scale": 0.50,
    "max_time_scale": 2.00,
    "rolling_opt_window": 60,
    "optimizer_workers": 1,
    "n_estimators": 5,
    "model_n_jobs": 1,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "min_training_rows": 30,
    "random_state": 42,
    "optimizer_maxiter": 0,
    "optimizer_popsize": 1,
}


def normalize_config(config):
    merged = DEFAULT_CONFIG.copy()
    merged.update(config or {})
    if "initial_cash" in merged:
        merged["initial_cash"] = float(merged["initial_cash"])
    if "initial_shares" in merged:
        merged["initial_shares"] = float(merged["initial_shares"])
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


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


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
    model_buy = min(cash * daily_fraction, cash)

    deployment_budget = max(_safe_float(config.get("deployment_budget", config.get("initial_cash", cash))), 1e-9)
    invested_total = max(0.0, _safe_float(config.get("invested_total_before_buy", 0.0)))
    elapsed_days = max(0.0, _safe_float(config.get("elapsed_plan_days", 1.0), 1.0))
    soft_horizon = max(1.0, _safe_float(config.get("soft_horizon_days", 76.0), 76.0))
    soft_target_fraction = float(np.clip(_safe_float(config.get("soft_target_fraction", 0.80), 0.80), 0.0, 1.0))
    target_progress = soft_target_fraction * min(elapsed_days / soft_horizon, 1.0)
    actual_progress = invested_total / deployment_budget
    pace_gap = target_progress - actual_progress
    raw_time_scale = float(np.exp(_safe_float(config.get("time_strength", 1.5), 1.5) * pace_gap))
    min_scale = _safe_float(config.get("min_time_scale", 0.50), 0.50)
    max_scale = _safe_float(config.get("max_time_scale", 2.00), 2.00)
    if max_scale < min_scale:
        max_scale = min_scale
    time_scale = float(np.clip(raw_time_scale, min_scale, max_scale))

    invest = min(model_buy * time_scale, cash)
    if model_buy > 0.0 and 0.0 < invest < config["min_daily_investment"]:
        invest = min(config["min_daily_investment"], cash)

    return {
        "daily_investment": invest,
        "target_daily_fraction": daily_fraction,
        "model_buy": model_buy,
        "time_scale": time_scale,
        "soft_target_progress": float(target_progress),
        "actual_progress": float(actual_progress),
        "pace_gap": float(pace_gap),
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

    min_training_rows = int(config.get("min_training_rows", 30))
    if len(training_df) < min_training_rows:
        raise ValueError(
            f"Not enough usable training rows for this ticker/date range. "
            f"Need at least {min_training_rows}, found {len(training_df)}."
        )

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


def prepare_feature_predictions(ticker, config, data_dirs=None):
    dataset_path = find_dataset(ticker, data_dirs=data_dirs)

    if dataset_path is None:
        raise FileNotFoundError(f"No dataset found for {ticker}.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    model = train_prediction_model(feature_df, config)
    feature_df["predicted_return"] = model.predict(feature_df[FEATURES])
    return feature_df


def compute_today_investment(ticker, config, data_dirs=None):
    config = normalize_config(config)
    feature_df = prepare_feature_predictions(ticker, config, data_dirs=data_dirs)

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


def compute_today_target_weight_investment(ticker, config, data_dirs=None):
    config = normalize_config(config)
    today_row = get_live_prediction_snapshot(ticker, config, data_dirs=data_dirs)
    today_date = today_row.name
    price = float(today_row["close"])
    cash = max(0.0, float(config.get("initial_cash", 0.0)))
    shares = max(0.0, float(config.get("initial_shares", 0.0)))

    prediction_frame = prepare_prediction_frame(
        ticker,
        config,
        data_dirs=data_dirs,
        live=True,
    )
    theta_history_df = prediction_frame.iloc[:-1].copy()
    if not theta_history_df.empty and "maturity_date" in theta_history_df:
        maturity_dates = pd.to_datetime(theta_history_df["maturity_date"])
        theta_history_df = theta_history_df.loc[
            theta_history_df["is_mature"].astype(bool)
            & maturity_dates.notna()
            & (maturity_dates <= today_date)
        ]
    theta_history_df = theta_history_df.tail(int(config.get("rolling_opt_window", 120)))

    theta_diagnostics = {
        "optimizer_fun": None,
        "optimizer_nfev": 0,
        "optimizer_success": False,
        "optimizer_message": "insufficient_history",
    }
    if len(theta_history_df) >= int(config.get("min_training_rows", 30)):
        theta_t, theta_diagnostics = optimize_policy_theta(
            theta_history_df,
            float(config.get("deployment_budget", config.get("initial_cash", cash))),
            config,
            return_diagnostics=True,
        )
    else:
        theta_t = np.zeros(8, dtype=float)

    model_trained_through = today_row.get("model_trained_through")
    forecast_age_days = (
        int((pd.Timestamp(today_date) - pd.Timestamp(model_trained_through)).days)
        if pd.notna(model_trained_through)
        else 0
    )
    today_row["forecast_age_days"] = forecast_age_days
    today_row["elapsed_trading_days"] = int(config.get("elapsed_plan_days", 1))
    today_row["deployment_budget"] = float(
        config.get("deployment_budget", config.get("initial_cash", cash))
    )

    buy = compute_target_weight_buy(theta_t, cash, shares, price, config, today_row)
    daily_investment = float(buy["daily_investment"])

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
        "predicted_return": float(today_row["predicted_return_60d"]),
        "predicted_return_20d": float(today_row["predicted_return_20d"]),
        "predicted_return_60d": float(today_row["predicted_return_60d"]),
        "predicted_downside_60d": float(today_row["predicted_downside_60d"]),
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
        "forecast_age_days": forecast_age_days,
        "theta_g0": float(theta_t[0]),
        "theta_drawdown": float(theta_t[4]),
        "theta_predicted_return": float(theta_t[2]),
        "theta_volatility": 0.0,
        "theta_mu20": float(theta_t[1]),
        "theta_mu60": float(theta_t[2]),
        "theta_downside": float(theta_t[3]),
        "theta_drawdown_correction": float(theta_t[5]),
        "theta_drawdown_riskoff": float(theta_t[6]),
        "theta_momentum_rally": float(theta_t[7]),
        "theta_optimizer_success": bool(theta_diagnostics["optimizer_success"]),
        "theta_optimizer_message": str(theta_diagnostics["optimizer_message"]),
        "theta_optimizer_nfev": int(theta_diagnostics["optimizer_nfev"]),
        **buy,
    }
