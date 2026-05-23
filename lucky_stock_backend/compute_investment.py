import numpy as np


SELL_BOUNDS = [
    (0.0000, 0.0300),
    (0.0000, 0.0600),
    (0.0000, 0.0600),
    (0.0000, 8.0000),
    (0.0000, 0.0800),
    (0.0000, 0.0600),
    (0.0000, 0.0600),
]


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


def compute_daily_sell(theta, shares, config, row):
    shares = max(0.0, float(shares))
    if shares <= 0:
        return _empty_sale(config)

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


def _empty_sale(config):
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
