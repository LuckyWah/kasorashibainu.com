

# ===== Market regime helpers =====
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REGIME_COLUMNS = ["p_rally", "p_correction", "p_riskoff"]
REGIME_INPUT_COLUMNS = [
    "spy_return_20d",
    "spy_return_60d",
    "spy_drawdown",
    "vix",
    "market_realized_volatility",
    "treasury_10y_change",
]


def _neutral_probabilities(index):
    return pd.DataFrame(
        {
            "p_rally": np.full(len(index), 1.0 / 3.0),
            "p_correction": np.full(len(index), 1.0 / 3.0),
            "p_riskoff": np.full(len(index), 1.0 / 3.0),
        },
        index=index,
    )


def _label_components(model, scaler, columns):
    centers = pd.DataFrame(
        scaler.inverse_transform(model.means_),
        columns=columns,
    )
    stress = (
        centers.get("spy_drawdown", 0.0)
        + centers.get("market_realized_volatility", 0.0)
        + centers.get("vix", 0.0) / 100.0
        - centers.get("spy_return_60d", 0.0)
    )
    momentum = centers.get("spy_return_20d", 0.0) + centers.get("spy_return_60d", 0.0)

    riskoff_component = int(stress.idxmax())
    rally_component = int(momentum.drop(index=riskoff_component, errors="ignore").idxmax())
    correction_component = next(
        component
        for component in range(len(centers))
        if component not in {riskoff_component, rally_component}
    )
    return {
        rally_component: "p_rally",
        correction_component: "p_correction",
        riskoff_component: "p_riskoff",
    }
def fit_rolling_hmm(
    market_df,
    window=252,
    min_rows=90,
    random_state=42,
    transition_smoothing=0.15,
    refit_every=20,
):
    """Estimate rolling 3-state market-regime probabilities.

    The app does not ship an HMM dependency. This uses a rolling Gaussian mixture
    with light probability smoothing, which preserves the intended API and
    avoids introducing a new runtime install requirement.
    """
    if market_df.empty:
        return _neutral_probabilities(market_df.index)

    columns = [column for column in REGIME_INPUT_COLUMNS if column in market_df.columns]
    if len(columns) < 3:
        return _neutral_probabilities(market_df.index)

    clean = market_df[columns].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    probabilities = _neutral_probabilities(clean.index)
    previous = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    model = None
    scaler = None
    component_map = None

    for position, current_date in enumerate(clean.index):
        start = max(0, position - int(window) + 1)
        history = clean.iloc[start : position + 1].dropna()
        if len(history) < int(min_rows):
            probabilities.loc[current_date] = previous
            continue

        try:
            should_refit = model is None or position % int(refit_every) == 0
            if should_refit:
                scaler = StandardScaler()
                X = scaler.fit_transform(history)
                model = GaussianMixture(
                    n_components=3,
                    covariance_type="full",
                    random_state=random_state,
                    reg_covar=1e-5,
                    n_init=1,
                    max_iter=100,
                )
                model.fit(X)
                component_map = _label_components(model, scaler, columns)
            raw = model.predict_proba(scaler.transform(clean.iloc[[position]]))[0]
            mapped = np.zeros(3, dtype=float)
            for component, column in component_map.items():
                mapped[REGIME_COLUMNS.index(column)] = raw[component]
        except Exception:
            mapped = previous.copy()

        smoothed = (1.0 - transition_smoothing) * mapped + transition_smoothing * previous
        total = smoothed.sum()
        previous = smoothed / total if total > 0 else previous
        probabilities.loc[current_date] = previous

    return probabilities


# ===== Feature engineering helpers =====
import numpy as np
import pandas as pd



BASE_FEATURES = [
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

KALMAN_FEATURES = [
    "kalman_residual_z",
    "kalman_slope",
    "kalman_uncertainty",
    "volatility_ratio",
]

REGIME_FEATURES = [
    "spy_return_20d",
    "spy_return_60d",
    "market_realized_volatility",
    "treasury_10y_change",
    *REGIME_COLUMNS,
]

UPGRADED_FEATURES = BASE_FEATURES + KALMAN_FEATURES + REGIME_FEATURES


def add_kalman_features(feature_df, process_variance=1e-5, observation_variance=1e-3):
    df = feature_df.copy()
    close = df["close"].astype(float).replace(0, np.nan).ffill()
    log_price = np.log(close)

    level = np.zeros(len(df), dtype=float)
    uncertainty = np.zeros(len(df), dtype=float)
    if len(df) == 0:
        return df

    level[0] = float(log_price.iloc[0])
    uncertainty[0] = 1.0
    for idx in range(1, len(df)):
        prior_level = level[idx - 1]
        prior_uncertainty = uncertainty[idx - 1] + process_variance
        innovation = float(log_price.iloc[idx]) - prior_level
        innovation_var = prior_uncertainty + observation_variance
        gain = prior_uncertainty / innovation_var
        level[idx] = prior_level + gain * innovation
        uncertainty[idx] = (1.0 - gain) * prior_uncertainty

    level_series = pd.Series(level, index=df.index)
    residual = log_price - level_series
    residual_scale = residual.rolling(60, min_periods=20).std().replace(0, np.nan)
    long_vol = df["return"].astype(float).rolling(252, min_periods=60).std() * np.sqrt(252)

    df["kalman_residual_z"] = (residual / residual_scale).replace([np.inf, -np.inf], np.nan)
    df["kalman_slope"] = level_series.diff(20)
    df["kalman_uncertainty"] = uncertainty
    df["volatility_ratio"] = (
        df["volatility_20d"].astype(float) / long_vol.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    return df


def add_market_regime_inputs(feature_df):
    df = feature_df.copy()
    spy_proxy = (1.0 + df["spy_return"].astype(float).fillna(0.0)).cumprod()
    df["spy_return_20d"] = spy_proxy.pct_change(20, fill_method=None)
    df["spy_return_60d"] = spy_proxy.pct_change(60, fill_method=None)
    df["market_realized_volatility"] = (
        df["spy_return"].astype(float).rolling(20, min_periods=10).std() * np.sqrt(252)
    )
    df["treasury_10y_change"] = df["treasury_10y"].astype(float).diff(20)
    return df


def build_upgraded_feature_frame(feature_df, regime_window=252):
    df = add_market_regime_inputs(add_kalman_features(feature_df))
    regime_probabilities = fit_rolling_hmm(df, window=regime_window)
    for column in REGIME_COLUMNS:
        df[column] = regime_probabilities[column]

    df[UPGRADED_FEATURES] = df[UPGRADED_FEATURES].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=UPGRADED_FEATURES).copy()


# ===== Forecast helpers =====
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = ["future_return_20d", "future_return_60d", "future_downside_60d"]


def add_medium_horizon_targets(feature_df):
    df = feature_df.copy()
    close = df["close"].astype(float)
    df["future_return_20d"] = close.shift(-20) / close - 1.0
    df["future_return_60d"] = close.shift(-60) / close - 1.0

    forward_returns = pd.concat(
        [close.shift(-offset) / close - 1.0 for offset in range(1, 61)],
        axis=1,
    )
    df["future_downside_60d"] = forward_returns.min(axis=1)
    return df


def _model_for(kind, target, config):
    random_state = int(config.get("random_state", 42))
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if target == "future_downside_60d":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.10,
                max_iter=int(config.get("gb_max_iter", 120)),
                max_leaf_nodes=15,
                learning_rate=0.05,
                random_state=random_state,
            ),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 100)),
            max_depth=config.get("max_depth", 8),
            min_samples_leaf=int(config.get("min_samples_leaf", 5)),
            random_state=random_state,
            n_jobs=int(config.get("model_n_jobs", -1)),
        ),
    )


def fit_forecast_models(training_df, feature_columns, config=None, model_kind="rf"):
    config = config or {}
    if not set(TARGET_COLUMNS).issubset(training_df.columns):
        training_df = add_medium_horizon_targets(training_df)
    models = {}
    for target in TARGET_COLUMNS:
        model_df = training_df.dropna(subset=[target, *feature_columns]).copy()
        min_training_rows = int(config.get("min_training_rows", 30))
        if len(model_df) < min_training_rows:
            raise ValueError(
                f"Not enough rows to train {target}. Need {min_training_rows}, found {len(model_df)}."
            )
        model = _model_for(model_kind, target, config)
        model.fit(model_df[feature_columns], model_df[target])
        models[target] = model
    return models


def predict_forecasts(models, row, feature_columns):
    X = row[feature_columns].to_frame().T
    return {
        "predicted_return_20d": float(models["future_return_20d"].predict(X)[0]),
        "predicted_return_60d": float(models["future_return_60d"].predict(X)[0]),
        "predicted_downside_60d": float(models["future_downside_60d"].predict(X)[0]),
    }


# ===== Point-in-time forecast ledger =====
import pandas as pd



def _initial_kalman_state(config):
    initial_uncertainty = float(config.get("kalman_initial_uncertainty", 0.25))
    return {
        "future_return_20d": {"bias": 0.0, "uncertainty": initial_uncertainty, "updates": 0},
        "future_return_60d": {"bias": 0.0, "uncertainty": initial_uncertainty, "updates": 0},
    }


def _kalman_update(state, observed_error, config):
    process_variance = float(config.get("kalman_bias_process_variance", 1e-4))
    observation_variance = float(config.get("kalman_bias_observation_variance", 0.05))
    prior_uncertainty = state["uncertainty"] + process_variance
    gain = prior_uncertainty / max(prior_uncertainty + observation_variance, 1e-12)
    state["bias"] = state["bias"] + gain * (float(observed_error) - state["bias"])
    state["uncertainty"] = (1.0 - gain) * prior_uncertainty
    state["updates"] += 1
    return float(gain)


def _apply_kalman_adjustment(raw_predictions, kalman_state, config):
    if not bool(config.get("use_kalman_forecast_adjustment", True)):
        return {
            "predicted_return_20d": raw_predictions["predicted_return_20d"],
            "predicted_return_60d": raw_predictions["predicted_return_60d"],
            "predicted_downside_60d": raw_predictions["predicted_downside_60d"],
            "kalman_gain_20d": 0.0,
            "kalman_gain_60d": 0.0,
        }

    return {
        "predicted_return_20d": (
            raw_predictions["predicted_return_20d"]
            + kalman_state["future_return_20d"]["bias"]
        ),
        "predicted_return_60d": (
            raw_predictions["predicted_return_60d"]
            + kalman_state["future_return_60d"]["bias"]
        ),
        "predicted_downside_60d": raw_predictions["predicted_downside_60d"],
        "kalman_gain_20d": float(kalman_state["future_return_20d"].get("last_gain", 0.0)),
        "kalman_gain_60d": float(kalman_state["future_return_60d"].get("last_gain", 0.0)),
    }


def generate_point_in_time_predictions(
    feature_df,
    feature_columns,
    config=None,
    min_train_rows=None,
    max_horizon=60,
    model_kind="rf",
    retrain_every=20,
):
    """Create a strict out-of-sample prediction ledger.

    For date t, the training set is limited to rows whose 60-day outcome had
    already matured before t. Realized outcomes are included in the ledger only
    when they are known.
    """
    config = config or {}
    min_train_rows = int(min_train_rows or config.get("min_training_rows", 30))
    retrain_every = max(1, int(retrain_every))

    target_df = add_medium_horizon_targets(feature_df).copy()
    rows = []
    models = None
    model_trained_through = None
    kalman_state = _initial_kalman_state(config)
    kalman_updated_labels = set()

    for position, (prediction_date, row) in enumerate(target_df.iterrows()):
        train_end_position = position - int(max_horizon)
        if train_end_position < min_train_rows:
            continue

        for row_index, ledger_row in enumerate(rows):
            for target, horizon in [
                ("future_return_20d", "20d"),
                ("future_return_60d", "60d"),
            ]:
                update_key = (row_index, horizon)
                if update_key in kalman_updated_labels:
                    continue
                maturity_date = ledger_row.get(f"maturity_date_{horizon}")
                if pd.isna(maturity_date) or pd.Timestamp(maturity_date) > pd.Timestamp(prediction_date):
                    continue
                actual_key = f"actual_return_{horizon}"
                raw_key = f"raw_base_return_{horizon}"
                if actual_key not in ledger_row or raw_key not in ledger_row:
                    continue
                observed_error = ledger_row[actual_key] - ledger_row[raw_key]
                gain = _kalman_update(kalman_state[target], observed_error, config)
                kalman_state[target]["last_gain"] = gain
                kalman_updated_labels.add(update_key)

        if models is None or len(rows) % retrain_every == 0:
            train_df = target_df.iloc[:train_end_position].dropna(subset=feature_columns)
            if len(train_df) < min_train_rows:
                continue
            models = fit_forecast_models(
                train_df,
                feature_columns,
                config=config,
                model_kind=model_kind,
            )
            model_trained_through = train_df.index[-1]

        raw_predictions = predict_forecasts(models, row, feature_columns)
        predictions = _apply_kalman_adjustment(raw_predictions, kalman_state, config)
        maturity_position = position + int(max_horizon)
        maturity_20_position = position + 20
        maturity_60_position = position + 60
        mature = maturity_position < len(target_df)
        mature_20d = maturity_20_position < len(target_df)
        mature_60d = maturity_60_position < len(target_df)

        ledger_row = {
            "date": prediction_date,
            "model_trained_through": model_trained_through,
            "maturity_date": target_df.index[maturity_position] if mature else pd.NaT,
            "maturity_date_20d": target_df.index[maturity_20_position] if mature_20d else pd.NaT,
            "maturity_date_60d": target_df.index[maturity_60_position] if mature_60d else pd.NaT,
            "is_mature": bool(mature),
            "raw_base_return_20d": raw_predictions["predicted_return_20d"],
            "raw_base_return_60d": raw_predictions["predicted_return_60d"],
            "kalman_adjusted_return_20d": predictions["predicted_return_20d"],
            "kalman_adjusted_return_60d": predictions["predicted_return_60d"],
            "kalman_bias_20d": kalman_state["future_return_20d"]["bias"],
            "kalman_bias_60d": kalman_state["future_return_60d"]["bias"],
            "kalman_gain_20d": predictions["kalman_gain_20d"],
            "kalman_gain_60d": predictions["kalman_gain_60d"],
            "kalman_update_count_20d": kalman_state["future_return_20d"]["updates"],
            "kalman_update_count_60d": kalman_state["future_return_60d"]["updates"],
            **predictions,
        }
        if mature:
            ledger_row.update(
                {
                    "actual_return_60d": row["future_return_60d"],
                    "actual_downside_60d": row["future_downside_60d"],
                }
            )
        if mature_20d:
            ledger_row["actual_return_20d"] = row["future_return_20d"]
        rows.append(ledger_row)

    return pd.DataFrame(rows)


# ===== Buy allocation helpers =====

# ===== Shared dataset/frame helpers =====
from pathlib import Path


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


def prepare_prediction_frame(ticker, config, data_dirs=None, live=False):
    """
    Load the dataset, build upgraded features, generate/join the strict
    point-in-time ledger, and return the usable prediction frame.
    """
    config = config or {}
    dataset_path = find_dataset(ticker, data_dirs=data_dirs)

    if dataset_path is None:
        raise FileNotFoundError(f"No dataset found for {ticker}.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_df = df.dropna(subset=BASE_FEATURES).copy()

    if feature_df.empty:
        raise ValueError("Dataset has no usable feature rows.")

    if live:
        live_min_rows = int(config.get("live_ledger_min_rows", 250) or 0)
        live_max_rows = int(config.get("live_ledger_max_rows", 300) or 0)
        legacy_lookback_rows = int(config.get("live_ledger_lookback_rows", 0) or 0)
        if live_max_rows <= 0 and legacy_lookback_rows > 0:
            live_max_rows = legacy_lookback_rows

        if live_min_rows > 0 or live_max_rows > 0:
            required_rows = (
                int(config.get("rolling_opt_window", 120))
                + int(config.get("downside_horizon", 60))
                + int(config.get("min_training_rows", 30))
                + int(config.get("live_ledger_warmup_rows", 40))
            )
            target_rows = max(live_min_rows, required_rows)
            if live_max_rows > 0:
                target_rows = min(target_rows, live_max_rows)
            feature_df = feature_df.tail(target_rows).copy()

    feature_df = build_upgraded_feature_frame(feature_df)
    ledger_df = generate_point_in_time_predictions(
        feature_df,
        UPGRADED_FEATURES,
        config=config,
        max_horizon=int(config.get("downside_horizon", 60)),
        model_kind=config.get("forecast_model_kind", "rf"),
        retrain_every=int(config.get("ledger_retrain_every", 60)),
    )
    if ledger_df.empty:
        raise ValueError("Not enough history to build an out-of-sample forecast ledger.")

    ledger_df = ledger_df.set_index("date")
    ledger_columns = [column for column in ledger_df.columns if column not in feature_df.columns]
    feature_df = feature_df.join(ledger_df[ledger_columns], how="left")
    feature_df = feature_df.dropna(
        subset=["predicted_return_20d", "predicted_return_60d", "predicted_downside_60d"]
    )
    if feature_df.empty:
        raise ValueError("No usable point-in-time predictions are available yet.")

    return feature_df


def get_live_prediction_snapshot(ticker, config, data_dirs=None):
    frame = prepare_prediction_frame(ticker, config, data_dirs=data_dirs, live=True)
    return frame.iloc[-1].copy()
