from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Settings
# ======================
prediction_days = 1
test_days = 60

df = pd.read_csv(
    "IBM_adaptive_investing_dataset.csv",
    index_col=0,
    parse_dates=True
)

features = [
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
    "cpi_yoy"
]

# ======================
# Target
# ======================
target = f"future_return_{prediction_days}d"
actual_price_col = f"actual_{prediction_days}d_price"
predicted_price_col = f"predicted_{prediction_days}d_price"

df[target] = df["close"].shift(-prediction_days) / df["close"] - 1
df[actual_price_col] = df["close"].shift(-prediction_days)

# Latest rows only need features
latest_df = df.dropna(subset=features).copy()

# Training rows need both features and target
train_df = df.dropna(subset=features + [target]).copy()

X = train_df[features]
y = train_df[target]

# ======================
# Train/test split
# ======================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ======================
# Model
# ======================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# ======================
# Latest 60-day prediction
# ======================
test_df = latest_df.iloc[-test_days:].copy()

X_test = test_df[features]
predicted_return = model.predict(X_test)

test_df[predicted_price_col] = test_df["close"] * (1 + predicted_return)
test_df[actual_price_col] = test_df["close"].shift(-prediction_days)

# MSE only where actual future price exists
eval_df = test_df.dropna(subset=[actual_price_col, predicted_price_col]).copy()

mse_price = mean_squared_error(
    eval_df[actual_price_col],
    eval_df[predicted_price_col]
)

print(f"{prediction_days}-day price MSE:", mse_price)

# ======================
# Latest prediction
# ======================
latest_x = latest_df[features].iloc[[-1]]
latest_close = latest_df["close"].iloc[-1]
latest_predicted_return = model.predict(latest_x)[0]
latest_predicted_price = latest_close * (1 + latest_predicted_return)

print("\nLatest close:", latest_close)
print(f"Predicted {prediction_days}-day return:", latest_predicted_return)
print(f"Predicted {prediction_days}-day price:", latest_predicted_price)

# ======================
# Feature importance
# ======================
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# ======================
# Plot
# ======================
plot_df = eval_df.copy()

plt.figure(figsize=(12, 6))

plt.plot(
    plot_df.index,
    plot_df[actual_price_col],
    label=f"Actual {prediction_days}-Day Future Price"
)

plt.plot(
    plot_df.index,
    plot_df[predicted_price_col],
    label=f"Predicted {prediction_days}-Day Future Price"
)

plt.legend()

plt.title(
    f"Actual vs Predicted {prediction_days}-Day Future Price (Last {test_days} Days)"
)

plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()