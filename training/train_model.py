import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model(
    csv_path='data/housing.csv',
    model_path='model/model.pkl',
    metrics_path='metrics/metrics.txt'
):
    # Load dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("❌ Dataset is empty after dropping missing values.")

    if not all(col in df.columns for col in ['area', 'bedrooms', 'age', 'price']):
        raise KeyError("❌ Dataset must contain 'area', 'bedrooms', 'age', and 'price' columns.")

    X = df[['area', 'bedrooms', 'age']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y, predictions)

    # Custom Accuracy (e.g., percentage of predictions within 20% of actual value)
    threshold = 0.20
    accuracy = (abs(predictions - y) / y < threshold).mean() * 100

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"{mae},{rmse},{mse},{r2},{accuracy}")
    print(f"📊 Metrics saved | MAE: {mae:.2f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train_model()
