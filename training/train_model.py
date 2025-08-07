import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def train_model(
    csv_path='data/housing.csv',
    model_path='model/model.pkl',
    metrics_path='metrics/metrics.txt'
):
    # Step 1: Load dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("❌ Dataset is empty after dropping missing values.")

    # Step 2: Features and target
    if not all(col in df.columns for col in ['area', 'bedrooms', 'age', 'price']):
        raise KeyError("❌ Dataset must contain 'area', 'bedrooms', 'age', and 'price' columns.")

    X = df[['area', 'bedrooms', 'age']]
    y = df['price']

    # Step 3: Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Step 4: Predict and calculate metrics
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions, squared=False)

    # Step 5: Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    # Step 6: Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"{mae},{rmse}")
    print(f"📊 Metrics saved to: {metrics_path} | MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_model()
