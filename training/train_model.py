import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(csv_path='data/housing.csv', model_path='model/model.pkl'):
    # Step 1: Load dataset
    df = pd.read_csv(csv_path)

    # Step 2: Clean data (drop rows with missing values)
    df.dropna(inplace=True)

    # Step 3: Split features and label
    X = df[['area', 'bedrooms', 'age']]
    y = df['price']

    # Step 4: Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Step 5: Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()
