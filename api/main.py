from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/model.pkl')

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
    print(f"❌ Model not found at {MODEL_PATH}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Prediction API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    try:
        area = data["area"]
        bedrooms = data["bedrooms"]
        age = data["age"]
        features = np.array([[area, bedrooms, age]])
        price = model.predict(features)[0]
        return jsonify({"predicted_price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
