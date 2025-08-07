import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# URLs (replace with your actual ones)
GITHUB_ACTION_URL = "https://api.github.com/repos/jashankaamboj/ml-pipeline/actions/workflows/train.yml/dispatches"
RENDER_DEPLOY_HOOK = "https://api.render.com/deploy/srv-d2afmn63jp1c73ajl8g0?key=2ex5OnoJkTk"
API_URL = "https://ml-pipeline-1-dhl1.onrender.com/predict"

# Set Streamlit page
st.set_page_config(page_title="ML Pipeline Dashboard", layout="centered")
st.title("🧠 ML Model Dashboard")

# ------------------ Upload Dataset ------------------
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    # Preview uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df)

    # Save uploaded dataset with timestamp
    os.makedirs("data/versions", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"data/versions/dataset_{timestamp}.csv"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Dataset saved to: `{save_path}`")

# ------------------ Trigger GitHub Action ------------------
st.header("⚙️ Train Model")
if st.button("Trigger GitHub Action for Training"):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "ref": "main"
    }
    response = requests.post(GITHUB_ACTION_URL, json=payload, headers=headers)
    if response.status_code == 204:
        st.success("✅ Training triggered via GitHub Actions!")
    else:
        st.error(f"❌ Failed to trigger training: {response.text}")

# ------------------ Trigger Deployment ------------------
st.header("🚀 Deploy Model")
if st.button("Trigger Render Deployment"):
    response = requests.post(RENDER_DEPLOY_HOOK)
    if response.status_code in [200, 202]:  # Accept both 200 and 202
        st.success("✅ Deployment triggered! Check Render dashboard for progress.")
    else:
        st.error(f"❌ Failed to deploy: {response.text}")

# ------------------ Show Model Metrics ------------------
st.header("📊 Model Performance")
try:
    with open("metrics/metrics.txt", "r") as f:
        mae, rmse, mse, r2, accuracy = f.read().strip().split(",")
        st.metric("MAE", round(float(mae), 2))
        st.metric("RMSE", round(float(rmse), 2))
        st.metric("MSE", round(float(mse), 2))
        st.metric("R² Score", round(float(r2), 4))
        st.metric("Accuracy (%)", f"{float(accuracy):.2f}%")
except Exception as e:
    st.warning("⚠️ Metrics not available. Please train the model.")

# ------------------ Make Prediction ------------------
st.header("🔮 Make a Prediction")
area = st.number_input("Area (sq ft)", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
age = st.number_input("Age of Property (years)", min_value=0)

if st.button("Predict Price"):
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "age": age
    }
    try:
        res = requests.post(API_URL, json=input_data)
        result = res.json()
        if "predicted_price" in result:
            st.success(f"💰 Predicted Price: ₹{round(result['predicted_price'], 2)}")
        else:
            st.error(f"API Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Request failed: {e}")
