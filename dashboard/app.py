import streamlit as st
import requests
import pandas as pd

# URLs (replace with your actual ones)
GITHUB_ACTION_URL = "https://api.github.com/repos/jashankaamboj/ml-pipeline/actions/workflows/train.yml/dispatches"
GITHUB_TOKEN = "ghp_xxx"  # Replace with your GitHub token
RENDER_DEPLOY_HOOK = "https://api.render.com/deploy/your-hook-id"  # Your actual hook
API_URL = "https://ml-pipeline-1-dhl.onrender.com/predict"  # Your Render API endpoint

# Set page
st.set_page_config(page_title="ML Pipeline Dashboard", layout="centered")

st.title("🧠 ML Model Dashboard")

# Upload dataset
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df)

# Trigger Training
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
        st.success("Training triggered via GitHub Actions!")
    else:
        st.error(f"Failed to trigger training: {response.text}")

# Trigger Deploy
st.header("🚀 Deploy Model")
if st.button("Trigger Render Deployment"):
    response = requests.post(RENDER_DEPLOY_HOOK)
    if response.status_code == 200:
        st.success("Deployment triggered!")
    else:
        st.error(f"Failed to deploy: {response.text}")

# Predict
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
            st.success(f"Predicted Price: ₹{round(result['predicted_price'], 2)}")
        else:
            st.error(f"API Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Request failed: {e}")
