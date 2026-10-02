import streamlit as st
import numpy as np
import joblib

# Page title & icon
st.set_page_config(page_title="Smart Car Price Predictor", page_icon="🚗")

st.title("🚗 Smart Car Price Predictor")
st.write("Apni car ki details enter karke estimated price jaanein.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

try:
    model = load_model()
except:
    st.error("Model file nahi mili! Pehle train_model.py run karke model save karein.")
    st.stop()

# User Inputs
year = st.number_input("Purchase Year", min_value=2000, max_value=2026, value=2018)
present_price = st.number_input("Current Showroom Price (Lakhs ₹)", min_value=0.5, value=5.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=25000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Encoding Inputs
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0
trans_manual = 1 if transmission == "Manual" else 0

# Prediction
if st.button("Predict Price 💰"):
    input_features = np.array([[year, present_price, kms_driven, fuel_diesel, fuel_petrol, trans_manual]])
    predicted_price = model.predict(input_features)[0]
    st.success(f"Estimated Selling Price: **₹ {round(predicted_price, 2)} Lakhs**")
  
