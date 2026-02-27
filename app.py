import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD TRAINED MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("yield_model.pkl")

model = load_model()

# ===============================
# APP TITLE
# ===============================
st.title("🌾 Crop Yield Prediction App")

st.write("Enter the details below to predict crop yield (q/acre).")

# ===============================
# USER INPUTS
# ===============================
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=800.0)
temperature = st.number_input("Average Temperature (°C)", min_value=0.0, value=27.0)
base_yield = st.number_input("Base Yield (q/acre)", min_value=0.0, value=20.0)
fertility = st.number_input("Fertility Index", min_value=0.0, value=1.1)
efficiency = st.number_input("Efficiency Index", min_value=0.0, value=1.0)

crop = st.selectbox("Crop Name", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"])
soil = st.selectbox("Soil Type", ["Black", "Red", "Alluvial", "Laterite"])
irrigation = st.selectbox("Irrigation Type", ["Canal", "Tube Well", "Rainfed"])
season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
state = st.selectbox("State", ["Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka"])

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("Predict Yield"):
    input_data = pd.DataFrame({
        "Rainfall_mm": [rainfall],
        "Avg_Temperature_C": [temperature],
        "Base_Yield_q_per_acre": [base_yield],
        "Fertility_Index": [fertility],
        "Efficiency_Index": [efficiency],
        "Crop_Name": [crop],
        "Soil_Type": [soil],
        "Irrigation_Type": [irrigation],
        "Season": [season],
        "State": [state]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"🌱 Predicted Yield: {prediction:.2f} q/acre")
