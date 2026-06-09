import streamlit as st
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Disaster Prediction App")
st.title("🌍 Disaster Prediction App")

# Inputs
disaster_type = st.number_input("Disaster Type (Encoded)", min_value=0, value=0)
location = st.number_input("Location (Encoded)", min_value=0, value=0)

latitude = st.number_input("Latitude", value=17.3850)
longitude = st.number_input("Longitude", value=78.4867)

severity_level = st.slider("Severity Level", 1, 10, 5)

affected_population = st.number_input(
    "Affected Population",
    min_value=0,
    value=1000
)

estimated_economic_loss_usd = st.number_input(
    "Estimated Economic Loss (USD)",
    min_value=0.0,
    value=100000.0
)

response_time_hours = st.number_input(
    "Response Time (Hours)",
    min_value=0.0,
    value=24.0
)

aid_provided = st.number_input(
    "Aid Provided (Encoded)",
    min_value=0,
    value=0
)

infrastructure_damage_index = st.slider(
    "Infrastructure Damage Index",
    0,
    100,
    50
)

if st.button("Predict"):

    data = pd.DataFrame({
        "disaster_type": [disaster_type],
        "location": [location],
        "latitude": [latitude],
        "longitude": [longitude],
        "severity_level": [severity_level],
        "affected_population": [affected_population],
        "estimated_economic_loss_usd": [estimated_economic_loss_usd],
        "response_time_hours": [response_time_hours],
        "aid_provided": [aid_provided],
        "infrastructure_damage_index": [infrastructure_damage_index]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Major Disaster Predicted")
    else:
        st.success("✅ Non-Major Disaster Predicted")
