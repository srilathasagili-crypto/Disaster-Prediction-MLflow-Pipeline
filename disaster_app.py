import streamlit as st
import pickle
import pandas as pd

MODEL_PATH = "disaster_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Disaster Prediction App", page_icon="🌍")

st.title("🌍 Disaster Prediction App")

severity_level = st.slider("Severity Level", 1, 10, 5)
affected_population = st.number_input("Affected Population", min_value=0, value=1000)
estimated_economic_loss_usd = st.number_input("Estimated Economic Loss (USD)", min_value=0.0, value=10000.0)
response_time_hours = st.number_input("Response Time (Hours)", min_value=0.0, value=12.0)
infrastructure_damage_index = st.slider("Infrastructure Damage Index", 0, 100, 50)

aid_provided = st.selectbox("Aid Provided", ["Yes", "No"])
disaster_type = st.selectbox("Disaster Type", ["Flood", "Earthquake", "Cyclone", "Wildfire"])
location = st.selectbox("Location", ["India", "USA", "Japan", "Australia"])

latitude = st.number_input(
    "Latitude",
    value=17.3850
)

longitude = st.number_input(
    "Longitude",
    value=78.4867
)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "severity_level":[severity_level],
        "affected_population":[affected_population],
        "estimated_economic_loss_usd":[estimated_economic_loss_usd],
        "response_time_hours":[response_time_hours],
        "infrastructure_damage_index":[infrastructure_damage_index],
        "aid_provided":[aid_provided],
        "disaster_type":[disaster_type],
        "location":[location],
        "latitude":[latitude],
        "longitude":[longitude]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error(" Major Disaster Predicted")
    else:
        st.success(" Minor Disaster Predicted")