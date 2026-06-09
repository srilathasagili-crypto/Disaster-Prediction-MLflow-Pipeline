import streamlit as st
import pickle
import pandas as pd

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Disaster Prediction App")

st.title("🌍 Disaster Prediction App")

# Inputs
severity = st.slider("Severity Level", 1, 10, 5)
affected_population = st.number_input("Affected Population", min_value=0, value=1000)

if st.button("Predict"):

    data = pd.DataFrame({
        "Severity": [severity],
        "Affected_Population": [affected_population]
    })

    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")