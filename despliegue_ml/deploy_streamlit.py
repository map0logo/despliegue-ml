import pickle

import pandas as pd
import streamlit as st

with open("../models/adult_census_lr.pkl", "rb") as f:
    trained_model = pickle.load(f)

st.set_page_config(page_title="Adult Census Prediction")

st.title("Adult Census Prediction")

st.write(
    "Enter values for age, capital-gain, capital-loss, and hours-per-week to get a prediction."
)

# Get input data from user
age = st.number_input("Age:", value=25, min_value=0, max_value=100)
capital_gain = st.number_input("Capital Gain:", value=0, min_value=0, max_value=99999)
capital_loss = st.number_input("Capital Loss:", value=0, min_value=0, max_value=99999)
hours_per_week = st.number_input(
    "Hours per Week:", value=40, min_value=0, max_value=100
)

input_data = pd.DataFrame(
    [
        {
            "age": age,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
        }
    ]
)

# Make prediction
prediction = trained_model.predict(input_data)

# Display prediction
st.write("Prediction:", prediction[0])
