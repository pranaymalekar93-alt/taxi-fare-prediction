import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Taxi Fare Prediction")
st.write("Estimate your taxi fare using Machine Learning.")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "random_forest_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter Your Trip Details")

distance = st.number_input(
    "Distance (km)", min_value=0.1, value=10.0, step=0.5
)
duration = st.number_input(
    "Duration (minutes)", min_value=1.0, value=25.0, step=1.0
)
pickup_hour = st.slider("Pickup Hour", 0, 23, 14)

day_names = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
day_name = st.selectbox("Pickup Day", day_names)
pickup_day = day_names.index(day_name)

passengers = st.number_input(
    "Number of Passengers",
    min_value=1, max_value=6, value=2
)

if st.button("Predict Fare"):
    trip = pd.DataFrame([{
        "distance_km": distance,
        "duration_min": duration,
        "pickup_hour": pickup_hour,
        "pickup_day": pickup_day,
        "passenger_count": passengers
    }])

    fare = model.predict(trip)[0]
    st.success(f"Estimated Taxi Fare: {fare:.2f}")
    st.caption(
        "This is an estimate from a model trained on simulated data."
    )
