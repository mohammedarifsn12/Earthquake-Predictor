import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_earthquake(data1, data2, data3):
    arr = np.array([[data1, data2, data3]], dtype=np.float64)  # Ensure correct dtype
    print(f"Input shape: {arr.shape}, Input data: {arr}")  # Debugging
    output = model.predict(arr)
    return output[0]

def get_severity_label(output):
    severity_levels = {0: "Low", 1: "Moderate", 2: "High"}
    return severity_levels.get(output, "Unknown")

# Streamlit UI
st.title("Earthquake Prediction App")
st.write("Enter the required values to predict earthquake severity.")

data1 = st.number_input("Enter value for feature 1:", min_value=0.0, step=0.1)
data2 = st.number_input("Enter value for feature 2:", min_value=0.0, step=0.1)
data3 = st.number_input("Enter value for feature 3:", min_value=0.0, step=0.1)

if st.button("Predict"):
    output = predict_earthquake(float(data1), float(data2), float(data3))
    severity = get_severity_label(output)
    st.write(f"### Prediction: {output}")
    st.write(f"### Severity Level: {severity}")
