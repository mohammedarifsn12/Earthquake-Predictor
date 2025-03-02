import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

def predict_earthquake(data1, data2, data3):
    arr = np.array([[data1, data2, data3]])
    output = model.predict(arr)
    return output[0]

def get_severity_label(output):
    if output < 4:
        return 'No'
    elif 4 <= output < 6:
        return 'Low'
    elif 6 <= output < 8:
        return 'Moderate'
    elif 8 <= output < 9:
        return 'High'
    else:
        return 'Very High'

# Streamlit UI
st.title("Earthquake Prediction App")
st.write("Enter the required parameters to predict earthquake severity.")

# Input fields
data1 = st.number_input("Enter value for a:", min_value=0.0, step=0.1)
data2 = st.number_input("Enter value for b:", min_value=0.0, step=0.1)
data3 = st.number_input("Enter value for c:", min_value=0.0, step=0.1)

if st.button("Predict"):
    output = predict_earthquake(int(data1), int(data2), int(data3))
    severity = get_severity_label(output)
    st.write(f"### Prediction: {output}")
    st.write(f"### Severity Level: {severity}")