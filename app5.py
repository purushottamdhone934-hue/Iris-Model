import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Iris Flower Prediction")

st.title("Iris Flower Species Prediction")

sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })

    prediction = model.predict(input_data)

    species = label_encoder.inverse_transform(prediction)

    st.success("Predicted Species : " + species[0])