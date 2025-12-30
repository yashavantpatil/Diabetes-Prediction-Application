import pandas as pd
import numpy as np
import joblib
import streamlit as st

model = joblib.load(open("logistic_regression_model for Diabetes.joblib", "rb"))

st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies", 0, 20)
glu = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 100)

input_data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])

if st.button("Predict"):
    result = model.predict(input_data)
    st.success("Diabetic" if result[0] == 1 else "Not Diabetic")

