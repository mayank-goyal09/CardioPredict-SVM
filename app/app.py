import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="CardioPredict – Heart Disease Risk", page_icon="🫀", layout="centered")

st.title("🫀 CardioPredict – Heart Disease Risk Classifier")
st.write("Enter patient health details to predict the likelihood of heart disease using an SVM classifier.")

@st.cache_resource
def load_model():
    model = joblib.load("models/svm_model.pkl")
    return model

model = load_model()

st.sidebar.header("Patient Health Inputs")

age = st.sidebar.slider("Age", 18, 100, 45)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 230)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ("No", "Yes"))
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1)

sex_val = 1 if sex == "Male" else 0
fasting_bs_val = 1 if fasting_bs == "Yes" else 0
exercise_angina_val = 1 if exercise_angina == "Yes" else 0

input_features = np.array([[age, sex_val, resting_bp, cholesterol, fasting_bs_val, max_hr, exercise_angina_val, oldpeak]])

if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(input_features)[0]
    proba = None
    try:
        proba = model.predict_proba(input_features)[0][1]
    except Exception:
        pass

    if prediction == 1:
        st.error("High risk of heart disease detected. Further medical consultation is recommended.")
    else:
        st.success("Low risk of heart disease detected. Keep maintaining a healthy lifestyle!")

    if proba is not None:
        st.write(f"Model confidence (risk probability): **{proba:.2f}**")

st.markdown("---")
st.caption("This app is for educational purposes only and should not be used as a substitute for professional medical diagnosis.")
