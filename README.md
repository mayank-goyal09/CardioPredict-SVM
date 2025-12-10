# 🫀 CardioPredict – Heart Disease Risk Classifier (SVM)

CardioPredict is a machine learning project that predicts the likelihood of heart disease based on patient health indicators such as **age**, **blood pressure**, **cholesterol**, **glucose levels**, and lifestyle factors. It uses a **Support Vector Machine (SVM)** classifier and an interactive **Streamlit** web app for easy risk assessment.

## 🚀 Project Overview

- Predicts heart disease risk (0 = low risk, 1 = high risk).
- Built using **Python, Pandas, NumPy, Scikit-learn**.
- Uses **Support Vector Machine (SVM)** for classification.
- Deployed as an interactive app using **Streamlit**.

## 📂 Project Structure

```bash
CardioPredict-SVM/
├── data/
│   └── cardio_data.csv          # Dataset (add your dataset here)
├── notebooks/
│   └── cardio_eda_model.ipynb   # Exploratory Data Analysis & model building
├── app/
│   └── app.py                   # Streamlit app script
├── models/
│   └── svm_model.pkl            # Trained SVM model (generated after training)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

> Note: Some files like the dataset, notebook, and trained model need to be added from your local project.

## 🧠 Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **ML Algorithm:** Support Vector Machine (SVM)
- **App Framework:** Streamlit

## ⚙️ Installation

```bash
git clone https://github.com/mayank-goyal09/CardioPredict-SVM.git
cd CardioPredict-SVM

pip install -r requirements.txt
```

## 📊 Training the Model

Create a Python script or Jupyter notebook (e.g., `notebooks/cardio_eda_model.ipynb`) where you:

1. Load the dataset from `data/cardio_data.csv`.
2. Perform data cleaning and preprocessing.
3. Apply feature scaling.
4. Split the data into train and test sets.
5. Train an SVM classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
7. Save the trained model as `models/svm_model.pkl` using `joblib` or `pickle`.

Example snippet to save model:

```python
import joblib
joblib.dump(svm_model, "models/svm_model.pkl")
```

## 🖥️ Streamlit App (`app/app.py`)

Below is a basic template you can paste into `app/app.py` and then customize based on your final feature names and preprocessing logic:

```python
import streamlit as st
import numpy as np
import pandas as pd
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

# Map categorical inputs
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
```

Make sure the feature order and preprocessing (like scaling or encoding) in the app matches the training pipeline.

## 🧪 Model Evaluation

During training, evaluate the SVM classifier using:

- Accuracy score
- Precision, Recall, F1-score
- Confusion matrix
- Classification report

You can include visualizations like heatmaps or ROC curves in your notebook.

## 🌐 Running the Streamlit App

From the project root directory, run:

```bash
streamlit run app/app.py
```

## 📌 Future Improvements

- Add hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Try other algorithms (Logistic Regression, Random Forest, XGBoost) and compare performance.
- Add more features like smoking status, BMI, and physical activity.
- Deploy the app on Streamlit Community Cloud.

---

If you like this project, feel free to ⭐ the repo and share it!
