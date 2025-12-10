import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

# =========================================================
# MODEL CLASSES (Keep your existing logic)
# =========================================================

@dataclass
class CardioPredictConfig:
    numeric_features: List[str]
    categorical_features: List[str]
    test_size: float = 0.2
    random_state: int = 42
    param_grid: Optional[Dict[str, List[Any]]] = None


class CardioPredictSVM:
    def __init__(self, config: CardioPredictConfig):
        self.config = config

        if self.config.param_grid is None:
            self.config.param_grid = {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": [0.01, 0.1, 1, "scale"],
            }

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.numeric_features),
                ("cat", categorical_transformer, self.config.categorical_features),
            ]
        )

        base_svm = Pipeline(
            steps=[
                ("preprocess", self.preprocessor),
                ("model", SVC(kernel="rbf")),
            ]
        )

        self.grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=self.config.param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )

        self.best_model_ = None
        self.history_: Dict[str, Any] = {}

    def train_test_split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

    def fit(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        self.best_model_ = self.grid_search.best_estimator_
        self.history_["best_params"] = self.grid_search.best_params_
        self.history_["best_cv_score"] = self.grid_search.best_score_
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def decision_scores(self, X):
        return self.best_model_.decision_function(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)

        try:
            scores = self.decision_scores(X_test)
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = None

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc,
            "best_params": self.history_.get("best_params"),
            "best_cv_score": self.history_.get("best_cv_score"),
        }

    def predict_patient(self, patient_dict):
        X_row = pd.DataFrame([patient_dict])

        for col in self.config.numeric_features + self.config.categorical_features:
            if col not in X_row.columns:
                X_row[col] = np.nan

        X_row = X_row[
            self.config.numeric_features + self.config.categorical_features
        ]

        pred = self.predict(X_row)[0]
        score = None
        try:
            score = float(self.decision_scores(X_row)[0])
        except Exception:
            pass

        return {"pred_class": int(pred), "score": score}


# =========================================================
# LOAD MODEL (Cached)
# =========================================================

@st.cache_resource
def load_trained_model():
    df = pd.read_csv("CVD_Dataset.csv")
    threshold = df["CVD Risk Score"].quantile(0.75)
    df["cvd_high_risk"] = (df["CVD Risk Score"] >= threshold).astype(int)

    drop_cols = ["cvd_high_risk", "CVD Risk Score", "CVD Risk Level", "ID"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["cvd_high_risk"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    config = CardioPredictConfig(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=0.2,
        random_state=42,
    )

    model = CardioPredictSVM(config)
    X_train, X_test, y_train, y_test = model.train_test_split(X, y)
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)

    return model, results, numeric_features, categorical_features


cardio_model, eval_results, numeric_features, categorical_features = (
    load_trained_model()
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CardioPredict Pro | CVD Risk Engine",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# MEDICAL RED + DARK BROWN THEME
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Red Hat Display', sans-serif;
}

/* Background: Very dark brown with subtle red glow */
.stApp {
    background: radial-gradient(circle at top right, #2d0f0f 0%, #1a0a0a 40%, #0a0404 100%);
    color: #f5e6e6;
}

.block-container {
    max-width: 1300px;
}

/* Hide default header */
[data-testid="stHeader"] {background: transparent;}
header {background: transparent;}

/* Hero Medical Header */
.cardio-hero {
    background: linear-gradient(135deg, #3d0f0f 0%, #1a0a0a 60%);
    border-radius: 20px;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 18px 45px rgba(139, 0, 0, 0.6);
    border: 2px solid rgba(220, 38, 38, 0.4);
    margin-bottom: 22px;
}
.hero-left {
    display: flex;
    gap: 16px;
    align-items: center;
}
.hero-icon {
    font-size: 2.8rem;
    padding: 12px;
    border-radius: 18px;
    background: radial-gradient(circle at 30% 20%, #dc2626 0, #991b1b 40%, #3d0f0f 85%);
    box-shadow: 0 0 35px rgba(220, 38, 38, 0.7);
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    color: #fecaca;
}
.hero-sub {
    font-size: 0.9rem;
    color: #d1b8b8;
    margin-top: 2px;
}
.hero-tag {
    display: inline-block;
    margin-top: 8px;
    font-size: 0.75rem;
    color: #fca5a5;
    background: rgba(127,29,29,0.35);
    border-radius: 999px;
    padding: 3px 11px;
    border: 1px solid rgba(220,38,38,0.5);
}
.hero-badge {
    font-size: 0.8rem;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(153,27,27,0.2);
    border: 1px solid #dc2626;
    color: #fecaca;
    text-align: center;
    line-height: 1.3;
}

/* Sidebar - Dark Brown Medical Panel */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1a1a 0%, #1a0a0a 100%);
    border-right: 2px solid rgba(220,38,38,0.3);
}
[data-testid="stSidebar"] * {
    color: #f5e6e6;
}
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #fca5a5;
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
}
.sidebar-caption {
    font-size: 0.8rem;
    color: #b8a6a6;
    margin-bottom: 0.6rem;
}

/* Inputs */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background-color: #1a0a0a !important;
    border-radius: 10px !important;
    border: 1px solid #4a2626 !important;
    color: #f5e6e6 !important;
}
.stNumberInput label,
.stSelectbox label {
    color: #d1b8b8 !important;
    font-size: 0.85rem;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(120deg, #dc2626, #b91c1c);
    color: #fef2f2;
    font-weight: 800;
    border-radius: 999px;
    border: none;
    padding: 0.75rem 1.8rem;
    font-size: 1rem;
    box-shadow: 0 12px 28px rgba(220, 38, 38, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: #991b1b;
    color: #fef2f2;
}

/* Glass Panel */
.medical-panel {
    background: radial-gradient(circle at top left, rgba(45,15,15,0.95) 0%, rgba(26,10,10,0.9) 60%);
    border-radius: 18px;
    padding: 20px 22px;
    border: 1px solid rgba(74,38,38,0.8);
    box-shadow: 0 14px 35px rgba(0,0,0,0.9);
    margin-bottom: 20px;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #fca5a5;
    margin-bottom: 5px;
}
.panel-sub {
    font-size: 0.82rem;
    color: #b8a6a6;
    margin-bottom: 14px;
}

/* Result Card */
.result-card {
    border-radius: 18px;
    padding: 18px 20px;
    margin-top: 12px;
    margin-bottom: 10px;
    border: 2px solid rgba(220,38,38,0.6);
    background: radial-gradient(circle at top left, #3d0f0f 0%, #1a0a0a 60%);
    box-shadow: 0 16px 45px rgba(139,0,0,0.8);
}
.result-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #fca5a5;
}
.result-high-risk {
    font-size: 1.6rem;
    font-weight: 900;
    color: #dc2626;
    text-shadow: 0 0 12px rgba(220,38,38,0.6);
}
.result-low-risk {
    font-size: 1.6rem;
    font-weight: 900;
    color: #10b981;
    text-shadow: 0 0 12px rgba(16,185,129,0.6);
}
.result-score {
    font-size: 0.95rem;
    color: #f5e6e6;
    margin-top: 6px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: rgba(45,15,15,0.4);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(74,38,38,0.7);
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}
[data-testid="stMetricLabel"] {
    color: #d1b8b8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #fca5a5 !important;
    font-weight: 800;
    font-size: 1.4rem;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(45,15,15,0.6);
    border-radius: 12px;
    border: 1px solid rgba(220,38,38,0.4);
    color: #fca5a5;
    font-weight: 600;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(74,38,38,0.7);
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
<div class="cardio-hero">
    <div class="hero-left">
        <div class="hero-icon">🫀</div>
        <div>
            <div class="hero-title">CardioPredict Pro</div>
            <div class="hero-sub">Clinical Decision Support System for Cardiovascular Risk Stratification</div>
            <div class="hero-tag">SVM-Based Risk Engine · CAIR-CVD-2025 Dataset · Medical-Grade Analytics</div>
        </div>
    </div>
    <div class="hero-badge">
        Model v2.0<br/>
        Research Prototype
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR - PATIENT VITALS INPUT
# =========================================================

with st.sidebar:
    st.markdown("### 🩺 Patient Vitals & History")
    st.markdown(
        '<p class="sidebar-caption">Enter clinical parameters for CVD risk assessment.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">Demographics</div>', unsafe_allow_html=True)
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=52)
    sex = st.selectbox("Sex", ["Male", "Female"])

    st.markdown('<div class="sidebar-title">Physical Measurements</div>', unsafe_allow_html=True)
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=28.3)
    sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=145)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=92)

    st.markdown('<div class="sidebar-title">Laboratory Values</div>', unsafe_allow_html=True)
    chol = st.number_input(
        "Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=230.0
    )
    hdl = st.number_input("HDL (mg/dL)", min_value=10.0, max_value=100.0, value=38.0)
    fbs = st.number_input(
        "Fasting Blood Sugar (mmol/L)", min_value=3.0, max_value=20.0, value=7.2
    )

    st.markdown('<div class="sidebar-title">Risk Factors</div>', unsafe_allow_html=True)
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    diabetes = st.selectbox("Diabetes Status", ["No", "Yes"])
    activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    fam_hist = st.selectbox("Family History of CVD", ["No", "Yes"])

    st.markdown("---")
    predict_btn = st.button("⚡ Analyze Cardiovascular Risk")

# =========================================================
# MAIN DASHBOARD
# =========================================================

# Top Row: Model Performance Summary
st.markdown(
    """
<div class="medical-panel">
    <div class="panel-title">📊 Model Performance (Validation Set)</div>
    <div class="panel-sub">SVM classifier trained on CAIR-CVD-2025 dataset with GridSearchCV hyperparameter tuning.</div>
""",
    unsafe_allow_html=True,
)

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric("Model Accuracy", f"{eval_results['accuracy']:.3f}")
with perf_col2:
    if eval_results["roc_auc"] is not None:
        st.metric("ROC-AUC Score", f"{eval_results['roc_auc']:.3f}")
    else:
        st.metric("ROC-AUC Score", "N/A")
with perf_col3:
    cv_score = eval_results.get("best_cv_score", 0.0)
    st.metric("Cross-Val Score", f"{cv_score:.3f}")
with perf_col4:
    st.metric("Training Method", "5-Fold CV")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREDICTION RESULTS
# =========================================================

if predict_btn:
    patient_dict = {
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "Systolic BP": sbp,
        "Diastolic BP": dbp,
        "Total Cholesterol": chol,
        "HDL": hdl,
        "Fasting Blood Sugar": fbs,
        "Smoking Status": smoking,
        "Diabetes Status": diabetes,
        "Physical Activity Level": activity,
        "Family History of CVD": fam_hist,
    }

    result = cardio_model.predict_patient(patient_dict)
    pred_class = result["pred_class"]
    score = result["score"]

    # Result Card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">🩺 Risk Assessment Result</div>', unsafe_allow_html=True)

    if pred_class == 1:
        st.markdown(
            '<div class="result-high-risk">⚠️ HIGH CARDIOVASCULAR RISK</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="color:#fca5a5; font-size:0.9rem; margin-top:8px;">
            <b>Clinical Recommendation:</b> Immediate referral to cardiologist recommended. 
            Consider comprehensive cardiac workup including ECG, echocardiography, and stress testing.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-low-risk">✅ LOW/MODERATE CARDIOVASCULAR RISK</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="color:#d1f4e0; font-size:0.9rem; margin-top:8px;">
            <b>Clinical Recommendation:</b> Continue routine preventive care. 
            Maintain healthy lifestyle with regular exercise and balanced diet. Annual screening recommended.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if score is not None:
        st.markdown(
            f'<div class="result-score">Decision Function Score: <b>{score:.3f}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Patient Summary Panel
    st.markdown(
        """
<div class="medical-panel">
    <div class="panel-title">📋 Patient Clinical Summary</div>
""",
        unsafe_allow_html=True,
    )

    sum_col1, sum_col2, sum_col3 = st.columns(3)

    with sum_col1:
        st.markdown("**Demographics**")
        st.write(f"• Age: {age} years")
        st.write(f"• Sex: {sex}")
        st.write(f"• BMI: {bmi} kg/m²")

    with sum_col2:
        st.markdown("**Hemodynamics**")
        st.write(f"• SBP: {sbp} mmHg")
        st.write(f"• DBP: {dbp} mmHg")
        st.write(f"• Pulse Pressure: {sbp - dbp} mmHg")

    with sum_col3:
        st.markdown("**Lab Results**")
        st.write(f"• Total Chol: {chol} mg/dL")
        st.write(f"• HDL: {hdl} mg/dL")
        st.write(f"• FBS: {fbs} mmol/L")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# MODEL DETAILS (Expandable)
# =========================================================

with st.expander("🔬 View Model Technical Details"):
    st.markdown("**Best Hyperparameters:**")
    st.json(eval_results.get("best_params", {}))

    st.markdown("**Confusion Matrix:**")
    st.write(eval_results["confusion_matrix"])

    st.markdown("**Classification Report:**")
    st.text(eval_results["classification_report"])

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<hr style="margin-top:2.5rem; border-color:#4a2626;">
<div style="text-align:center; padding:0.9rem 0; color:#b8a6a6; font-size:0.82rem;">
    <div style="margin-bottom:4px; color:#d1b8b8;">
        © 2025 CardioPredict Pro · Clinical Decision Support System · Built by <span style="color:#fca5a5; font-weight:600;">Mayank Goyal</span>
    </div>
    <div>
        <a href="https://www.linkedin.com/in/mayank-goyal09" target="_blank"
           style="color:#fca5a5; text-decoration:none; margin-right:18px;">
            🔗 LinkedIn
        </a>
        <a href="https://github.com/mayank-goyal09" target="_blank"
           style="color:#fca5a5; text-decoration:none;">
            💻 GitHub
        </a>
    </div>
    <div style="margin-top:6px; font-size:0.75rem; color:#8a7373;">
        ⚠️ Research prototype for educational purposes only. Not for clinical diagnosis or treatment decisions.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

