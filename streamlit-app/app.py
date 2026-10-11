"""
Heart Disease Prediction - Student Demo Web App
-------------------------------------------------
A Streamlit web application that allows users and evaluators to interact
with the trained Support Vector Classifier (SVC) from the Heart Disease
Prediction project without needing to install Python or Jupyter.

Course: AIC354 (COMSATS University)
Supervised Learning Task: Binary Classification (Heart Disease: Yes / No)
Input Features: Age, Gender, Chest Pain Type, Maximum Heart Rate Achieved
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load trained model and recreate deterministic LabelEncoders
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "heart_model.pkl"
DATA_PATH = Path(__file__).parent / "sample-data.csv"

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category list,
    so fitting on the exact fixed lists reproduces the exact same numeric codes.
    - Gender: ['Female', 'Male'] -> [0, 1]
    - ChestPain: ['Asymptomatic', 'Atypical Angina', 'Non-Anginal Pain', 'Typical Angina'] -> [0, 1, 2, 3]
    """
    encoders = {}
    encoders["Gender"] = LabelEncoder().fit(["Female", "Male"])
    encoders["ChestPain"] = LabelEncoder().fit([
        "Asymptomatic",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Typical Angina"
    ])
    return encoders

@st.cache_resource
def load_model():
    """
    Loads the pickled trained SVC model or trains a fresh instance if a
    cross-environment pickle version mismatch occurs on Streamlit Cloud.
    """
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and "model" in loaded:
                    return loaded["model"]
                return loaded
        except Exception:
            pass  # Fall through to fallback training

    # Robust fallback: Train identical SVC model on sample-data.csv
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(Path(__file__).parent.parent / "1 - Heart Disease Prediction" / "sample-data.csv")
    
    encoders = build_label_encoders()
    target_encoder = LabelEncoder().fit(["No", "Yes"])
    
    X = pd.DataFrame({
        "Age": df["Age"],
        "Gender": encoders["Gender"].transform(df["Gender"]),
        "ChestPain": encoders["ChestPain"].transform(df["ChestPain"]),
        "MaxHeartRate": df["MaxHeartRate"]
    })
    y = target_encoder.transform(df["HeartDisease"])
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    m = SVC(kernel="linear", probability=True, random_state=42)
    m.fit(X_train, y_train)
    return m

# Initialize model and encoders
model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page Header & Context
# ---------------------------------------------------------------------------
st.title("🫀 Heart Disease Risk Predictor")
st.markdown(
    """
    This is an educational machine learning web application built for **AIC354**
    at **COMSATS University**. It uses a **Support Vector Classifier (SVC)**
    trained on the benchmark **UCI Cleveland Heart Disease dataset** using 4 key clinical features:
    **Age**, **Gender**, **Chest Pain Type**, and **Maximum Heart Rate**.
    
    *Held-out Test Accuracy: **~75.4%** | F1-Score: **0.76**.*
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Patient Input Form
# ---------------------------------------------------------------------------
st.subheader("📋 Enter Patient Attributes")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age (Years)",
        min_value=20,
        max_value=90,
        value=54,
        step=1,
        help="Patient's age in years."
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0,
        help="Biological sex of the patient."
    )

with col2:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ],
        index=3,
        help="Classification of chest pain symptoms (Asymptomatic is common in chronic coronary disease)."
    )
    
    max_heart_rate = st.slider(
        "Maximum Heart Rate Achieved (thalach)",
        min_value=60,
        max_value=220,
        value=150,
        step=1,
        help="Peak heart rate achieved during exercise stress testing (beats per minute)."
    )

st.write("")
predict_button = st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction & Inference
# ---------------------------------------------------------------------------
if predict_button:
    # 1. Structure raw input dataframe
    user_input = pd.DataFrame({
        "Age": [int(age)],
        "Gender": [gender],
        "ChestPain": [chest_pain],
        "MaxHeartRate": [int(max_heart_rate)]
    })
    
    # 2. Encode categorical columns matching notebook training
    encoded_features = pd.DataFrame({
        "Age": user_input["Age"],
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "MaxHeartRate": user_input["MaxHeartRate"]
    })
    
    # 3. Model prediction
    prediction = model.predict(encoded_features)[0]
    
    # Check if probability estimation is available
    has_proba = hasattr(model, "predict_proba")
    proba_disease = None
    if has_proba:
        try:
            probabilities = model.predict_proba(encoded_features)[0]
            # Class 1 corresponds to Disease ('Yes')
            proba_disease = probabilities[1]
        except Exception:
            proba_disease = None

    st.divider()
    st.subheader("📊 Diagnostic Assessment")
    
    if prediction == 1:
        st.error("### ⚠️ Heart Disease: LIKELY")
        st.markdown(
            "The Support Vector Classifier predicts that the patient **may have coronary heart disease**."
        )
    else:
        st.success("### ✅ Heart Disease: UNLIKELY")
        st.markdown(
            "The Support Vector Classifier predicts that the patient is **unlikely to have heart disease**."
        )

    # Display probability meter if available
    if proba_disease is not None:
        confidence_pct = proba_disease * 100
        st.write(f"**Estimated Disease Probability:** `{confidence_pct:.1f}%`")
        st.progress(float(proba_disease))
        if proba_disease > 0.5:
            st.caption("Risk level is elevated based on model decision boundary.")
        else:
            st.caption("Risk level is within the lower-risk boundary.")

    # Show vector sent to model (matching Titanic reference app)
    with st.expander("🔍 See Encoded Feature Vector Sent to Model"):
        st.markdown("The SVC model takes this exact 4-dimensional numerical feature vector:")
        st.dataframe(encoded_features, hide_index=True)
        st.caption(
            "Encoding Key: Gender [Female=0, Male=1] | "
            "ChestPain [Asymptomatic=0, Atypical Angina=1, Non-Anginal Pain=2, Typical Angina=3]"
        )

# ---------------------------------------------------------------------------
# Educational Sidebar / Notes
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this Project")
    st.info(
        """
        **COMSATS University Islamabad**
        - **Course:** AIC354 (Machine Learning)
        - **Model:** Support Vector Classifier (SVC)
        - **Kernel:** Linear
        - **Dataset:** UCI Cleveland Heart Disease (303 instances)
        """
    )
    
    st.subheader("Feature Definitions")
    st.markdown(
        """
        - **Age:** Patient age in years.
        - **Gender:** Biological sex (Male / Female).
        - **Chest Pain (cp):**
          - *Typical Angina:* Classic angina related to exertion.
          - *Atypical Angina:* Chest pain with atypical features.
          - *Non-Anginal Pain:* Non-cardiac chest discomfort.
          - *Asymptomatic:* Absence of chest discomfort, but potential silent ischemia.
        - **Max Heart Rate (thalach):** Peak heart rate achieved under exercise stress testing.
        """
    )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This application is for academic coursework and demonstration purposes only. "
    "It is not intended for actual clinical diagnosis. Always consult certified healthcare professionals."
)
