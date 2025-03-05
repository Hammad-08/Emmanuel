# ------------------------------
# Heart Disease Prediction Web App Using Streamlit
# ------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import shap
from joblib import load

# Load trained model & preprocessor
rf_clf = load("random_forest_model.pkl")  # Trained Random Forest model
preprocessor = load("preprocessor.pkl")  # Preprocessing pipeline
feature_names_transformed = np.load("feature_names.npy", allow_pickle=True)  # Feature names

# ------------------------------
# User Input Section
# ------------------------------

st.subheader("📝 Enter Patient Data for Prediction")

# Define input fields for user
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholestoral = st.number_input("Cholestoral Level (mg/dL)", min_value=100, max_value=500, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
rest_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
num_vessels = st.selectbox("Number of Vessels Colored by Fluoroscopy", ["0", "1", "2", "3"])
thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# ------------------------------
# Convert Categorical Inputs
# ------------------------------
sex = 1 if sex == "Male" else 0
chest_pain_mapping = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}
chest_pain = chest_pain_mapping[chest_pain]
fasting_bs = 1 if fasting_bs == "Yes" else 0
rest_ecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
rest_ecg = rest_ecg_mapping[rest_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
slope_mapping = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
slope = slope_mapping[slope]
thal_mapping = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
thalassemia = thal_mapping[thalassemia]
num_vessels = int(num_vessels)

# ------------------------------
# Convert to DataFrame with Correct Column Names
# ------------------------------

original_features = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholestoral",
    "fasting_blood_sugar", "rest_ecg", "Max_heart_rate", "exercise_induced_angina",
    "oldpeak", "slope", "vessels_colored_by_flourosopy", "thalassemia"
]

user_input = np.array([[age, sex, chest_pain, rest_bp, cholestoral, fasting_bs, rest_ecg, max_heart_rate,
                        exercise_angina, oldpeak, slope, num_vessels, thalassemia]])

user_input_df = pd.DataFrame(user_input, columns=original_features)

# ------------------------------
# Apply Preprocessing
# ------------------------------

user_input_transformed = preprocessor.transform(user_input_df)

# Convert transformed output back to DataFrame with transformed feature names
user_input_transformed_df = pd.DataFrame(user_input_transformed, columns=feature_names_transformed)

# ------------------------------
# Prediction Button
# ------------------------------

if st.button("Predict Heart Disease"):
    # Model Prediction
    prediction = rf_clf.predict(user_input_transformed_df)[0]

    # Display Results
    if prediction == 1:
        st.error(f"⚠️ High Risk!")
    else:
        st.success(f"✅ Low Risk!")