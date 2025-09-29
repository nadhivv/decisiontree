import streamlit as st
import numpy as np
import joblib

# === Load Model & Scaler ===
clf = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🫀 Prediksi Penyakit Jantung")
st.write("Model diload dari file `heart_model.pkl`")

# === Input User ===
age = st.number_input("Usia", min_value=1, max_value=120, value=45)
sex = st.selectbox("Jenis Kelamin (1=Laki-laki, 0=Perempuan)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown)", [0, 1, 2, 3])

# Gabungkan input user
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scaling input
user_input_scaled = scaler.transform(user_input)

# Prediksi
if st.button("🔍 Prediksi"):
    prediction = clf.predict(user_input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ Pasien Berisiko Mengidap Penyakit Jantung")
    else:
        st.success("✅ Pasien Tidak Berisiko Penyakit Jantung")
