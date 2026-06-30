import streamlit as st
import numpy as np
import joblib

# Muat model dan scaler di bagian paling atas (Prinsip Inference)
loaded_model = joblib.load('model_risiko_v1.joblib')
loaded_scaler = joblib.load('scaler_risiko_v1.joblib')

st.title("Simulator Risiko Kegagalan Sistem")
st.write("Aplikasi untuk memprediksi skor risiko mesin berdasarkan input sensor.")

# Input dari pengguna
suhu_input = st.number_input("Suhu Mesin", value=85.0)
getaran_input = st.number_input("Getaran Mesin", value=7.0)

# Monitoring Drift Sederhana (Sesuai Lampiran Panduan)
if suhu_input > 120 or suhu_input < 10:
    st.warning("⚠️ Input di luar jangkauan data latihan. Hasil simulasi mungkin tidak akurat!")

if st.button("Jalankan Simulasi"):
    # Proses data input
    data_baru = np.array([[suhu_input, getaran_input]])
    data_baru_scaled = loaded_scaler.transform(data_baru)
    
    # Prediksi
    hasil_simulasi = loaded_model.predict(data_baru_scaled)
    st.success(f"Hasil Simulasi Risiko: {hasil_simulasi[0]:.2f}")