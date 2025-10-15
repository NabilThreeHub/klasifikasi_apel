import streamlit as st
import pandas as pd
import joblib

model =joblib.load("model_klasifikasi_apel.joblib")

st.set_page_config(
	page_title = "Klasifikasi Apel",
	page_icon = ":apple:",
)

st.title(":apple: Belajar Klasifikasi Apel")
st.markdown("Aplikasi Machine Learning untuk Klasifikasi Apel Bagus, Sedang, Jelek")

diameter = st.slider ("Diameter", 5.0, 10.0, 7.0)
berat = st.slider ("Berat", 100.0, 300.0,150.0)
tebal_kulit =  st.slider ("Tebal Kulit", 0.5, 1.0, 0.7)
kadar_gula =  st.slider ("Kadar Gula", 8.0, 11.0, 14.0)
asal_daerah = st.pills ("Asal Daerah", ["Boyolali", "Malang", "Garut"], default ="Garut")
warna =  st.pills ("Warna", ["hijau", "kuning kemerahan", "merah"], default ="merah")
musim_panen =  st.pills ("Musim Panen", ["kemarau", "hujan"], default ="hujan")

if st.button ("Prediksi", type ="primary"):
	data_baru = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]],columns=["diameter", "berat", "tebal_kulit", "kadar_gula", 	"asal_daerah", 	"warna", "musim_panen"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Prediksi {prediksi} dengan keyakinan {presentase*100:.2f}%")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :apple: oleh Nabil Albara")