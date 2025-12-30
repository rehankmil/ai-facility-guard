import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Konfigurasi Halaman
st.set_page_config(
    page_title="AI FacilityGuard",
    page_icon="🛡️",
    layout="centered"
)

# Judul dan Deskripsi
st.title("🛡️ AI FacilityGuard")
st.write("Prototipe Deteksi Kerusakan Struktural Kampus (Tembok/Lantai)")

# Fungsi Load Model (Di-cache agar cepat)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    return model

# Load model saat aplikasi mulai
with st.spinner('Sedang memuat model AI...'):
    model = load_model()

# Fungsi Preprocessing & Prediksi
def import_and_predict(image_data, model):
    # Resize gambar ke 224x224 (sesuai input Teachable Machine)
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Ubah ke array numpy
    img_array = np.asarray(image)
    
    # Normalisasi (sama seperti di Teachable Machine)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    # Buat batch data (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Prediksi
    prediction = model.predict(data)
    return prediction

# --- INPUT USER ---
opt = st.selectbox("Pilih Metode Input:", ("Kamera HP", "Upload Foto"))

image = None

if opt == "Kamera HP":
    img_file_buffer = st.camera_input("Ambil Foto Dinding/Lantai")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
else:
    file_uploader = st.file_uploader("Upload gambar", type=['jpg', 'png', 'jpeg'])
    if file_uploader is not None:
        image = Image.open(file_uploader)

# --- PROSES DETEKSI ---
if image is not None:
    st.image(image, caption="Gambar Input", use_column_width=True)
    
    # Tombol Analisis
    if st.button("🔍 Deteksi Kondisi"):
        prediction = import_and_predict(image, model)
        
        # Baca hasil (Teachable Machine urutannya sesuai label yang kita buat)
        # Indeks 0 = Normal, Indeks 1 = Retak (Tergantung urutan Anda di Teachable Machine!)
        # Pastikan cek file labels.txt Anda. Asumsi di sini: 0=Normal, 1=Retak.
        
        class_names = ['Normal', 'Retak'] 
        confidence_scores = prediction[0]
        max_score_index = np.argmax(confidence_scores)
        result_label = class_names[max_score_index]
        confidence = confidence_scores[max_score_index]

        st.divider()
        
        # Tampilkan Hasil
        if result_label == "Normal":
            st.success(f"✅ Kondisi: **NORMAL**")
            st.info("Fasilitas dalam kondisi baik.")
        else:
            st.error(f"⚠️ Kondisi: **RETAK (RUSAK)**")
            st.warning("Rekomendasi: Perlu pemeriksaan struktural segera!")
            
        st.write(f"Tingkat Keyakinan AI: **{confidence*100:.2f}%**")