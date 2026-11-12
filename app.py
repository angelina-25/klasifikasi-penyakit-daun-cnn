
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# 1. KONFIGURASI APLIKASI
st.set_page_config(
    page_title="Deteksi Penyakit Daun",
    page_icon="🌿",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "cnn_plantvillage.keras"
INFO_PATH = BASE_DIR / "info_kelas.json"

# 2. MEMUAT MODEL CNN
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# 3. MEMUAT INFORMASI KELAS
@st.cache_data
def load_class_info():
    with open(INFO_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

# 4. TAMPILAN APLIKASI
st.title("🌿 Deteksi Penyakit Daun Tanaman")

st.markdown(
    """
    Selamat datang di aplikasi klasifikasi penyakit daun tanaman.

    Aplikasi ini menggunakan metode **Custom Convolutional Neural Network (CNN)**
    dengan dataset PlantVillage untuk mengenali kondisi daun tanaman.
    """
)

st.divider()

st.subheader("📷 Upload Gambar Daun")

uploaded_file = st.file_uploader(
    "Pilih gambar daun tanaman",
    type=["jpg", "jpeg", "png"]
)

# 5. PROSES KLASIFIKASI
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar daun yang diunggah",
        use_container_width=True
    )

    if st.button("🔍 Analisis Gambar", type="primary"):

        try:
            with st.spinner("Sedang menganalisis gambar..."):

                # Memuat model dan informasi kelas
                model = load_model()
                info_kelas = load_class_info()

                # Mengambil ukuran gambar
                img_size = tuple(info_kelas["img_size"])

                # Mengubah ukuran gambar menjadi 128x128
                image = image.resize(
                    (img_size[1], img_size[0])
                )

                # Mengubah gambar menjadi array
                image_array = np.array(
                    image,
                    dtype=np.float32
                )

                # Menambahkan dimensi batch
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Prediksi menggunakan model CNN
                # Tidak perlu membagi 255 karena model
                # sudah memiliki layer Rescaling.
                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                # Mengambil indeks prediksi tertinggi
                predicted_index = int(
                    np.argmax(prediction[0])
                )

                # Mengambil nama kelas
                nama_kelas = info_kelas["kelas"]

                hasil_kelas = nama_kelas[predicted_index]

                # Mengambil nama tampilan kelas
                label_tampilan = info_kelas["label_tampilan"]

                hasil_prediksi = label_tampilan.get(
                    hasil_kelas,
                    hasil_kelas
                )

                # Mengambil tingkat keyakinan
                confidence = float(
                    prediction[0][predicted_index]
                ) * 100

            # 6. MENAMPILKAN HASIL PREDIKSI
            st.divider()

            st.subheader("📊 Hasil Klasifikasi")

            st.success(
                f"Prediksi: {hasil_prediksi}"
            )

            st.metric(
                label="Tingkat Keyakinan Model",
                value=f"{confidence:.2f}%"
            )

            # 7. PROBABILITAS SETIAP KELAS
            st.subheader("Probabilitas Setiap Kelas")

            urutan = np.argsort(
                prediction[0]
            )[::-1]

            for index in urutan:

                nama = label_tampilan.get(
                    nama_kelas[index],
                    nama_kelas[index]
                )

                nilai = float(
                    prediction[0][index]
                )

                st.write(
                    f"**{nama}** — {nilai * 100:.2f}%"
                )

                st.progress(nilai)

            st.info(
                "Hasil ini merupakan prediksi berdasarkan "
                "model CNN dan bukan diagnosis pasti penyakit tanaman."
            )

        except Exception as error:

            st.error(
                "Terjadi kesalahan saat memproses gambar."
            )

            st.exception(error)

# 8. FOOTER
st.divider()

st.caption(
    "Klasifikasi Penyakit Daun Tanaman | "
    "Custom CNN | Dataset PlantVillage"
)
