import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Deteksi Penyakit Daun",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "cnn_plantvillage.keras"
INFO_PATH = BASE_DIR / "info_kelas.json"


# ==========================================
# 2. DESAIN APLIKASI (CUSTOM CSS)
# ==========================================

CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --hijau-tua: #14532d;
    --hijau: #16a34a;
    --hijau-muda: #86efac;
    --hijau-pucat: #f0fdf4;
    --teks: #1f2937;
    --abu: #6b7280;
}

/* Background aplikasi */
.stApp {
    background:
        radial-gradient(circle at 10% 0%, #dcfce7 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, #d1fae5 0%, transparent 35%),
        linear-gradient(180deg, #f7fff9 0%, #ecfdf5 100%);
    font-family: 'Poppins', sans-serif;
}

/* Font */
html, body, [class*="css"], .stMarkdown, p, label, span {
    font-family: 'Poppins', sans-serif;
}

/* Sembunyikan elemen bawaan Streamlit */
#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Kontainer utama */
.block-container {
    max-width: 760px;
    padding: 2.2rem 2.4rem 2.5rem 2.4rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(134, 239, 172, 0.55);
    border-radius: 28px;
    box-shadow: 0 20px 50px rgba(22, 101, 52, 0.12);
    animation: munculNaik 0.7s ease both;
}

/* Animasi */
@keyframes munculNaik {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Judul utama */
h1 {
    text-align: center;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(90deg, #14532d, #16a34a 60%, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding-bottom: 0.4rem;
}

/* Subjudul */
h2, h3 {
    color: var(--hijau-tua) !important;
    font-weight: 700 !important;
}

h3 {
    padding-left: 0.8rem;
    border-left: 5px solid var(--hijau);
    border-radius: 3px;
    margin-top: 0.5rem;
}

/* Paragraf */
.stMarkdown p {
    color: var(--teks);
    line-height: 1.75;
}

/* Teks tebal */
.stMarkdown strong {
    color: var(--hijau-tua);
}

/* Garis pemisah */
hr {
    border: none !important;
    height: 3px !important;
    border-radius: 99px;
    background: linear-gradient(
        90deg,
        transparent,
        var(--hijau-muda),
        transparent
    ) !important;
    margin: 1.6rem 0 !important;
}

/* Area upload */
[data-testid="stFileUploader"] section {
    background: var(--hijau-pucat);
    border: 2px dashed var(--hijau);
    border-radius: 20px;
    padding: 1.6rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"] section:hover {
    background: #dcfce7;
    border-color: var(--hijau-tua);
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(22, 163, 74, 0.18);
}

[data-testid="stFileUploader"] label p {
    font-weight: 600;
    color: var(--hijau-tua);
}

/* Tombol upload */
[data-testid="stFileUploader"] button {
    border-radius: 12px;
    border: 1.5px solid var(--hijau);
    color: var(--hijau-tua);
    font-weight: 600;
}

/* Gambar upload */
[data-testid="stImage"] img {
    border-radius: 22px;
    border: 5px solid #ffffff;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
    transition: transform 0.35s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.015);
}

/* Caption gambar */
[data-testid="stImageCaption"],
figcaption {
    text-align: center;
    color: var(--abu) !important;
    font-style: italic;
}

/* Tombol analisis */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 0.85rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: #ffffff;
    background: linear-gradient(
        135deg,
        #15803d 0%,
        #16a34a 50%,
        #22c55e 100%
    );
    box-shadow: 0 10px 24px rgba(22, 163, 74, 0.38);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 32px rgba(22, 163, 74, 0.5);
    color: #ffffff;
}

/* Hasil prediksi */
[data-testid="stAlert"] {
    border-radius: 18px;
    border: none;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.07);
    font-weight: 500;
}

/* Kartu keyakinan */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #f0fdf4);
    border: 1.5px solid var(--hijau-muda);
    border-radius: 20px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    box-shadow: 0 8px 22px rgba(22, 163, 74, 0.12);
}

[data-testid="stMetricLabel"] {
    justify-content: center;
    color: var(--abu);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    justify-content: center;
    color: var(--hijau-tua);
    font-weight: 800;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #e5e7eb;
    border-radius: 99px;
    height: 10px;
}

[data-testid="stProgress"] > div > div > div > div {
    background-image: linear-gradient(90deg, #16a34a, #86efac);
    border-radius: 99px;
}

/* Footer */
[data-testid="stCaptionContainer"] {
    text-align: center;
    color: var(--abu);
    font-weight: 500;
}

/* Tampilan HP */
@media (max-width: 640px) {
    .block-container {
        padding: 1.4rem 1.1rem;
        border-radius: 20px;
    }

    h1 {
        font-size: 1.7rem !important;
    }
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# 3. MEMUAT MODEL CNN
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH.name}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ==========================================
# 4. MEMUAT INFORMASI KELAS
# ==========================================

@st.cache_data
def load_class_info():
    if not INFO_PATH.exists():
        raise FileNotFoundError(
            f"File informasi kelas tidak ditemukan: {INFO_PATH.name}"
        )

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)

    # Memastikan informasi kelas tersedia
    if not all(
        key in info
        for key in ["img_size", "kelas", "label_tampilan"]
    ):
        raise ValueError(
            "Struktur info_kelas.json tidak lengkap."
        )

    return info


# ==========================================
# 5. HEADER APLIKASI
# ==========================================

st.title("🌿 Deteksi Penyakit Daun Tanaman")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        font-size:15px;
        line-height:1.8;
        margin-bottom:20px;
    ">
        Sistem klasifikasi kondisi daun tanaman
        menggunakan teknologi
        <b>Custom Convolutional Neural Network (CNN)</b>.
        <br>
        Unggah gambar daun untuk mendapatkan hasil prediksi.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 6. AREA UPLOAD GAMBAR
# ==========================================

st.subheader("📷 Upload Gambar Daun")

st.write(
    "Silakan pilih gambar daun tanaman "
    "dengan format JPG, JPEG, atau PNG."
)

uploaded_file = st.file_uploader(
    "Pilih gambar daun tanaman",
    type=["jpg", "jpeg", "png"],
    help="Unggah gambar daun yang ingin diklasifikasikan."
)


# ==========================================
# 7. MENAMPILKAN GAMBAR
# ==========================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Gambar daun yang diunggah",
            use_container_width=True
        )

        st.divider()

        # ======================================
        # 8. TOMBOL ANALISIS
        # ======================================

        if st.button(
            "🔍 Analisis Gambar",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("🌱 Sedang menganalisis gambar..."):

                # Memuat model dan informasi kelas
                model = load_model()
                info_kelas = load_class_info()

                # Mengambil ukuran gambar
                img_size = tuple(info_kelas["img_size"])

                # Memastikan jumlah kelas sesuai output model
                nama_kelas = info_kelas["kelas"]
                label_tampilan = info_kelas["label_tampilan"]

                if len(nama_kelas) != model.output_shape[-1]:
                    raise ValueError(
                        "Jumlah kelas pada JSON tidak sesuai "
                        "dengan jumlah output model."
                    )

                # Mengubah ukuran gambar
                image_resize = image.resize(
                    (img_size[1], img_size[0])
                )

                # Mengubah gambar menjadi array
                image_array = np.array(
                    image_resize,
                    dtype=np.float32
                )

                # Menambahkan dimensi batch
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Prediksi model CNN
                # Tidak membagi 255 lagi jika model
                # sudah memiliki layer Rescaling(1/255).
                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                # Mengambil indeks prediksi tertinggi
                predicted_index = int(
                    np.argmax(prediction[0])
                )

                # Mengambil hasil prediksi
                hasil_kelas = nama_kelas[predicted_index]

                hasil_prediksi = label_tampilan.get(
                    hasil_kelas,
                    hasil_kelas
                )

                # Menghitung tingkat keyakinan
                confidence = float(
                    prediction[0][predicted_index]
                ) * 100

            # ==================================
            # 9. MENAMPILKAN HASIL KLASIFIKASI
            # ==================================

            st.divider()

            st.subheader("📊 Hasil Klasifikasi")

            st.success(
                f"🌿 Prediksi: {hasil_prediksi}"
            )

            st.metric(
                label="Tingkat Keyakinan Model",
                value=f"{confidence:.2f}%"
            )

            # ==================================
            # 10. PROBABILITAS SETIAP KELAS
            # ==================================

            st.subheader("📈 Probabilitas Setiap Kelas")

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

                st.progress(
                    min(max(nilai, 0.0), 1.0)
                )

            # ==================================
            # 11. INFORMASI TAMBAHAN
            # ==================================

            st.info(
                "Hasil ini merupakan prediksi berdasarkan "
                "model CNN dan bukan diagnosis pasti "
                "penyakit tanaman."
            )

    except Exception as error:

        st.error(
            "Terjadi kesalahan saat memproses gambar."
        )

        st.exception(error)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        font-size:12px;
        line-height:1.8;
    ">
        🌿 <b>Deteksi Penyakit Daun Tanaman</b>
        <br>
        Custom Convolutional Neural Network (CNN)
        <br>
        Dataset PlantVillage
    </div>
    """,
    unsafe_allow_html=True
)
