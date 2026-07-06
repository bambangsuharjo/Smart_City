# ============================================================
# Analisis Sentimen Smart City
# File: smartcity_sentiment.py
#
# Fitur:
# 1. Data contoh
# 2. Upload CSV
# 3. Link API / CSV online
# 4. Input teks manual
# 5. Analisis sentimen berbasis kamus sederhana
# 6. Klasifikasi sentimen: Positif, Netral, Negatif
# 7. Analisis isu Smart City:
#    - Transportasi
#    - Kemacetan
#    - Sampah
#    - Air
#    - Energi
#    - Banjir
#    - Keamanan
#    - Pelayanan Publik
# 8. Statistik sentimen
# 9. Grafik batang
# 10. Grafik tren waktu
# 11. Word frequency
# 12. Tabel teks negatif prioritas
# 13. Smart insight otomatis
#
# Jalankan:
# streamlit run smartcity_sentiment.py
#
# Install:
# pip install streamlit pandas numpy matplotlib requests
# ============================================================

import re
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Smart City",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# DATA CONTOH
# ============================================================
def buat_data_contoh():
    data = [
        ["2025-01-05", "Jakarta", "Transportasi umum sekarang semakin nyaman dan terintegrasi."],
        ["2025-01-10", "Jakarta", "Kemacetan pagi ini sangat parah dan membuat perjalanan terlambat."],
        ["2025-02-03", "Bandung", "Aplikasi layanan publik kota sangat membantu warga."],
        ["2025-02-12", "Bandung", "Sampah menumpuk di beberapa titik dan belum segera diangkut."],
        ["2025-03-01", "Surabaya", "Lampu jalan pintar membantu keamanan lingkungan pada malam hari."],
        ["2025-03-15", "Surabaya", "Kualitas air menurun dan warga mengeluhkan bau tidak sedap."],
        ["2025-04-04", "Medan", "Pelayanan pengaduan online cepat dan responsif."],
        ["2025-04-21", "Medan", "Banjir masih sering terjadi saat hujan deras."],
        ["2025-05-05", "Makassar", "Program kota hijau sangat baik untuk kenyamanan warga."],
        ["2025-05-18", "Makassar", "Biaya energi rumah tangga terasa meningkat."],
        ["2025-06-07", "Jakarta", "CCTV dan penerangan jalan membuat lingkungan lebih aman."],
        ["2025-06-22", "Bandung", "Transportasi publik masih penuh dan kurang nyaman."],
        ["2025-07-03", "Surabaya", "Pengelolaan sampah mulai membaik setelah ada sistem pelaporan digital."],
        ["2025-07-19", "Medan", "Jalan rusak membuat lalu lintas tidak lancar."],
        ["2025-08-02", "Makassar", "Aplikasi smart city mudah digunakan dan informatif."],
        ["2025-08-20", "Jakarta", "Polusi dan kemacetan semakin mengganggu aktivitas harian."],
    ]

    return pd.DataFrame(data, columns=["Tanggal", "Kota", "Teks"])


# ============================================================
# KAMUS SENTIMEN SEDERHANA
# ============================================================
KATA_POSITIF = {
    "baik", "bagus", "nyaman", "membantu", "cepat", "responsif",
    "aman", "mudah", "informatif", "terintegrasi", "bersih",
    "lancar", "membaik", "efektif", "efisien", "puas",
    "modern", "cerdas", "hijau", "tertib", "ramah"
}

KATA_NEGATIF = {
    "buruk", "parah", "macet", "kemacetan", "terlambat", "menumpuk",
    "sampah", "banjir", "rusak", "penuh", "kurang", "tidak",
    "menurun", "bau", "mahal", "meningkat", "polusi", "mengganggu",
    "lambat", "rawan", "kotor", "sulit", "gagal", "keluhan",
    "mengeluhkan", "krisis"
}

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan",
    "ini", "itu", "sangat", "masih", "ada", "atau", "dalam",
    "sebagai", "karena", "saat", "lebih", "agar", "oleh", "warga",
    "kota", "publik", "smart", "city"
}

KATEGORI_ISU = {
    "Transportasi": ["transportasi", "angkutan", "bus", "kereta", "halte", "terminal", "ojek"],
    "Kemacetan": ["macet", "kemacetan", "lalu lintas", "jalan", "terlambat"],
    "Sampah": ["sampah", "limbah", "angkut", "tpa", "daur ulang", "kotor"],
    "Air": ["air", "pdam", "bau", "kualitas air", "krisis air"],
    "Energi": ["energi", "listrik", "lampu", "penerangan", "pln"],
    "Banjir": ["banjir", "hujan", "drainase", "genangan"],
    "Keamanan": ["aman", "keamanan", "cctv", "rawan", "kriminal"],
    "Pelayanan Publik": ["layanan", "pelayanan", "pengaduan", "aplikasi", "online", "responsif"],
}


# ============================================================
# FUNGSI DATA ONLINE
# ============================================================
def ambil_data_online(url):
    if url.lower().endswith(".csv"):
        return pd.read_csv(url)

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    json_data = response.json()

    if isinstance(json_data, list):
        return pd.DataFrame(json_data)

    if isinstance(json_data, dict):
        for value in json_data.values():
            if isinstance(value, list):
                return pd.DataFrame(value)
        return pd.DataFrame([json_data])

    raise ValueError("Format API tidak dapat dibaca menjadi tabel.")


# ============================================================
# PREPROCESSING DAN ANALISIS SENTIMEN
# ============================================================
def bersihkan_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+", " ", teks)
    teks = re.sub(r"@\w+|#\w+", " ", teks)
    teks = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def tokenisasi(teks):
    teks = bersihkan_teks(teks)
    tokens = [t for t in teks.split() if t not in STOPWORDS and len(t) > 2]
    return tokens


def hitung_sentimen(teks):
    tokens = tokenisasi(teks)

    skor_pos = sum(1 for t in tokens if t in KATA_POSITIF)
    skor_neg = sum(1 for t in tokens if t in KATA_NEGATIF)

    skor = skor_pos - skor_neg

    if skor > 0:
        label = "Positif"
    elif skor < 0:
        label = "Negatif"
    else:
        label = "Netral"

    return skor, label, skor_pos, skor_neg


def deteksi_isu(teks):
    teks_bersih = bersihkan_teks(teks)
    isu_terdeteksi = []

    for isu, kata_kunci in KATEGORI_ISU.items():
        for kata in kata_kunci:
            if kata in teks_bersih:
                isu_terdeteksi.append(isu)
                break

    if not isu_terdeteksi:
        return "Lainnya"

    return ", ".join(isu_terdeteksi)


def validasi_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Teks" not in df.columns:
        st.error("CSV harus memiliki kolom `Teks`.")
        st.info("Kolom yang disarankan: Tanggal, Kota, Teks")
        st.stop()

    if "Tanggal" not in df.columns:
        df["Tanggal"] = pd.Timestamp.today().strftime("%Y-%m-%d")

    if "Kota" not in df.columns:
        df["Kota"] = "Semua Wilayah"

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df["Tanggal"] = df["Tanggal"].fillna(pd.Timestamp.today())

    df["Kota"] = df["Kota"].astype(str)
    df["Teks"] = df["Teks"].astype(str)

    return df


def analisis_sentimen_df(df):
    df = df.copy()

    hasil = df["Teks"].apply(hitung_sentimen)

    df["Skor Sentimen"] = hasil.apply(lambda x: x[0])
    df["Sentimen"] = hasil.apply(lambda x: x[1])
    df["Jumlah Kata Positif"] = hasil.apply(lambda x: x[2])
    df["Jumlah Kata Negatif"] = hasil.apply(lambda x: x[3])
    df["Isu Smart City"] = df["Teks"].apply(deteksi_isu)

    return df


def hitung_frekuensi_kata(df):
    semua_tokens = []

    for teks in df["Teks"]:
        semua_tokens.extend(tokenisasi(teks))

    freq = pd.Series(semua_tokens).value_counts().reset_index()
    freq.columns = ["Kata", "Frekuensi"]

    return freq


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def buat_insight(df):
    insight = []

    total = len(df)
    if total == 0:
        return "Tidak ada data untuk dianalisis."

    sentimen_count = df["Sentimen"].value_counts()
    pos = sentimen_count.get("Positif", 0)
    neg = sentimen_count.get("Negatif", 0)
    net = sentimen_count.get("Netral", 0)

    persen_pos = pos / total * 100
    persen_neg = neg / total * 100
    persen_net = net / total * 100

    insight.append(
        f"- Dari **{total}** teks yang dianalisis, sentimen positif sebesar "
        f"**{persen_pos:.1f}%**, netral **{persen_net:.1f}%**, dan negatif **{persen_neg:.1f}%**."
    )

    if persen_neg > 40:
        insight.append(
            "- Proporsi sentimen negatif cukup tinggi, sehingga diperlukan respons kebijakan dan komunikasi publik yang lebih aktif."
        )
    elif persen_pos > 50:
        insight.append(
            "- Sentimen publik cenderung positif, menunjukkan penerimaan yang baik terhadap layanan Smart City."
        )
    else:
        insight.append(
            "- Sentimen publik relatif berimbang, sehingga perlu analisis lebih dalam terhadap isu utama yang muncul."
        )

    isu_top = df["Isu Smart City"].value_counts().head(3)

    if not isu_top.empty:
        isu_text = ", ".join([f"{idx} ({val})" for idx, val in isu_top.items()])
        insight.append(f"- Isu yang paling banyak muncul adalah: **{isu_text}**.")

    negatif = df[df["Sentimen"] == "Negatif"]
    if not negatif.empty:
        isu_neg = negatif["Isu Smart City"].value_counts().head(1)
        if not isu_neg.empty:
            insight.append(
                f"- Isu negatif paling menonjol adalah **{isu_neg.index[0]}**, "
                "sehingga dapat menjadi prioritas penanganan."
            )

    insight.append(
        "- Rekomendasi: gunakan hasil analisis ini sebagai masukan awal untuk memetakan persepsi publik, "
        "menentukan prioritas layanan, dan menyusun strategi komunikasi pemerintah daerah."
    )

    return "\n".join(insight)


# ============================================================
# JUDUL
# ============================================================
st.title("💬 Analisis Sentimen Smart City")
st.markdown(
    "Aplikasi ini digunakan untuk menganalisis opini publik tentang layanan kota, "
    "transportasi, kemacetan, sampah, air, energi, banjir, keamanan, dan pelayanan publik."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Pengaturan Sentimen")

sumber_data = st.sidebar.selectbox(
    "Pilih Sumber Data",
    ["📊 Data Contoh", "📁 Upload CSV", "🌐 Link API / CSV Online", "✍️ Input Teks Manual"]
)

if sumber_data == "📊 Data Contoh":
    df = buat_data_contoh()

elif sumber_data == "📁 Upload CSV":
    file_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file_csv is None:
        st.warning("Silakan upload file CSV.")
        st.stop()

    df = pd.read_csv(file_csv)

elif sumber_data == "🌐 Link API / CSV Online":
    url = st.sidebar.text_input("Masukkan link API atau CSV online")

    if not url:
        st.info("Masukkan URL terlebih dahulu.")
        st.stop()

    try:
        df = ambil_data_online(url)
    except Exception as e:
        st.error(f"Gagal membaca data online: {e}")
        st.stop()

else:
    teks_manual = st.text_area(
        "Masukkan teks opini publik",
        height=180,
        placeholder="Contoh: Transportasi umum semakin baik, tetapi kemacetan masih parah."
    )

    kota_manual = st.text_input("Kota/Wilayah", value="Kota Contoh")

    if not teks_manual:
        st.info("Masukkan teks terlebih dahulu.")
        st.stop()

    df = pd.DataFrame({
        "Tanggal": [pd.Timestamp.today()],
        "Kota": [kota_manual],
        "Teks": [teks_manual],
    })


df = validasi_data(df)
df_hasil = analisis_sentimen_df(df)

st.sidebar.divider()
st.sidebar.subheader("Filter")

kota_list = sorted(df_hasil["Kota"].dropna().astype(str).unique().tolist())
kota_pilih = st.sidebar.multiselect(
    "Pilih Kota / Wilayah",
    kota_list,
    default=kota_list
)

sentimen_pilih = st.sidebar.multiselect(
    "Pilih Sentimen",
    ["Positif", "Netral", "Negatif"],
    default=["Positif", "Netral", "Negatif"]
)

isu_list = sorted(df_hasil["Isu Smart City"].dropna().astype(str).unique().tolist())
isu_pilih = st.sidebar.multiselect(
    "Pilih Isu Smart City",
    isu_list,
    default=isu_list
)

tanggal_min = df_hasil["Tanggal"].min().date()
tanggal_max = df_hasil["Tanggal"].max().date()

rentang_tanggal = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(tanggal_min, tanggal_max)
)

if isinstance(rentang_tanggal, tuple) and len(rentang_tanggal) == 2:
    start_date, end_date = rentang_tanggal
else:
    start_date, end_date = tanggal_min, tanggal_max

df_filter = df_hasil[
    (df_hasil["Kota"].astype(str).isin(kota_pilih)) &
    (df_hasil["Sentimen"].isin(sentimen_pilih)) &
    (df_hasil["Isu Smart City"].isin(isu_pilih)) &
    (df_hasil["Tanggal"].dt.date >= start_date) &
    (df_hasil["Tanggal"].dt.date <= end_date)
].copy()

if df_filter.empty:
    st.warning("Data tidak tersedia untuk filter yang dipilih.")
    st.stop()


# ============================================================
# KPI
# ============================================================
st.subheader("📌 Ringkasan Sentimen")

total = len(df_filter)
jumlah_pos = int((df_filter["Sentimen"] == "Positif").sum())
jumlah_net = int((df_filter["Sentimen"] == "Netral").sum())
jumlah_neg = int((df_filter["Sentimen"] == "Negatif").sum())
rata_skor = df_filter["Skor Sentimen"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Teks", f"{total:,}")
col2.metric("Positif", f"{jumlah_pos:,}")
col3.metric("Netral", f"{jumlah_net:,}")
col4.metric("Negatif", f"{jumlah_neg:,}")
col5.metric("Rata-rata Skor", f"{rata_skor:.2f}")


# ============================================================
# TAB
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data",
    "📊 Statistik Sentimen",
    "📈 Tren",
    "🔤 Kata Kunci",
    "🤖 Insight",
])


# ============================================================
# TAB DATA
# ============================================================
with tab1:
    st.subheader("📋 Data Hasil Analisis Sentimen")
    st.dataframe(df_filter, use_container_width=True)

    st.download_button(
        "⬇️ Download Hasil Analisis CSV",
        data=df_filter.to_csv(index=False).encode("utf-8"),
        file_name="hasil_sentimen_smartcity.csv",
        mime="text/csv"
    )

    st.subheader("⚠️ Teks Negatif Prioritas")
    negatif = df_filter[df_filter["Sentimen"] == "Negatif"].copy()
    negatif = negatif.sort_values("Skor Sentimen")

    if negatif.empty:
        st.success("Tidak ada teks negatif pada filter ini.")
    else:
        st.dataframe(
            negatif[["Tanggal", "Kota", "Teks", "Skor Sentimen", "Isu Smart City"]],
            use_container_width=True
        )


# ============================================================
# TAB STATISTIK SENTIMEN
# ============================================================
with tab2:
    st.subheader("📊 Distribusi Sentimen")

    distribusi = df_filter["Sentimen"].value_counts().reindex(
        ["Positif", "Netral", "Negatif"]
    ).fillna(0).astype(int)

    st.dataframe(distribusi.reset_index().rename(columns={"index": "Sentimen", "Sentimen": "Jumlah"}))

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    ax_bar.bar(distribusi.index, distribusi.values)
    ax_bar.set_title("Distribusi Sentimen")
    ax_bar.set_xlabel("Sentimen")
    ax_bar.set_ylabel("Jumlah Teks")
    st.pyplot(fig_bar)

    st.download_button(
        "⬇️ Download Grafik Sentimen",
        data=fig_to_png(fig_bar),
        file_name="grafik_sentimen_smartcity.png",
        mime="image/png"
    )

    st.subheader("📊 Distribusi Isu Smart City")

    isu_count = df_filter["Isu Smart City"].value_counts()

    fig_isu, ax_isu = plt.subplots(figsize=(10, 5))
    ax_isu.bar(isu_count.index, isu_count.values)
    ax_isu.set_title("Distribusi Isu Smart City")
    ax_isu.set_xlabel("Isu")
    ax_isu.set_ylabel("Jumlah Teks")
    ax_isu.tick_params(axis="x", rotation=30)
    st.pyplot(fig_isu)


# ============================================================
# TAB TREN
# ============================================================
with tab3:
    st.subheader("📈 Tren Sentimen Waktu")

    df_filter["Bulan"] = df_filter["Tanggal"].dt.to_period("M").astype(str)

    tren = (
        df_filter
        .groupby(["Bulan", "Sentimen"])
        .size()
        .reset_index(name="Jumlah")
    )

    pivot_tren = tren.pivot(index="Bulan", columns="Sentimen", values="Jumlah").fillna(0)

    st.dataframe(pivot_tren, use_container_width=True)

    fig_tren, ax_tren = plt.subplots(figsize=(10, 5))
    for col in pivot_tren.columns:
        ax_tren.plot(pivot_tren.index, pivot_tren[col], marker="o", label=col)

    ax_tren.set_title("Tren Sentimen Bulanan")
    ax_tren.set_xlabel("Bulan")
    ax_tren.set_ylabel("Jumlah Teks")
    ax_tren.tick_params(axis="x", rotation=30)
    ax_tren.grid(True)
    ax_tren.legend()
    st.pyplot(fig_tren)

    st.download_button(
        "⬇️ Download Grafik Tren Sentimen",
        data=fig_to_png(fig_tren),
        file_name="tren_sentimen_smartcity.png",
        mime="image/png"
    )


# ============================================================
# TAB KATA KUNCI
# ============================================================
with tab4:
    st.subheader("🔤 Frekuensi Kata")

    freq = hitung_frekuensi_kata(df_filter)

    if freq.empty:
        st.warning("Tidak ada kata yang dapat dihitung.")
    else:
        top_n = st.slider("Jumlah kata teratas", min_value=5, max_value=30, value=15)

        st.dataframe(freq.head(top_n), use_container_width=True)

        fig_word, ax_word = plt.subplots(figsize=(10, 5))
        ax_word.bar(freq.head(top_n)["Kata"], freq.head(top_n)["Frekuensi"])
        ax_word.set_title("Frekuensi Kata Teratas")
        ax_word.set_xlabel("Kata")
        ax_word.set_ylabel("Frekuensi")
        ax_word.tick_params(axis="x", rotation=45)
        st.pyplot(fig_word)

        st.download_button(
            "⬇️ Download Grafik Kata Kunci",
            data=fig_to_png(fig_word),
            file_name="kata_kunci_sentimen_smartcity.png",
            mime="image/png"
        )


# ============================================================
# TAB INSIGHT
# ============================================================
with tab5:
    st.subheader("🤖 Smart Insight Otomatis")

    insight = buat_insight(df_filter)
    st.markdown(insight)

    st.subheader("📌 Rekomendasi Tindak Lanjut")

    st.markdown(
        """
        1. **Sentimen negatif** perlu diprioritaskan sebagai sinyal awal masalah layanan kota.
        2. **Isu yang paling sering muncul** dapat dijadikan dasar agenda kebijakan.
        3. **Tren sentimen bulanan** dapat digunakan untuk mengevaluasi dampak program pemerintah.
        4. **Teks negatif prioritas** dapat menjadi bahan awal untuk sistem pengaduan dan respons cepat.
        """
    )


# ============================================================
# FORMAT CSV
# ============================================================
st.divider()
st.subheader("🧾 Format CSV yang Disarankan")

contoh_format = pd.DataFrame({
    "Tanggal": ["2025-01-01", "2025-01-02"],
    "Kota": ["Kota Bandung", "Kota Bandung"],
    "Teks": [
        "Transportasi umum semakin nyaman dan terintegrasi.",
        "Sampah menumpuk dan belum segera diangkut."
    ],
})

st.dataframe(contoh_format, use_container_width=True)

st.markdown(
    """
    **Catatan:**
    - Kolom wajib: `Teks`.
    - Kolom opsional tetapi disarankan: `Tanggal`, `Kota`.
    - Analisis ini memakai pendekatan kamus sederhana agar mudah dipahami pembaca.
    - Untuk versi lanjutan, model dapat dikembangkan dengan Naive Bayes, SVM, IndoBERT, atau LLM.
    """
)
