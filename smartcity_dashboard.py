# ============================================================
# Dashboard Smart City Sederhana Versi 2.0
# File: smartcity_dashboard_v2.py
# Komponen: Python, Pandas, Matplotlib, Streamlit
# ============================================================

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="Dashboard Smart City",
    page_icon="🏙️",
    layout="wide"
)


def buat_data_contoh():
    data = {
        "Tahun": [2020, 2021, 2022, 2023, 2024, 2025,
                  2020, 2021, 2022, 2023, 2024, 2025],
        "Kota": ["Kota A"] * 6 + ["Kota B"] * 6,
        "Jumlah Penduduk": [1200000, 1235000, 1268000, 1302000, 1340000, 1385000,
                             900000, 925000, 948000, 980000, 1015000, 1048000],
        "Konsumsi Energi": [420, 445, 470, 495, 530, 560,
                             310, 330, 350, 372, 395, 420],
        "Konsumsi Air": [82, 86, 91, 95, 101, 106,
                          65, 68, 71, 75, 79, 83],
        "Volume Sampah": [650, 680, 710, 745, 785, 820,
                           480, 505, 530, 555, 585, 610],
        "Indeks Kemacetan": [55, 58, 61, 64, 67, 70,
                              42, 45, 47, 50, 53, 56],
    }
    return pd.DataFrame(data)


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


def bersihkan_kolom(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pastikan_kolom(df):
    kolom_wajib = [
        "Tahun",
        "Jumlah Penduduk",
        "Konsumsi Energi",
        "Konsumsi Air",
        "Volume Sampah",
        "Indeks Kemacetan",
    ]

    kurang = [k for k in kolom_wajib if k not in df.columns]
    if kurang:
        st.error("Kolom berikut belum ada pada data: " + ", ".join(kurang))
        st.info(
            "Format kolom minimal: Tahun, Jumlah Penduduk, Konsumsi Energi, "
            "Konsumsi Air, Volume Sampah, Indeks Kemacetan. Kolom Kota bersifat opsional."
        )
        st.stop()

    for k in kolom_wajib:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    if "Kota" not in df.columns:
        df["Kota"] = "Semua Wilayah"

    df = df.dropna(subset=["Tahun"])
    df["Tahun"] = df["Tahun"].astype(int)
    return df


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


st.title("🏙️ Dashboard Smart City Sederhana")
st.markdown(
    "Dashboard ini menampilkan indikator utama Smart City: **jumlah penduduk, "
    "konsumsi energi, konsumsi air, volume sampah, dan indeks kemacetan**."
)


# ============================================================
# PENGATURAN DASHBOARD
# ============================================================
st.sidebar.header("⚙️ Pengaturan Dashboard")

st.sidebar.subheader("1. Pilih Sumber Data")
sumber_data = st.sidebar.selectbox(
    "Sumber Data",
    [
        "📊 Data Contoh",
        "📁 Upload CSV",
        "🌐 Link API / CSV Online",
    ],
    index=0
)

if sumber_data == "📊 Data Contoh":
    df = buat_data_contoh()

elif sumber_data == "📁 Upload CSV":
    file_csv = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

    if file_csv is None:
        st.warning("Silakan upload file CSV pada bagian Pengaturan Dashboard.")
        st.stop()

    df = pd.read_csv(file_csv)

else:
    url = st.sidebar.text_input(
        "Masukkan link API atau CSV online",
        placeholder="https://contoh.go.id/data.csv"
    )

    if not url:
        st.info("Masukkan link API atau CSV online terlebih dahulu.")
        st.stop()

    try:
        df = ambil_data_online(url)
    except Exception as e:
        st.error(f"Data online gagal dibaca: {e}")
        st.stop()


df = bersihkan_kolom(df)
df = pastikan_kolom(df)

indikator_list = [
    "Jumlah Penduduk",
    "Konsumsi Energi",
    "Konsumsi Air",
    "Volume Sampah",
    "Indeks Kemacetan",
]

st.sidebar.divider()
st.sidebar.subheader("2. Pilih Indikator")

indikator_dipilih = st.sidebar.multiselect(
    "Indikator yang Ditampilkan",
    indikator_list,
    default=indikator_list
)

if not indikator_dipilih:
    st.warning("Pilih minimal satu indikator.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("3. Filter Data")

kota_list = sorted(df["Kota"].dropna().astype(str).unique().tolist())
kota_dipilih = st.sidebar.multiselect(
    "Kota / Wilayah",
    kota_list,
    default=kota_list
)

tahun_min = int(df["Tahun"].min())
tahun_max = int(df["Tahun"].max())

rentang_tahun = st.sidebar.slider(
    "Rentang Tahun",
    min_value=tahun_min,
    max_value=tahun_max,
    value=(tahun_min, tahun_max)
)

tema = st.sidebar.selectbox("Tema Tampilan", ["Terang", "Gelap"])

st.sidebar.info(
    "Satuan indikator:\n\n"
    "- Penduduk: jiwa\n"
    "- Energi: GWh\n"
    "- Air: juta m³\n"
    "- Sampah: ton/hari\n"
    "- Kemacetan: indeks 0–100"
)


df_filter = df[
    (df["Kota"].astype(str).isin(kota_dipilih)) &
    (df["Tahun"] >= rentang_tahun[0]) &
    (df["Tahun"] <= rentang_tahun[1])
].copy()

if df_filter.empty:
    st.warning("Data tidak tersedia untuk filter yang dipilih.")
    st.stop()

if tema == "Gelap":
    st.markdown(
        '''
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )


# ============================================================
# KPI CARD
# ============================================================
st.subheader("📌 Ringkasan Indikator Utama")

nilai_terakhir = df_filter.sort_values("Tahun").groupby("Kota", as_index=False).tail(1)
rata_terakhir = nilai_terakhir[indikator_list].mean(numeric_only=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Penduduk", f"{rata_terakhir['Jumlah Penduduk']:,.0f}")
col2.metric("Energi", f"{rata_terakhir['Konsumsi Energi']:,.1f} GWh")
col3.metric("Air", f"{rata_terakhir['Konsumsi Air']:,.1f} juta m³")
col4.metric("Sampah", f"{rata_terakhir['Volume Sampah']:,.1f} ton/hari")
col5.metric("Kemacetan", f"{rata_terakhir['Indeks Kemacetan']:,.1f}")


# ============================================================
# TABEL DAN STATISTIK
# ============================================================
st.subheader("📋 Tabel Data Smart City")
st.dataframe(df_filter, use_container_width=True)

csv_download = df_filter.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Data CSV",
    data=csv_download,
    file_name="data_smartcity_terfilter.csv",
    mime="text/csv"
)

st.subheader("📊 Statistik Deskriptif")
st.dataframe(df_filter[indikator_list].describe().T, use_container_width=True)


# ============================================================
# GRAFIK BATANG
# ============================================================
st.subheader("📊 Grafik Batang")

indikator_batang = st.selectbox(
    "Pilih indikator untuk grafik batang",
    indikator_dipilih
)

data_batang = (
    df_filter
    .groupby("Kota", as_index=False)[indikator_batang]
    .mean()
    .sort_values(indikator_batang, ascending=False)
)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(data_batang["Kota"], data_batang[indikator_batang])
ax1.set_title(f"Rata-rata {indikator_batang} per Kota/Wilayah")
ax1.set_xlabel("Kota / Wilayah")
ax1.set_ylabel(indikator_batang)
ax1.tick_params(axis="x", rotation=30)
st.pyplot(fig1)

st.download_button(
    "⬇️ Download Grafik Batang PNG",
    data=fig_to_png(fig1),
    file_name="grafik_batang_smartcity.png",
    mime="image/png"
)


# ============================================================
# GRAFIK GARIS
# ============================================================
st.subheader("📈 Grafik Garis Tren Tahunan")

indikator_garis = st.selectbox(
    "Pilih indikator untuk grafik garis",
    indikator_dipilih,
    key="indikator_garis"
)

data_garis = (
    df_filter
    .groupby("Tahun", as_index=False)[indikator_garis]
    .mean()
    .sort_values("Tahun")
)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(data_garis["Tahun"], data_garis[indikator_garis], marker="o")
ax2.set_title(f"Tren {indikator_garis} per Tahun")
ax2.set_xlabel("Tahun")
ax2.set_ylabel(indikator_garis)
ax2.grid(True)
st.pyplot(fig2)

st.download_button(
    "⬇️ Download Grafik Garis PNG",
    data=fig_to_png(fig2),
    file_name="grafik_garis_smartcity.png",
    mime="image/png"
)


# ============================================================
# PIE CHART
# ============================================================
st.subheader("🥧 Pie Chart Komposisi Indikator")

indikator_pie = st.selectbox(
    "Pilih indikator untuk pie chart",
    indikator_dipilih,
    key="indikator_pie"
)

data_pie = df_filter.groupby("Kota", as_index=False)[indikator_pie].mean()

fig3, ax3 = plt.subplots(figsize=(7, 7))
ax3.pie(
    data_pie[indikator_pie],
    labels=data_pie["Kota"],
    autopct="%1.1f%%",
    startangle=90
)
ax3.set_title(f"Komposisi {indikator_pie} per Kota/Wilayah")
st.pyplot(fig3)

st.download_button(
    "⬇️ Download Pie Chart PNG",
    data=fig_to_png(fig3),
    file_name="pie_chart_smartcity.png",
    mime="image/png"
)


# ============================================================
# FORMAT CSV
# ============================================================
st.subheader("🧾 Format CSV yang Disarankan")

contoh_format = pd.DataFrame({
    "Tahun": [2024, 2025],
    "Kota": ["Kota A", "Kota A"],
    "Jumlah Penduduk": [1340000, 1385000],
    "Konsumsi Energi": [530, 560],
    "Konsumsi Air": [101, 106],
    "Volume Sampah": [785, 820],
    "Indeks Kemacetan": [67, 70],
})

st.dataframe(contoh_format, use_container_width=True)

st.markdown(
    '''
    **Catatan:**
    - Nama kolom harus sama seperti contoh.
    - Kolom `Kota` dapat digunakan untuk membandingkan beberapa wilayah.
    - Link online dapat berupa file `.csv` atau API JSON sederhana.
    '''
)
