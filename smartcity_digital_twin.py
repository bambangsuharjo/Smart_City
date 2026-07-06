# ============================================================
# Digital Twin Smart City
# File: smartcity_digital_twin.py
#
# Fitur:
# 1. Data contoh
# 2. Upload CSV
# 3. Link API / CSV online
# 4. Pilih kota/wilayah
# 5. Simulasi skenario:
#    - Pertumbuhan penduduk
#    - Efisiensi energi
#    - Efisiensi air
#    - Pengurangan sampah
#    - Peningkatan transportasi publik
#    - Intensitas kebijakan smart city
# 6. Proyeksi 1–10 tahun
# 7. Indeks tekanan kota
# 8. Indeks keberlanjutan
# 9. Tingkat risiko kota
# 10. Grafik baseline vs skenario
# 11. Radar indikator
# 12. Rekomendasi otomatis
#
# Jalankan:
# streamlit run smartcity_digital_twin.py
#
# Install:
# pip install streamlit pandas numpy matplotlib requests
# ============================================================

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
    page_title="Digital Twin Smart City",
    page_icon="🌐",
    layout="wide"
)


# ============================================================
# DATA CONTOH
# ============================================================
def buat_data_contoh():
    data = {
        "Tahun": [2020, 2021, 2022, 2023, 2024, 2025] * 5,
        "Kota": (
            ["Jakarta"] * 6
            + ["Bandung"] * 6
            + ["Surabaya"] * 6
            + ["Medan"] * 6
            + ["Makassar"] * 6
        ),
        "Jumlah Penduduk": [
            10560000, 10680000, 10800000, 10920000, 11050000, 11180000,
            2500000, 2530000, 2565000, 2600000, 2635000, 2670000,
            2900000, 2935000, 2970000, 3005000, 3040000, 3075000,
            2450000, 2480000, 2510000, 2540000, 2575000, 2610000,
            1450000, 1480000, 1510000, 1540000, 1580000, 1620000
        ],
        "Konsumsi Energi": [
            6200, 6350, 6510, 6680, 6900, 7150,
            1750, 1810, 1880, 1950, 2030, 2120,
            2300, 2380, 2470, 2560, 2660, 2770,
            1600, 1660, 1725, 1800, 1875, 1960,
            950, 990, 1035, 1085, 1140, 1200
        ],
        "Konsumsi Air": [
            590, 605, 620, 638, 655, 675,
            205, 211, 218, 225, 233, 241,
            255, 263, 271, 280, 290, 300,
            198, 204, 211, 219, 227, 236,
            120, 126, 132, 139, 146, 154
        ],
        "Volume Sampah": [
            7600, 7800, 8050, 8300, 8600, 8900,
            1650, 1720, 1800, 1880, 1970, 2070,
            2100, 2190, 2290, 2390, 2500, 2620,
            1500, 1570, 1640, 1720, 1810, 1900,
            870, 920, 970, 1030, 1090, 1160
        ],
        "Indeks Kemacetan": [
            75, 77, 79, 81, 83, 85,
            62, 64, 66, 68, 70, 72,
            65, 67, 69, 71, 73, 75,
            55, 57, 59, 61, 63, 65,
            45, 47, 49, 51, 53, 55
        ],
    }
    return pd.DataFrame(data)


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
# VALIDASI DATA
# ============================================================
def validasi_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

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
        st.error("Kolom berikut belum ada: " + ", ".join(kurang))
        st.info(
            "Format minimal: Tahun, Kota, Jumlah Penduduk, Konsumsi Energi, "
            "Konsumsi Air, Volume Sampah, Indeks Kemacetan."
        )
        st.stop()

    if "Kota" not in df.columns:
        df["Kota"] = "Semua Wilayah"

    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df = df.dropna(subset=["Tahun"])
    df["Tahun"] = df["Tahun"].astype(int)

    for col in kolom_wajib[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=kolom_wajib)
    return df


# ============================================================
# FUNGSI SIMULASI DIGITAL TWIN
# ============================================================
def hitung_cagr(series, tahun_series):
    series = np.array(series, dtype=float)
    tahun_series = np.array(tahun_series, dtype=float)

    if len(series) < 2:
        return 0.0

    awal = series[0]
    akhir = series[-1]
    n = tahun_series[-1] - tahun_series[0]

    if awal <= 0 or n <= 0:
        return 0.0

    return (akhir / awal) ** (1 / n) - 1


def proyeksi_baseline(df_kota, tahun_depan):
    indikator = [
        "Jumlah Penduduk",
        "Konsumsi Energi",
        "Konsumsi Air",
        "Volume Sampah",
        "Indeks Kemacetan",
    ]

    tahun_hist = df_kota["Tahun"].values
    tahun_akhir = int(df_kota["Tahun"].max())

    nilai_akhir = df_kota[df_kota["Tahun"] == tahun_akhir].iloc[-1]

    rows = []

    cagr = {}
    for ind in indikator:
        cagr[ind] = hitung_cagr(df_kota[ind].values, tahun_hist)

    for t in range(tahun_akhir + 1, tahun_akhir + tahun_depan + 1):
        langkah = t - tahun_akhir
        row = {"Tahun": t}

        for ind in indikator:
            row[ind] = nilai_akhir[ind] * ((1 + cagr[ind]) ** langkah)

        rows.append(row)

    return pd.DataFrame(rows), cagr


def proyeksi_skenario(
    baseline_df,
    pertumbuhan_penduduk,
    efisiensi_energi,
    efisiensi_air,
    pengurangan_sampah,
    perbaikan_transportasi,
    intensitas_kebijakan
):
    skenario = baseline_df.copy()

    kebijakan = intensitas_kebijakan / 100

    for i in range(len(skenario)):
        langkah = i + 1

        faktor_penduduk = (1 + pertumbuhan_penduduk / 100) ** langkah
        faktor_energi = (1 - (efisiensi_energi / 100) * kebijakan) ** langkah
        faktor_air = (1 - (efisiensi_air / 100) * kebijakan) ** langkah
        faktor_sampah = (1 - (pengurangan_sampah / 100) * kebijakan) ** langkah
        faktor_kemacetan = (1 - (perbaikan_transportasi / 100) * kebijakan) ** langkah

        skenario.loc[skenario.index[i], "Jumlah Penduduk"] *= faktor_penduduk
        skenario.loc[skenario.index[i], "Konsumsi Energi"] *= faktor_energi
        skenario.loc[skenario.index[i], "Konsumsi Air"] *= faktor_air
        skenario.loc[skenario.index[i], "Volume Sampah"] *= faktor_sampah
        skenario.loc[skenario.index[i], "Indeks Kemacetan"] *= faktor_kemacetan

    return skenario


def hitung_indeks(df):
    hasil = df.copy()

    # Normalisasi berbasis nilai maksimum internal skenario.
    # Semakin tinggi tekanan berarti beban kota semakin berat.
    penduduk_norm = hasil["Jumlah Penduduk"] / hasil["Jumlah Penduduk"].max()
    energi_norm = hasil["Konsumsi Energi"] / hasil["Konsumsi Energi"].max()
    air_norm = hasil["Konsumsi Air"] / hasil["Konsumsi Air"].max()
    sampah_norm = hasil["Volume Sampah"] / hasil["Volume Sampah"].max()
    kemacetan_norm = hasil["Indeks Kemacetan"] / 100

    hasil["Indeks Tekanan Kota"] = (
        0.20 * penduduk_norm
        + 0.20 * energi_norm
        + 0.20 * air_norm
        + 0.20 * sampah_norm
        + 0.20 * kemacetan_norm
    ) * 100

    hasil["Indeks Keberlanjutan"] = 100 - hasil["Indeks Tekanan Kota"]

    def level_risiko(x):
        if x >= 75:
            return "Sangat Tinggi"
        if x >= 60:
            return "Tinggi"
        if x >= 40:
            return "Sedang"
        return "Rendah"

    hasil["Level Risiko"] = hasil["Indeks Tekanan Kota"].apply(level_risiko)

    return hasil


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def buat_radar_chart(nilai_dict, title):
    labels = list(nilai_dict.keys())
    values = list(nilai_dict.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title(title)

    return fig


def normalisasi_untuk_radar(row):
    return {
        "Penduduk": min(row["Jumlah Penduduk"] / 12000000 * 100, 100),
        "Energi": min(row["Konsumsi Energi"] / 8000 * 100, 100),
        "Air": min(row["Konsumsi Air"] / 800 * 100, 100),
        "Sampah": min(row["Volume Sampah"] / 10000 * 100, 100),
        "Kemacetan": min(row["Indeks Kemacetan"], 100),
    }


def buat_rekomendasi(row):
    rekomendasi = []

    if row["Indeks Tekanan Kota"] >= 75:
        rekomendasi.append("- Prioritas utama: lakukan intervensi lintas sektor karena tekanan kota sangat tinggi.")
    elif row["Indeks Tekanan Kota"] >= 60:
        rekomendasi.append("- Tekanan kota tinggi. Perlu percepatan program efisiensi energi, air, sampah, dan transportasi.")
    elif row["Indeks Tekanan Kota"] >= 40:
        rekomendasi.append("- Tekanan kota sedang. Program Smart City perlu diarahkan untuk mencegah peningkatan risiko.")
    else:
        rekomendasi.append("- Tekanan kota relatif rendah. Fokus pada pemeliharaan kualitas layanan dan pencegahan risiko.")

    if row["Indeks Kemacetan"] >= 70:
        rekomendasi.append("- Perkuat transportasi publik, manajemen lalu lintas, integrasi rute, dan sistem informasi perjalanan.")
    if row["Volume Sampah"] >= 2000:
        rekomendasi.append("- Tingkatkan pengelolaan sampah berbasis 3R, pemilahan, rute pengangkutan cerdas, dan fasilitas daur ulang.")
    if row["Konsumsi Energi"] >= 2500:
        rekomendasi.append("- Dorong efisiensi energi, smart grid, penerangan jalan pintar, dan energi terbarukan.")
    if row["Konsumsi Air"] >= 250:
        rekomendasi.append("- Perkuat manajemen air, deteksi kebocoran, penghematan air, dan sistem monitoring distribusi.")
    if row["Jumlah Penduduk"] >= 3000000:
        rekomendasi.append("- Sesuaikan kapasitas infrastruktur kota dengan pertumbuhan penduduk dan pola mobilitas warga.")

    rekomendasi.append("- Gunakan simulasi ini sebagai dasar awal untuk diskusi kebijakan, bukan sebagai keputusan final.")

    return "\n".join(rekomendasi)


# ============================================================
# JUDUL
# ============================================================
st.title("🌐 Digital Twin Smart City")
st.markdown(
    "Aplikasi ini mensimulasikan kondisi kota digital untuk melihat dampak perubahan "
    "penduduk, energi, air, sampah, kemacetan, dan kebijakan Smart City terhadap risiko kota."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Pengaturan Digital Twin")

st.sidebar.subheader("1. Sumber Data")
sumber_data = st.sidebar.selectbox(
    "Pilih Sumber Data",
    ["📊 Data Contoh", "📁 Upload CSV", "🌐 Link API / CSV Online"]
)

if sumber_data == "📊 Data Contoh":
    df = buat_data_contoh()

elif sumber_data == "📁 Upload CSV":
    file_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file_csv is None:
        st.warning("Silakan upload file CSV.")
        st.stop()

    df = pd.read_csv(file_csv)

else:
    url = st.sidebar.text_input("Masukkan link API atau CSV online")

    if not url:
        st.info("Masukkan URL terlebih dahulu.")
        st.stop()

    try:
        df = ambil_data_online(url)
    except Exception as e:
        st.error(f"Gagal membaca data online: {e}")
        st.stop()


df = validasi_data(df)

st.sidebar.divider()
st.sidebar.subheader("2. Pilihan Kota")

kota_list = sorted(df["Kota"].dropna().astype(str).unique().tolist())

kota_pilih = st.sidebar.selectbox(
    "Pilih Kota / Wilayah",
    kota_list
)

tahun_depan = st.sidebar.slider(
    "Simulasi Berapa Tahun ke Depan",
    min_value=1,
    max_value=10,
    value=5
)


st.sidebar.divider()
st.sidebar.subheader("3. Skenario Kebijakan")

pertumbuhan_penduduk = st.sidebar.slider(
    "Tambahan Pertumbuhan Penduduk per Tahun (%)",
    min_value=-2.0,
    max_value=5.0,
    value=0.5,
    step=0.1
)

efisiensi_energi = st.sidebar.slider(
    "Efisiensi Energi per Tahun (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

efisiensi_air = st.sidebar.slider(
    "Efisiensi Air per Tahun (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

pengurangan_sampah = st.sidebar.slider(
    "Pengurangan Sampah per Tahun (%)",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.5
)

perbaikan_transportasi = st.sidebar.slider(
    "Perbaikan Transportasi / Penurunan Kemacetan per Tahun (%)",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.5
)

intensitas_kebijakan = st.sidebar.slider(
    "Intensitas Implementasi Smart City (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5
)


# ============================================================
# DATA KOTA DAN PROYEKSI
# ============================================================
df_kota = (
    df[df["Kota"].astype(str) == kota_pilih]
    .groupby("Tahun", as_index=False)[
        [
            "Jumlah Penduduk",
            "Konsumsi Energi",
            "Konsumsi Air",
            "Volume Sampah",
            "Indeks Kemacetan",
        ]
    ]
    .mean()
    .sort_values("Tahun")
)

if len(df_kota) < 2:
    st.warning("Data historis minimal membutuhkan dua tahun.")
    st.stop()

baseline_df, cagr_dict = proyeksi_baseline(df_kota, tahun_depan)

skenario_df = proyeksi_skenario(
    baseline_df=baseline_df,
    pertumbuhan_penduduk=pertumbuhan_penduduk,
    efisiensi_energi=efisiensi_energi,
    efisiensi_air=efisiensi_air,
    pengurangan_sampah=pengurangan_sampah,
    perbaikan_transportasi=perbaikan_transportasi,
    intensitas_kebijakan=intensitas_kebijakan
)

baseline_indeks = hitung_indeks(baseline_df)
skenario_indeks = hitung_indeks(skenario_df)

tahun_akhir = int(df_kota["Tahun"].max())
kondisi_terakhir = df_kota[df_kota["Tahun"] == tahun_akhir].iloc[-1]


# ============================================================
# KPI
# ============================================================
st.subheader(f"📌 Kondisi Digital Twin: {kota_pilih}")

akhir_skenario = skenario_indeks.iloc[-1]
akhir_baseline = baseline_indeks.iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tahun Simulasi Akhir", int(akhir_skenario["Tahun"]))
col2.metric("Tekanan Kota", f"{akhir_skenario['Indeks Tekanan Kota']:.1f}")
col3.metric("Keberlanjutan", f"{akhir_skenario['Indeks Keberlanjutan']:.1f}")
col4.metric("Level Risiko", akhir_skenario["Level Risiko"])
col5.metric(
    "Selisih Tekanan vs Baseline",
    f"{akhir_skenario['Indeks Tekanan Kota'] - akhir_baseline['Indeks Tekanan Kota']:.1f}"
)


# ============================================================
# TAB
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data",
    "📈 Simulasi",
    "📊 Indeks Risiko",
    "🕸️ Radar",
    "🤖 Rekomendasi",
])


# ============================================================
# TAB 1 DATA
# ============================================================
with tab1:
    st.subheader("📋 Data Historis Kota")
    st.dataframe(df_kota, use_container_width=True)

    st.subheader("📈 CAGR Historis")
    cagr_table = pd.DataFrame({
        "Indikator": list(cagr_dict.keys()),
        "CAGR Historis (%)": [v * 100 for v in cagr_dict.values()]
    })
    st.dataframe(cagr_table, use_container_width=True)

    st.download_button(
        "⬇️ Download Data Historis",
        data=df_kota.to_csv(index=False).encode("utf-8"),
        file_name="data_historis_digital_twin.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 2 SIMULASI
# ============================================================
with tab2:
    st.subheader("📈 Baseline vs Skenario")

    indikator_list = [
        "Jumlah Penduduk",
        "Konsumsi Energi",
        "Konsumsi Air",
        "Volume Sampah",
        "Indeks Kemacetan",
    ]

    indikator_pilih = st.selectbox(
        "Pilih Indikator Simulasi",
        indikator_list
    )

    fig_sim, ax_sim = plt.subplots(figsize=(10, 5))

    ax_sim.plot(df_kota["Tahun"], df_kota[indikator_pilih], marker="o", label="Historis")
    ax_sim.plot(baseline_df["Tahun"], baseline_df[indikator_pilih], marker="s", linestyle="--", label="Baseline")
    ax_sim.plot(skenario_df["Tahun"], skenario_df[indikator_pilih], marker="^", linestyle="--", label="Skenario")

    ax_sim.set_title(f"Simulasi {indikator_pilih}: Historis vs Baseline vs Skenario")
    ax_sim.set_xlabel("Tahun")
    ax_sim.set_ylabel(indikator_pilih)
    ax_sim.grid(True)
    ax_sim.legend()

    st.pyplot(fig_sim)

    st.download_button(
        "⬇️ Download Grafik Simulasi",
        data=fig_to_png(fig_sim),
        file_name="simulasi_digital_twin.png",
        mime="image/png"
    )

    st.subheader("Tabel Skenario")
    skenario_output = skenario_indeks.copy()
    st.dataframe(skenario_output, use_container_width=True)

    st.download_button(
        "⬇️ Download Hasil Skenario CSV",
        data=skenario_output.to_csv(index=False).encode("utf-8"),
        file_name="hasil_skenario_digital_twin.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 3 INDEKS RISIKO
# ============================================================
with tab3:
    st.subheader("📊 Indeks Tekanan Kota dan Keberlanjutan")

    fig_idx, ax_idx = plt.subplots(figsize=(10, 5))

    ax_idx.plot(
        baseline_indeks["Tahun"],
        baseline_indeks["Indeks Tekanan Kota"],
        marker="o",
        label="Tekanan Baseline"
    )
    ax_idx.plot(
        skenario_indeks["Tahun"],
        skenario_indeks["Indeks Tekanan Kota"],
        marker="s",
        label="Tekanan Skenario"
    )
    ax_idx.plot(
        skenario_indeks["Tahun"],
        skenario_indeks["Indeks Keberlanjutan"],
        marker="^",
        label="Keberlanjutan Skenario"
    )

    ax_idx.set_title("Indeks Tekanan Kota dan Indeks Keberlanjutan")
    ax_idx.set_xlabel("Tahun")
    ax_idx.set_ylabel("Indeks")
    ax_idx.set_ylim(0, 100)
    ax_idx.grid(True)
    ax_idx.legend()

    st.pyplot(fig_idx)

    st.download_button(
        "⬇️ Download Grafik Indeks",
        data=fig_to_png(fig_idx),
        file_name="indeks_digital_twin.png",
        mime="image/png"
    )

    st.subheader("Interpretasi Risiko")
    st.dataframe(
        skenario_indeks[["Tahun", "Indeks Tekanan Kota", "Indeks Keberlanjutan", "Level Risiko"]],
        use_container_width=True
    )


# ============================================================
# TAB 4 RADAR
# ============================================================
with tab4:
    st.subheader("🕸️ Radar Kondisi Kota")

    kondisi_hist_radar = normalisasi_untuk_radar(kondisi_terakhir)
    kondisi_skenario_radar = normalisasi_untuk_radar(akhir_skenario)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_radar1 = buat_radar_chart(
            kondisi_hist_radar,
            f"Kondisi Historis Terakhir {tahun_akhir}"
        )
        st.pyplot(fig_radar1)

    with col_b:
        fig_radar2 = buat_radar_chart(
            kondisi_skenario_radar,
            f"Kondisi Skenario {int(akhir_skenario['Tahun'])}"
        )
        st.pyplot(fig_radar2)


# ============================================================
# TAB 5 REKOMENDASI
# ============================================================
with tab5:
    st.subheader("🤖 Rekomendasi Otomatis")

    rekomendasi = buat_rekomendasi(akhir_skenario)
    st.markdown(rekomendasi)

    st.subheader("📌 Ringkasan Skenario")
    st.markdown(
        f"""
        - Kota/wilayah yang disimulasikan: **{kota_pilih}**
        - Periode simulasi: **{tahun_akhir + 1}–{int(akhir_skenario['Tahun'])}**
        - Intensitas kebijakan Smart City: **{intensitas_kebijakan}%**
        - Level risiko akhir: **{akhir_skenario['Level Risiko']}**
        - Indeks tekanan kota akhir: **{akhir_skenario['Indeks Tekanan Kota']:.2f}**
        - Indeks keberlanjutan akhir: **{akhir_skenario['Indeks Keberlanjutan']:.2f}**
        """
    )


# ============================================================
# PENJELASAN
# ============================================================
st.divider()
st.subheader("📘 Konsep Digital Twin Smart City")

st.markdown(
    """
    **Digital Twin Smart City** adalah representasi digital dari kondisi kota nyata.
    Melalui data historis dan simulasi skenario, pemerintah dapat menguji dampak
    kebijakan sebelum diterapkan di lapangan.

    Dalam contoh ini, Digital Twin digunakan untuk menjawab pertanyaan seperti:

    - Apa dampak pertumbuhan penduduk terhadap energi, air, sampah, dan kemacetan?
    - Apa pengaruh efisiensi energi dan air terhadap keberlanjutan kota?
    - Bagaimana pengurangan sampah dan perbaikan transportasi menurunkan tekanan kota?
    - Seberapa besar risiko kota dalam beberapa tahun ke depan?

    **Format CSV yang disarankan:**
    """
)

contoh_format = pd.DataFrame({
    "Tahun": [2024, 2025],
    "Kota": ["Kota Bandung", "Kota Bandung"],
    "Jumlah Penduduk": [2635000, 2670000],
    "Konsumsi Energi": [2030, 2120],
    "Konsumsi Air": [233, 241],
    "Volume Sampah": [1970, 2070],
    "Indeks Kemacetan": [70, 72],
})

st.dataframe(contoh_format, use_container_width=True)
