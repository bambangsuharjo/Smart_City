# ============================================================
# AI Decision Support System Smart City
# File: smartcity_dss.py
#
# Fitur:
# 1. Data contoh
# 2. Upload CSV
# 3. Link API / CSV online
# 4. Bobot indikator berbasis AHP sederhana / manual weighting
# 5. Smart City Risk Score
# 6. Smart City Readiness Score
# 7. Ranking wilayah prioritas
# 8. Matriks prioritas kebijakan
# 9. Analisis skenario kebijakan
# 10. Rekomendasi keputusan otomatis
# 11. Dashboard ringkasan
# 12. Download hasil keputusan CSV
#
# Jalankan:
# streamlit run smartcity_dss.py
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
    page_title="AI DSS Smart City",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# DATA CONTOH
# ============================================================
def buat_data_contoh():
    data = {
        "Tahun": [2025] * 8,
        "Kota": [
            "Jakarta", "Bandung", "Surabaya", "Medan",
            "Makassar", "Semarang", "Denpasar", "Yogyakarta"
        ],
        "Jumlah Penduduk": [
            11180000, 2670000, 3075000, 2610000,
            1620000, 1690000, 750000, 420000
        ],
        "Konsumsi Energi": [
            7150, 2120, 2770, 1960,
            1200, 1320, 780, 520
        ],
        "Konsumsi Air": [
            675, 241, 300, 236,
            154, 165, 90, 70
        ],
        "Volume Sampah": [
            8900, 2070, 2620, 1900,
            1160, 1250, 700, 420
        ],
        "Indeks Kemacetan": [
            85, 72, 75, 65,
            55, 60, 58, 50
        ],
        "Kualitas Udara": [
            68, 55, 60, 58,
            48, 52, 42, 40
        ],
        "Indeks Layanan Digital": [
            78, 72, 74, 65,
            62, 68, 70, 76
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
        "Kota",
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

    # Kolom opsional dibuat jika belum ada
    if "Kualitas Udara" not in df.columns:
        df["Kualitas Udara"] = 50

    if "Indeks Layanan Digital" not in df.columns:
        df["Indeks Layanan Digital"] = 60

    kolom_numerik = [
        "Tahun",
        "Jumlah Penduduk",
        "Konsumsi Energi",
        "Konsumsi Air",
        "Volume Sampah",
        "Indeks Kemacetan",
        "Kualitas Udara",
        "Indeks Layanan Digital",
    ]

    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=kolom_wajib)
    df["Tahun"] = df["Tahun"].astype(int)
    df["Kota"] = df["Kota"].astype(str)

    return df


# ============================================================
# NORMALISASI DAN SKOR DSS
# ============================================================
def normalisasi_minmax(series):
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        return pd.Series([50] * len(s), index=s.index)

    return (s - min_val) / (max_val - min_val) * 100


def hitung_dss(df, bobot):
    hasil = df.copy()

    # Indikator tekanan kota: semakin tinggi semakin berisiko
    hasil["N_Penduduk"] = normalisasi_minmax(hasil["Jumlah Penduduk"])
    hasil["N_Energi"] = normalisasi_minmax(hasil["Konsumsi Energi"])
    hasil["N_Air"] = normalisasi_minmax(hasil["Konsumsi Air"])
    hasil["N_Sampah"] = normalisasi_minmax(hasil["Volume Sampah"])
    hasil["N_Kemacetan"] = normalisasi_minmax(hasil["Indeks Kemacetan"])
    hasil["N_Udara"] = normalisasi_minmax(hasil["Kualitas Udara"])

    # Layanan digital adalah faktor kesiapan, sehingga semakin tinggi semakin baik.
    hasil["N_Digital"] = normalisasi_minmax(hasil["Indeks Layanan Digital"])

    total_bobot_risiko = (
        bobot["Penduduk"]
        + bobot["Energi"]
        + bobot["Air"]
        + bobot["Sampah"]
        + bobot["Kemacetan"]
        + bobot["Udara"]
    )

    if total_bobot_risiko == 0:
        total_bobot_risiko = 1

    hasil["Smart City Risk Score"] = (
        bobot["Penduduk"] * hasil["N_Penduduk"]
        + bobot["Energi"] * hasil["N_Energi"]
        + bobot["Air"] * hasil["N_Air"]
        + bobot["Sampah"] * hasil["N_Sampah"]
        + bobot["Kemacetan"] * hasil["N_Kemacetan"]
        + bobot["Udara"] * hasil["N_Udara"]
    ) / total_bobot_risiko

    # Readiness: kombinasi layanan digital dan kebalikan risiko
    hasil["Smart City Readiness Score"] = (
        0.55 * hasil["N_Digital"]
        + 0.45 * (100 - hasil["Smart City Risk Score"])
    )

    def level_risiko(x):
        if x >= 75:
            return "Sangat Tinggi"
        if x >= 60:
            return "Tinggi"
        if x >= 40:
            return "Sedang"
        return "Rendah"

    def prioritas(x):
        if x >= 75:
            return "Prioritas 1"
        if x >= 60:
            return "Prioritas 2"
        if x >= 40:
            return "Prioritas 3"
        return "Prioritas 4"

    hasil["Level Risiko"] = hasil["Smart City Risk Score"].apply(level_risiko)
    hasil["Prioritas Kebijakan"] = hasil["Smart City Risk Score"].apply(prioritas)

    return hasil


# ============================================================
# REKOMENDASI OTOMATIS
# ============================================================
def rekomendasi_wilayah(row):
    rekom = []

    if row["Level Risiko"] == "Sangat Tinggi":
        rekom.append("Intervensi lintas sektor harus segera dilakukan.")
    elif row["Level Risiko"] == "Tinggi":
        rekom.append("Perlu percepatan program prioritas Smart City.")
    elif row["Level Risiko"] == "Sedang":
        rekom.append("Perlu penguatan kebijakan pencegahan risiko.")
    else:
        rekom.append("Pertahankan kinerja dan lakukan monitoring berkala.")

    indikator_dominan = {
        "Penduduk": row["N_Penduduk"],
        "Energi": row["N_Energi"],
        "Air": row["N_Air"],
        "Sampah": row["N_Sampah"],
        "Kemacetan": row["N_Kemacetan"],
        "Udara": row["N_Udara"],
    }

    top = sorted(indikator_dominan.items(), key=lambda x: x[1], reverse=True)[:3]

    for nama, nilai in top:
        if nama == "Penduduk" and nilai >= 60:
            rekom.append("Perkuat perencanaan tata ruang, hunian, dan kapasitas infrastruktur dasar.")
        elif nama == "Energi" and nilai >= 60:
            rekom.append("Prioritaskan efisiensi energi, smart grid, dan energi terbarukan.")
        elif nama == "Air" and nilai >= 60:
            rekom.append("Perkuat sistem monitoring air, deteksi kebocoran, dan manajemen distribusi.")
        elif nama == "Sampah" and nilai >= 60:
            rekom.append("Bangun sistem pengelolaan sampah cerdas, 3R, dan optimasi rute pengangkutan.")
        elif nama == "Kemacetan" and nilai >= 60:
            rekom.append("Perkuat transportasi publik, manajemen lalu lintas adaptif, dan integrasi moda.")
        elif nama == "Udara" and nilai >= 60:
            rekom.append("Tingkatkan pengendalian polusi, ruang hijau, dan pemantauan kualitas udara.")

    if row["N_Digital"] < 50:
        rekom.append("Tingkatkan kesiapan digital melalui aplikasi layanan publik, integrasi data, dan command center.")

    return " ".join(rekom)


def buat_insight_kota(hasil):
    if hasil.empty:
        return "Tidak ada data untuk dianalisis."

    insight = []

    rata_risiko = hasil["Smart City Risk Score"].mean()
    rata_readiness = hasil["Smart City Readiness Score"].mean()

    insight.append(
        f"- Rata-rata **Smart City Risk Score** adalah **{rata_risiko:.2f}**."
    )
    insight.append(
        f"- Rata-rata **Smart City Readiness Score** adalah **{rata_readiness:.2f}**."
    )

    kota_prioritas = hasil.sort_values("Smart City Risk Score", ascending=False).iloc[0]
    insight.append(
        f"- Wilayah prioritas tertinggi adalah **{kota_prioritas['Kota']}** "
        f"dengan risiko **{kota_prioritas['Smart City Risk Score']:.2f}** "
        f"dan level **{kota_prioritas['Level Risiko']}**."
    )

    kota_siaps = hasil.sort_values("Smart City Readiness Score", ascending=False).iloc[0]
    insight.append(
        f"- Wilayah dengan kesiapan Smart City tertinggi adalah **{kota_siaps['Kota']}** "
        f"dengan skor **{kota_siaps['Smart City Readiness Score']:.2f}**."
    )

    jumlah_p1 = int((hasil["Prioritas Kebijakan"] == "Prioritas 1").sum())
    jumlah_p2 = int((hasil["Prioritas Kebijakan"] == "Prioritas 2").sum())

    insight.append(
        f"- Terdapat **{jumlah_p1}** wilayah Prioritas 1 dan **{jumlah_p2}** wilayah Prioritas 2."
    )

    insight.append(
        "- Rekomendasi umum: fokuskan anggaran dan program pada wilayah Prioritas 1–2, "
        "terutama pada indikator dominan penyebab risiko."
    )

    return "\n".join(insight)


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


# ============================================================
# SKENARIO KEBIJAKAN
# ============================================================
def terapkan_skenario(df, efisiensi_energi, efisiensi_air, reduksi_sampah,
                      reduksi_macet, peningkatan_digital, perbaikan_udara):
    skenario = df.copy()

    skenario["Konsumsi Energi"] = skenario["Konsumsi Energi"] * (1 - efisiensi_energi / 100)
    skenario["Konsumsi Air"] = skenario["Konsumsi Air"] * (1 - efisiensi_air / 100)
    skenario["Volume Sampah"] = skenario["Volume Sampah"] * (1 - reduksi_sampah / 100)
    skenario["Indeks Kemacetan"] = skenario["Indeks Kemacetan"] * (1 - reduksi_macet / 100)
    skenario["Kualitas Udara"] = skenario["Kualitas Udara"] * (1 - perbaikan_udara / 100)
    skenario["Indeks Layanan Digital"] = skenario["Indeks Layanan Digital"] * (1 + peningkatan_digital / 100)
    skenario["Indeks Layanan Digital"] = skenario["Indeks Layanan Digital"].clip(upper=100)

    return skenario


# ============================================================
# JUDUL
# ============================================================
st.title("🧭 AI Decision Support System Smart City")
st.markdown(
    "Aplikasi ini membantu pengambilan keputusan Smart City dengan menggabungkan "
    "indikator kota menjadi **Risk Score**, **Readiness Score**, prioritas kebijakan, "
    "analisis skenario, dan rekomendasi otomatis."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Pengaturan DSS")

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
st.sidebar.subheader("2. Filter Data")

tahun_list = sorted(df["Tahun"].dropna().unique().tolist())
tahun_pilih = st.sidebar.selectbox("Pilih Tahun", tahun_list, index=len(tahun_list)-1)

kota_list = sorted(df["Kota"].dropna().astype(str).unique().tolist())
kota_pilih = st.sidebar.multiselect(
    "Pilih Kota/Wilayah",
    kota_list,
    default=kota_list
)

df_filter = df[
    (df["Tahun"] == tahun_pilih) &
    (df["Kota"].astype(str).isin(kota_pilih))
].copy()

if df_filter.empty:
    st.warning("Data tidak tersedia untuk filter yang dipilih.")
    st.stop()


st.sidebar.divider()
st.sidebar.subheader("3. Bobot Risiko")

metode_bobot = st.sidebar.radio(
    "Metode Bobot",
    ["Bobot Seimbang", "Bobot Manual"],
    index=0
)

if metode_bobot == "Bobot Seimbang":
    bobot = {
        "Penduduk": 1,
        "Energi": 1,
        "Air": 1,
        "Sampah": 1,
        "Kemacetan": 1,
        "Udara": 1,
    }
else:
    bobot = {
        "Penduduk": st.sidebar.slider("Bobot Penduduk", 0.0, 5.0, 1.0, 0.1),
        "Energi": st.sidebar.slider("Bobot Energi", 0.0, 5.0, 1.0, 0.1),
        "Air": st.sidebar.slider("Bobot Air", 0.0, 5.0, 1.0, 0.1),
        "Sampah": st.sidebar.slider("Bobot Sampah", 0.0, 5.0, 1.0, 0.1),
        "Kemacetan": st.sidebar.slider("Bobot Kemacetan", 0.0, 5.0, 1.0, 0.1),
        "Udara": st.sidebar.slider("Bobot Kualitas Udara", 0.0, 5.0, 1.0, 0.1),
    }


st.sidebar.divider()
st.sidebar.subheader("4. Skenario Kebijakan")

aktifkan_skenario = st.sidebar.checkbox("Aktifkan Analisis Skenario", value=True)

efisiensi_energi = st.sidebar.slider("Efisiensi Energi (%)", 0, 30, 10)
efisiensi_air = st.sidebar.slider("Efisiensi Air (%)", 0, 30, 8)
reduksi_sampah = st.sidebar.slider("Reduksi Sampah (%)", 0, 40, 15)
reduksi_macet = st.sidebar.slider("Reduksi Kemacetan (%)", 0, 40, 12)
peningkatan_digital = st.sidebar.slider("Peningkatan Layanan Digital (%)", 0, 50, 15)
perbaikan_udara = st.sidebar.slider("Perbaikan Kualitas Udara / Penurunan Polusi (%)", 0, 40, 10)


# ============================================================
# HITUNG DSS
# ============================================================
hasil = hitung_dss(df_filter, bobot)
hasil["Rekomendasi Keputusan"] = hasil.apply(rekomendasi_wilayah, axis=1)

if aktifkan_skenario:
    df_skenario = terapkan_skenario(
        df_filter,
        efisiensi_energi=efisiensi_energi,
        efisiensi_air=efisiensi_air,
        reduksi_sampah=reduksi_sampah,
        reduksi_macet=reduksi_macet,
        peningkatan_digital=peningkatan_digital,
        perbaikan_udara=perbaikan_udara
    )

    hasil_skenario = hitung_dss(df_skenario, bobot)

    perbandingan = hasil[["Kota", "Smart City Risk Score", "Smart City Readiness Score"]].merge(
        hasil_skenario[["Kota", "Smart City Risk Score", "Smart City Readiness Score"]],
        on="Kota",
        suffixes=("_Baseline", "_Skenario")
    )

    perbandingan["Perubahan Risiko"] = (
        perbandingan["Smart City Risk Score_Skenario"]
        - perbandingan["Smart City Risk Score_Baseline"]
    )

    perbandingan["Perubahan Readiness"] = (
        perbandingan["Smart City Readiness Score_Skenario"]
        - perbandingan["Smart City Readiness Score_Baseline"]
    )


# ============================================================
# KPI
# ============================================================
st.subheader("📌 Dashboard Keputusan Smart City")

rata_risiko = hasil["Smart City Risk Score"].mean()
rata_readiness = hasil["Smart City Readiness Score"].mean()
jumlah_prioritas1 = int((hasil["Prioritas Kebijakan"] == "Prioritas 1").sum())
jumlah_prioritas2 = int((hasil["Prioritas Kebijakan"] == "Prioritas 2").sum())
wilayah_tertinggi = hasil.sort_values("Smart City Risk Score", ascending=False).iloc[0]["Kota"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rata-rata Risiko", f"{rata_risiko:.2f}")
col2.metric("Rata-rata Readiness", f"{rata_readiness:.2f}")
col3.metric("Prioritas 1", jumlah_prioritas1)
col4.metric("Prioritas 2", jumlah_prioritas2)
col5.metric("Risiko Tertinggi", wilayah_tertinggi)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data & Skor",
    "🏆 Ranking",
    "📊 Visualisasi",
    "🔮 Skenario",
    "🤖 Rekomendasi",
])


# ============================================================
# TAB 1 DATA
# ============================================================
with tab1:
    st.subheader("📋 Data dan Skor DSS")

    kolom_tampil = [
        "Tahun",
        "Kota",
        "Jumlah Penduduk",
        "Konsumsi Energi",
        "Konsumsi Air",
        "Volume Sampah",
        "Indeks Kemacetan",
        "Kualitas Udara",
        "Indeks Layanan Digital",
        "Smart City Risk Score",
        "Smart City Readiness Score",
        "Level Risiko",
        "Prioritas Kebijakan",
        "Rekomendasi Keputusan",
    ]

    st.dataframe(hasil[kolom_tampil], use_container_width=True)

    st.download_button(
        "⬇️ Download Hasil DSS CSV",
        data=hasil[kolom_tampil].to_csv(index=False).encode("utf-8"),
        file_name="hasil_dss_smartcity.csv",
        mime="text/csv"
    )

    st.subheader("Bobot yang Digunakan")
    st.dataframe(
        pd.DataFrame({
            "Indikator": list(bobot.keys()),
            "Bobot": list(bobot.values())
        }),
        use_container_width=True
    )


# ============================================================
# TAB 2 RANKING
# ============================================================
with tab2:
    st.subheader("🏆 Ranking Wilayah Prioritas")

    ranking_risiko = hasil.sort_values("Smart City Risk Score", ascending=False)
    ranking_readiness = hasil.sort_values("Smart City Readiness Score", ascending=False)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Ranking Risiko Tertinggi")
        st.dataframe(
            ranking_risiko[["Kota", "Smart City Risk Score", "Level Risiko", "Prioritas Kebijakan"]],
            use_container_width=True
        )

    with col_b:
        st.markdown("### Ranking Kesiapan Tertinggi")
        st.dataframe(
            ranking_readiness[["Kota", "Smart City Readiness Score", "Level Risiko", "Prioritas Kebijakan"]],
            use_container_width=True
        )

    st.subheader("Matriks Prioritas Kebijakan")

    matriks = hasil.groupby(["Level Risiko", "Prioritas Kebijakan"]).size().reset_index(name="Jumlah Wilayah")
    st.dataframe(matriks, use_container_width=True)


# ============================================================
# TAB 3 VISUALISASI
# ============================================================
with tab3:
    st.subheader("📊 Visualisasi DSS")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    data_plot = hasil.sort_values("Smart City Risk Score", ascending=False)
    ax1.bar(data_plot["Kota"], data_plot["Smart City Risk Score"])
    ax1.set_title("Smart City Risk Score per Wilayah")
    ax1.set_xlabel("Kota/Wilayah")
    ax1.set_ylabel("Risk Score")
    ax1.tick_params(axis="x", rotation=30)
    st.pyplot(fig1)

    st.download_button(
        "⬇️ Download Grafik Risk Score",
        data=fig_to_png(fig1),
        file_name="risk_score_smartcity.png",
        mime="image/png"
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.scatter(
        hasil["Smart City Risk Score"],
        hasil["Smart City Readiness Score"]
    )

    for _, row in hasil.iterrows():
        ax2.text(
            row["Smart City Risk Score"],
            row["Smart City Readiness Score"],
            row["Kota"],
            fontsize=8
        )

    ax2.set_title("Matriks Risiko vs Kesiapan Smart City")
    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Readiness Score")
    ax2.grid(True)
    st.pyplot(fig2)

    st.download_button(
        "⬇️ Download Matriks Risiko Readiness",
        data=fig_to_png(fig2),
        file_name="risk_readiness_matrix.png",
        mime="image/png"
    )


# ============================================================
# TAB 4 SKENARIO
# ============================================================
with tab4:
    st.subheader("🔮 Analisis Skenario Kebijakan")

    if not aktifkan_skenario:
        st.info("Aktifkan analisis skenario pada sidebar.")
    else:
        st.markdown(
            f"""
            **Skenario yang diterapkan:**
            - Efisiensi energi: **{efisiensi_energi}%**
            - Efisiensi air: **{efisiensi_air}%**
            - Reduksi sampah: **{reduksi_sampah}%**
            - Reduksi kemacetan: **{reduksi_macet}%**
            - Peningkatan layanan digital: **{peningkatan_digital}%**
            - Perbaikan kualitas udara: **{perbaikan_udara}%**
            """
        )

        st.dataframe(perbandingan, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(perbandingan["Kota"]))
        width = 0.35

        ax3.bar(
            x - width / 2,
            perbandingan["Smart City Risk Score_Baseline"],
            width,
            label="Baseline"
        )
        ax3.bar(
            x + width / 2,
            perbandingan["Smart City Risk Score_Skenario"],
            width,
            label="Skenario"
        )

        ax3.set_title("Perbandingan Risk Score Baseline vs Skenario")
        ax3.set_xlabel("Kota/Wilayah")
        ax3.set_ylabel("Risk Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(perbandingan["Kota"], rotation=30)
        ax3.legend()

        st.pyplot(fig3)

        st.download_button(
            "⬇️ Download Grafik Skenario",
            data=fig_to_png(fig3),
            file_name="skenario_dss_smartcity.png",
            mime="image/png"
        )

        st.download_button(
            "⬇️ Download Hasil Skenario CSV",
            data=perbandingan.to_csv(index=False).encode("utf-8"),
            file_name="hasil_skenario_dss_smartcity.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 5 REKOMENDASI
# ============================================================
with tab5:
    st.subheader("🤖 Rekomendasi Keputusan Otomatis")

    insight = buat_insight_kota(hasil)
    st.markdown(insight)

    st.subheader("Detail Rekomendasi per Wilayah")

    for _, row in hasil.sort_values("Smart City Risk Score", ascending=False).iterrows():
        with st.expander(
            f"{row['Kota']} | Risiko: {row['Smart City Risk Score']:.2f} | {row['Prioritas Kebijakan']}"
        ):
            st.markdown(f"**Level Risiko:** {row['Level Risiko']}")
            st.markdown(f"**Readiness Score:** {row['Smart City Readiness Score']:.2f}")
            st.markdown("**Rekomendasi:**")
            st.write(row["Rekomendasi Keputusan"])


# ============================================================
# PENJELASAN
# ============================================================
st.divider()
st.subheader("📘 Konsep AI Decision Support System Smart City")

st.markdown(
    """
    **AI Decision Support System (DSS) Smart City** adalah sistem pendukung keputusan
    yang menggabungkan data kota, bobot indikator, penilaian risiko, dan rekomendasi
    kebijakan untuk membantu pemerintah menentukan prioritas pembangunan.

    Dalam contoh ini:

    - **Smart City Risk Score** menunjukkan tingkat tekanan/risiko wilayah.
    - **Smart City Readiness Score** menunjukkan kesiapan kota dalam mengelola tantangan.
    - **Prioritas Kebijakan** membantu menentukan wilayah yang perlu diintervensi lebih dahulu.
    - **Analisis Skenario** memperlihatkan dampak kebijakan sebelum diterapkan.

    **Format CSV yang disarankan:**
    """
)

contoh_format = pd.DataFrame({
    "Tahun": [2025, 2025],
    "Kota": ["Kota Bandung", "Kota Surabaya"],
    "Jumlah Penduduk": [2670000, 3075000],
    "Konsumsi Energi": [2120, 2770],
    "Konsumsi Air": [241, 300],
    "Volume Sampah": [2070, 2620],
    "Indeks Kemacetan": [72, 75],
    "Kualitas Udara": [55, 60],
    "Indeks Layanan Digital": [72, 74],
})

st.dataframe(contoh_format, use_container_width=True)
