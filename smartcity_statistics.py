# ============================================================
# Smart City Statistical Analytics
# File: smartcity_statistics.py
#
# Fitur:
# 1. Data contoh
# 2. Upload CSV
# 3. Link API / CSV online
# 4. Statistik deskriptif lengkap
# 5. Distribusi data: histogram dan boxplot
# 6. Korelasi Pearson, Spearman, Kendall
# 7. Outlier detection: IQR dan Z-Score
# 8. Analisis tren: growth rate, YoY, CAGR, moving average
# 9. Ranking wilayah
# 10. Smart Insight otomatis berbasis aturan statistik sederhana
#
# Jalankan:
# streamlit run smartcity_statistics.py
# ============================================================

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


try:
    from scipy import stats
except Exception:
    stats = None


st.set_page_config(
    page_title="Smart City Statistical Analytics",
    page_icon="📊",
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


# ============================================================
# STATISTIK DESKRIPTIF LANJUTAN
# ============================================================
def statistik_lengkap(df, variabel):
    hasil = []

    for var in variabel:
        x = df[var].dropna()

        if len(x) == 0:
            continue

        mode_value = x.mode()
        mode_value = mode_value.iloc[0] if len(mode_value) > 0 else np.nan

        mean = x.mean()
        std = x.std(ddof=1)
        cv = (std / mean) * 100 if mean != 0 else np.nan

        hasil.append({
            "Variabel": var,
            "Jumlah Data": x.count(),
            "Minimum": x.min(),
            "Maksimum": x.max(),
            "Mean": mean,
            "Median": x.median(),
            "Modus": mode_value,
            "Variance": x.var(ddof=1),
            "Standar Deviasi": std,
            "Range": x.max() - x.min(),
            "Q1": x.quantile(0.25),
            "Q3": x.quantile(0.75),
            "IQR": x.quantile(0.75) - x.quantile(0.25),
            "Skewness": x.skew(),
            "Kurtosis": x.kurtosis(),
            "Coefficient of Variation (%)": cv,
        })

    return pd.DataFrame(hasil)


# ============================================================
# OUTLIER
# ============================================================
def deteksi_outlier_iqr(df, variabel):
    hasil = []

    for var in variabel:
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1

        batas_bawah = q1 - 1.5 * iqr
        batas_atas = q3 + 1.5 * iqr

        outlier = df[(df[var] < batas_bawah) | (df[var] > batas_atas)]

        hasil.append({
            "Variabel": var,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Batas Bawah": batas_bawah,
            "Batas Atas": batas_atas,
            "Jumlah Outlier": len(outlier),
        })

    return pd.DataFrame(hasil)


def deteksi_outlier_zscore(df, variabel, threshold=3.0):
    hasil = []
    detail = []

    for var in variabel:
        x = df[var].dropna()
        mean = x.mean()
        std = x.std(ddof=0)

        if std == 0:
            continue

        z = (df[var] - mean) / std
        outlier_mask = z.abs() > threshold
        outlier = df[outlier_mask].copy()
        outlier["Variabel Outlier"] = var
        outlier["Z-Score"] = z[outlier_mask]

        hasil.append({
            "Variabel": var,
            "Threshold": threshold,
            "Jumlah Outlier": int(outlier_mask.sum()),
        })

        detail.append(outlier)

    ringkasan = pd.DataFrame(hasil)
    detail_df = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return ringkasan, detail_df


# ============================================================
# ANALISIS TREN
# ============================================================
def analisis_tren(df, variabel):
    data = df.groupby("Tahun", as_index=False)[variabel].mean().sort_values("Tahun")
    data["Growth Rate (%)"] = data[variabel].pct_change() * 100
    data["Moving Average"] = data[variabel].rolling(window=3, min_periods=1).mean()

    if len(data) >= 2:
        awal = data[variabel].iloc[0]
        akhir = data[variabel].iloc[-1]
        n = data["Tahun"].iloc[-1] - data["Tahun"].iloc[0]

        if awal > 0 and n > 0:
            cagr = ((akhir / awal) ** (1 / n) - 1) * 100
        else:
            cagr = np.nan
    else:
        cagr = np.nan

    return data, cagr


# ============================================================
# SMART INSIGHT
# ============================================================
def buat_smart_insight(df, variabel):
    insight = []

    for var in variabel:
        data_tren, cagr = analisis_tren(df, var)

        if not np.isnan(cagr):
            if cagr > 5:
                status = "meningkat cukup tinggi"
            elif cagr > 0:
                status = "meningkat secara moderat"
            elif cagr < 0:
                status = "menurun"
            else:
                status = "stabil"

            insight.append(
                f"- **{var}** {status} dengan CAGR sekitar **{cagr:.2f}% per tahun**."
            )

    if len(variabel) >= 2:
        corr = df[variabel].corr(method="pearson")
        pasangan_terkuat = None
        nilai_terkuat = 0

        for i in range(len(variabel)):
            for j in range(i + 1, len(variabel)):
                nilai = corr.iloc[i, j]
                if pd.notna(nilai) and abs(nilai) > abs(nilai_terkuat):
                    nilai_terkuat = nilai
                    pasangan_terkuat = (variabel[i], variabel[j])

        if pasangan_terkuat:
            arah = "positif" if nilai_terkuat > 0 else "negatif"
            insight.append(
                f"- Korelasi terkuat adalah antara **{pasangan_terkuat[0]}** "
                f"dan **{pasangan_terkuat[1]}** dengan nilai **{nilai_terkuat:.2f}** "
                f"atau hubungan **{arah}**."
            )

    if "Indeks Kemacetan" in df.columns:
        kota_macet = (
            df.groupby("Kota")["Indeks Kemacetan"]
            .mean()
            .sort_values(ascending=False)
            .head(1)
        )

        if not kota_macet.empty:
            insight.append(
                f"- Wilayah dengan rata-rata indeks kemacetan tertinggi adalah "
                f"**{kota_macet.index[0]}**."
            )

    if "Volume Sampah" in df.columns:
        kota_sampah = (
            df.groupby("Kota")["Volume Sampah"]
            .mean()
            .sort_values(ascending=False)
            .head(1)
        )

        if not kota_sampah.empty:
            insight.append(
                f"- Wilayah dengan rata-rata volume sampah tertinggi adalah "
                f"**{kota_sampah.index[0]}**."
            )

    insight.append(
        "- Rekomendasi umum: pemerintah daerah perlu memprioritaskan indikator "
        "yang mengalami pertumbuhan tinggi dan memiliki korelasi kuat dengan indikator lain."
    )

    return "\n".join(insight)


# ============================================================
# JUDUL
# ============================================================
st.title("📊 Smart City Statistical Analytics")
st.markdown(
    "Aplikasi ini digunakan untuk menganalisis data Smart City secara statistik, "
    "meliputi statistik deskriptif, distribusi, korelasi, outlier, tren, ranking wilayah, "
    "dan smart insight otomatis."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Pengaturan")

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
st.sidebar.subheader("2. Filter Data")

kota_list = sorted(df["Kota"].dropna().astype(str).unique().tolist())
kota_dipilih = st.sidebar.multiselect(
    "Pilih Kota / Wilayah",
    kota_list,
    default=kota_list
)

tahun_min = int(df["Tahun"].min())
tahun_max = int(df["Tahun"].max())

rentang_tahun = st.sidebar.slider(
    "Pilih Rentang Tahun",
    min_value=tahun_min,
    max_value=tahun_max,
    value=(tahun_min, tahun_max)
)

variabel_dipilih = st.sidebar.multiselect(
    "Pilih Variabel Analisis",
    indikator_list,
    default=indikator_list
)

if not variabel_dipilih:
    st.warning("Pilih minimal satu variabel.")
    st.stop()

df_filter = df[
    (df["Kota"].astype(str).isin(kota_dipilih)) &
    (df["Tahun"] >= rentang_tahun[0]) &
    (df["Tahun"] <= rentang_tahun[1])
].copy()

if df_filter.empty:
    st.warning("Data tidak tersedia untuk filter yang dipilih.")
    st.stop()


# ============================================================
# TAB MENU
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Data",
    "📊 Statistik Deskriptif",
    "📈 Distribusi",
    "🔗 Korelasi",
    "⚠️ Outlier",
    "📉 Tren",
    "🏙️ Ranking & Insight",
])


# ============================================================
# TAB 1 DATA
# ============================================================
with tab1:
    st.subheader("📋 Data Smart City")
    st.dataframe(df_filter, use_container_width=True)

    csv_download = df_filter.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data Terfilter",
        data=csv_download,
        file_name="smartcity_data_terfilter.csv",
        mime="text/csv"
    )

    st.subheader("Struktur Data")
    struktur = pd.DataFrame({
        "Kolom": df_filter.columns,
        "Tipe Data": [str(df_filter[c].dtype) for c in df_filter.columns],
        "Jumlah Kosong": [df_filter[c].isna().sum() for c in df_filter.columns],
    })
    st.dataframe(struktur, use_container_width=True)


# ============================================================
# TAB 2 STATISTIK DESKRIPTIF
# ============================================================
with tab2:
    st.subheader("📊 Statistik Deskriptif Lengkap")
    stat_df = statistik_lengkap(df_filter, variabel_dipilih)
    st.dataframe(stat_df, use_container_width=True)

    st.markdown(
        """
        **Interpretasi singkat:**
        - Mean menunjukkan nilai rata-rata.
        - Median menunjukkan nilai tengah.
        - Standar deviasi menunjukkan sebaran data.
        - IQR menunjukkan rentang antar kuartil.
        - Skewness menunjukkan kemencengan distribusi.
        - Kurtosis menunjukkan keruncingan distribusi.
        - Coefficient of Variation menunjukkan keragaman relatif dalam persen.
        """
    )


# ============================================================
# TAB 3 DISTRIBUSI
# ============================================================
with tab3:
    st.subheader("📈 Distribusi Data")

    variabel_dist = st.selectbox(
        "Pilih variabel distribusi",
        variabel_dipilih
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Histogram")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        ax_hist.hist(df_filter[variabel_dist].dropna(), bins=10)
        ax_hist.set_title(f"Histogram {variabel_dist}")
        ax_hist.set_xlabel(variabel_dist)
        ax_hist.set_ylabel("Frekuensi")
        st.pyplot(fig_hist)

        st.download_button(
            "⬇️ Download Histogram PNG",
            data=fig_to_png(fig_hist),
            file_name="histogram_smartcity.png",
            mime="image/png"
        )

    with col_b:
        st.markdown("### Boxplot")
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        ax_box.boxplot(df_filter[variabel_dist].dropna(), vert=True)
        ax_box.set_title(f"Boxplot {variabel_dist}")
        ax_box.set_ylabel(variabel_dist)
        st.pyplot(fig_box)

        st.download_button(
            "⬇️ Download Boxplot PNG",
            data=fig_to_png(fig_box),
            file_name="boxplot_smartcity.png",
            mime="image/png"
        )

    if stats is not None and len(df_filter[variabel_dist].dropna()) >= 3:
        st.markdown("### Uji Normalitas Shapiro-Wilk")
        nilai_stat, p_value = stats.shapiro(df_filter[variabel_dist].dropna())

        st.write(f"Statistik uji: **{nilai_stat:.4f}**")
        st.write(f"p-value: **{p_value:.4f}**")

        if p_value >= 0.05:
            st.success("Data cenderung berdistribusi normal pada taraf 5%.")
        else:
            st.warning("Data cenderung tidak berdistribusi normal pada taraf 5%.")


# ============================================================
# TAB 4 KORELASI
# ============================================================
with tab4:
    st.subheader("🔗 Analisis Korelasi")

    metode = st.selectbox(
        "Pilih metode korelasi",
        ["pearson", "spearman", "kendall"]
    )

    corr = df_filter[variabel_dipilih].corr(method=metode)
    st.dataframe(corr, use_container_width=True)

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    im = ax_corr.imshow(corr, aspect="auto")
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_corr.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    ax_corr.set_title(f"Heatmap Korelasi {metode.capitalize()}")
    fig_corr.colorbar(im, ax=ax_corr)
    st.pyplot(fig_corr)

    st.download_button(
        "⬇️ Download Heatmap Korelasi PNG",
        data=fig_to_png(fig_corr),
        file_name="heatmap_korelasi_smartcity.png",
        mime="image/png"
    )


# ============================================================
# TAB 5 OUTLIER
# ============================================================
with tab5:
    st.subheader("⚠️ Deteksi Outlier")

    st.markdown("### Metode IQR")
    outlier_iqr = deteksi_outlier_iqr(df_filter, variabel_dipilih)
    st.dataframe(outlier_iqr, use_container_width=True)

    st.markdown("### Metode Z-Score")
    threshold = st.slider(
        "Threshold Z-Score",
        min_value=1.5,
        max_value=4.0,
        value=3.0,
        step=0.1
    )

    outlier_z, detail_z = deteksi_outlier_zscore(
        df_filter,
        variabel_dipilih,
        threshold=threshold
    )

    st.dataframe(outlier_z, use_container_width=True)

    if not detail_z.empty:
        st.markdown("### Detail Data Outlier")
        st.dataframe(detail_z, use_container_width=True)
    else:
        st.success("Tidak ditemukan outlier berdasarkan threshold Z-Score yang dipilih.")


# ============================================================
# TAB 6 TREN
# ============================================================
with tab6:
    st.subheader("📉 Analisis Tren")

    variabel_tren = st.selectbox(
        "Pilih variabel tren",
        variabel_dipilih,
        key="variabel_tren"
    )

    data_tren, cagr = analisis_tren(df_filter, variabel_tren)

    st.dataframe(data_tren, use_container_width=True)

    if not np.isnan(cagr):
        st.metric("CAGR", f"{cagr:.2f}% per tahun")

    fig_tren, ax_tren = plt.subplots(figsize=(10, 5))
    ax_tren.plot(data_tren["Tahun"], data_tren[variabel_tren], marker="o", label=variabel_tren)
    ax_tren.plot(data_tren["Tahun"], data_tren["Moving Average"], marker="s", label="Moving Average")
    ax_tren.set_title(f"Tren {variabel_tren}")
    ax_tren.set_xlabel("Tahun")
    ax_tren.set_ylabel(variabel_tren)
    ax_tren.grid(True)
    ax_tren.legend()
    st.pyplot(fig_tren)

    st.download_button(
        "⬇️ Download Grafik Tren PNG",
        data=fig_to_png(fig_tren),
        file_name="grafik_tren_smartcity.png",
        mime="image/png"
    )


# ============================================================
# TAB 7 RANKING DAN INSIGHT
# ============================================================
with tab7:
    st.subheader("🏙️ Ranking Wilayah")

    variabel_rank = st.selectbox(
        "Pilih variabel ranking",
        variabel_dipilih,
        key="variabel_rank"
    )

    ranking = (
        df_filter
        .groupby("Kota", as_index=False)[variabel_rank]
        .mean()
        .sort_values(variabel_rank, ascending=False)
    )

    st.markdown("### Top Wilayah")
    st.dataframe(ranking.head(10), use_container_width=True)

    st.markdown("### Bottom Wilayah")
    st.dataframe(ranking.tail(10).sort_values(variabel_rank), use_container_width=True)

    fig_rank, ax_rank = plt.subplots(figsize=(10, 5))
    ax_rank.bar(ranking["Kota"], ranking[variabel_rank])
    ax_rank.set_title(f"Ranking Wilayah berdasarkan {variabel_rank}")
    ax_rank.set_xlabel("Kota / Wilayah")
    ax_rank.set_ylabel(variabel_rank)
    ax_rank.tick_params(axis="x", rotation=30)
    st.pyplot(fig_rank)

    st.download_button(
        "⬇️ Download Grafik Ranking PNG",
        data=fig_to_png(fig_rank),
        file_name="grafik_ranking_smartcity.png",
        mime="image/png"
    )

    st.subheader("🤖 Smart Insight Otomatis")
    insight = buat_smart_insight(df_filter, variabel_dipilih)
    st.markdown(insight)


# ============================================================
# FORMAT CSV
# ============================================================
st.divider()
st.subheader("🧾 Format CSV yang Disarankan")

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

st.markdown(
    """
    **Catatan:**
    - Nama kolom sebaiknya sama dengan contoh.
    - Kolom `Kota` digunakan untuk filter dan ranking wilayah.
    - Link online dapat berupa file `.csv` atau API JSON sederhana.
    - Untuk uji normalitas, instal SciPy:

    ```bash
    pip install scipy
    ```
    """
)
