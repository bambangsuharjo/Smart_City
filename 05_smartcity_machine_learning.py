# ============================================================
# Machine Learning untuk Smart City
# File: smartcity_machine_learning.py
#
# Fitur:
# 1. Data contoh
# 2. Upload CSV
# 3. Link API / CSV online
# 4. Pilih wilayah dan indikator target
# 5. Model Machine Learning:
#    - Linear Regression
#    - Random Forest Regressor
#    - Gradient Boosting Regressor
# 6. Evaluasi model:
#    - MAE
#    - RMSE
#    - R2
#    - MAPE
# 7. Grafik aktual vs prediksi
# 8. Forecast 1–5 tahun ke depan
# 9. Feature importance
# 10. Insight otomatis
#
# Jalankan:
# streamlit run smartcity_machine_learning.py
# ============================================================

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Machine Learning Smart City",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# DATA CONTOH
# ============================================================
def buat_data_contoh():
    kota = ["Jakarta", "Bandung", "Surabaya", "Medan", "Makassar"]
    tahun = list(range(2016, 2026))

    rows = []
    np.random.seed(42)

    base = {
        "Jakarta":     [10200000, 5800, 520, 7000, 70],
        "Bandung":     [2400000, 1500, 180, 1400, 55],
        "Surabaya":    [2850000, 2000, 230, 1900, 58],
        "Medan":       [2300000, 1350, 175, 1250, 50],
        "Makassar":    [1350000, 800, 100, 760, 42],
    }

    for k in kota:
        p0, e0, a0, s0, m0 = base[k]

        for i, t in enumerate(tahun):
            penduduk = p0 * (1 + 0.014) ** i + np.random.normal(0, p0 * 0.005)
            energi = e0 * (1 + 0.035) ** i + np.random.normal(0, e0 * 0.03)
            air = a0 * (1 + 0.025) ** i + np.random.normal(0, a0 * 0.03)
            sampah = s0 * (1 + 0.032) ** i + np.random.normal(0, s0 * 0.03)
            kemacetan = m0 + i * 1.2 + np.random.normal(0, 1.5)

            rows.append({
                "Tahun": t,
                "Kota": k,
                "Jumlah Penduduk": round(penduduk),
                "Konsumsi Energi": round(energi, 2),
                "Konsumsi Air": round(air, 2),
                "Volume Sampah": round(sampah, 2),
                "Indeks Kemacetan": round(kemacetan, 2),
            })

    return pd.DataFrame(rows)


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
# METRIK EVALUASI
# ============================================================
def hitung_metrik(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    mask = y_true_arr != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    else:
        mape = np.nan

    return mae, rmse, r2, mape


def buat_model(nama_model, random_state=42):
    if nama_model == "Linear Regression":
        return LinearRegression()

    if nama_model == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            max_depth=None
        )

    if nama_model == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )

    return LinearRegression()


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


# ============================================================
# INSIGHT OTOMATIS
# ============================================================
def buat_insight(target, mae, rmse, r2, mape, forecast_df):
    insight = []

    if r2 >= 0.80:
        kualitas = "sangat baik"
    elif r2 >= 0.60:
        kualitas = "cukup baik"
    elif r2 >= 0.30:
        kualitas = "sedang"
    else:
        kualitas = "masih lemah"

    insight.append(
        f"- Model memiliki kualitas prediksi **{kualitas}** dengan nilai R² sebesar **{r2:.3f}**."
    )

    if not np.isnan(mape):
        if mape <= 10:
            akurasi = "tinggi"
        elif mape <= 20:
            akurasi = "cukup"
        else:
            akurasi = "perlu ditingkatkan"

        insight.append(
            f"- Nilai MAPE sebesar **{mape:.2f}%**, sehingga akurasi relatif model tergolong **{akurasi}**."
        )

    if not forecast_df.empty:
        awal = forecast_df[target].iloc[0]
        akhir = forecast_df[target].iloc[-1]

        if akhir > awal:
            arah = "meningkat"
        elif akhir < awal:
            arah = "menurun"
        else:
            arah = "stabil"

        insight.append(
            f"- Hasil forecast menunjukkan bahwa **{target}** diperkirakan **{arah}** "
            f"dari **{awal:,.2f}** menjadi **{akhir:,.2f}**."
        )

    insight.append(
        "- Rekomendasi: hasil prediksi dapat digunakan sebagai dasar awal perencanaan, "
        "tetapi tetap perlu divalidasi dengan data lapangan, kebijakan daerah, dan kondisi sosial-ekonomi."
    )

    return "\n".join(insight)


# ============================================================
# JUDUL
# ============================================================
st.title("🤖 Machine Learning untuk Smart City")
st.markdown(
    "Aplikasi ini menunjukkan bagaimana Machine Learning digunakan untuk memprediksi "
    "indikator Smart City seperti jumlah penduduk, konsumsi energi, konsumsi air, "
    "volume sampah, dan indeks kemacetan."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Pengaturan Machine Learning")

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

kota_pilih = st.sidebar.multiselect(
    "Pilih Kota / Wilayah",
    kota_list,
    default=kota_list
)

tahun_min = int(df["Tahun"].min())
tahun_max = int(df["Tahun"].max())

rentang_tahun = st.sidebar.slider(
    "Rentang Tahun Data Latih",
    min_value=tahun_min,
    max_value=tahun_max,
    value=(tahun_min, tahun_max)
)

df_filter = df[
    (df["Kota"].astype(str).isin(kota_pilih)) &
    (df["Tahun"] >= rentang_tahun[0]) &
    (df["Tahun"] <= rentang_tahun[1])
].copy()

if df_filter.empty:
    st.warning("Data tidak tersedia untuk filter yang dipilih.")
    st.stop()


st.sidebar.divider()
st.sidebar.subheader("3. Model Prediksi")

target = st.sidebar.selectbox(
    "Pilih Target Prediksi",
    indikator_list,
    index=4
)

fitur_default = [x for x in indikator_list if x != target]

fitur = st.sidebar.multiselect(
    "Pilih Fitur/Input Model",
    fitur_default + ["Tahun"],
    default=fitur_default + ["Tahun"]
)

if not fitur:
    st.warning("Pilih minimal satu fitur.")
    st.stop()

model_nama = st.sidebar.selectbox(
    "Pilih Model Machine Learning",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

test_size = st.sidebar.slider(
    "Proporsi Data Uji",
    min_value=0.10,
    max_value=0.40,
    value=0.25,
    step=0.05
)

forecast_years = st.sidebar.slider(
    "Prediksi Berapa Tahun ke Depan",
    min_value=1,
    max_value=5,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=1,
    max_value=9999,
    value=42
)


# ============================================================
# DATASET MODEL
# ============================================================
st.subheader("📋 Data yang Digunakan")
st.dataframe(df_filter, use_container_width=True)

X = df_filter[fitur].copy()
y = df_filter[target].copy()

if len(df_filter) < 8:
    st.warning(
        "Jumlah data terlalu sedikit. Gunakan minimal 8 baris data agar pelatihan dan evaluasi lebih stabil."
    )
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(random_state)
)

model = buat_model(model_nama, random_state=int(random_state))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae, rmse, r2, mape = hitung_metrik(y_test, y_pred)


# ============================================================
# HASIL EVALUASI
# ============================================================
st.subheader("📊 Evaluasi Model")

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("R²", f"{r2:.3f}")
col4.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "-")

hasil_prediksi = pd.DataFrame({
    "Aktual": y_test.values,
    "Prediksi": y_pred
}).reset_index(drop=True)

st.dataframe(hasil_prediksi, use_container_width=True)


# ============================================================
# GRAFIK AKTUAL VS PREDIKSI
# ============================================================
st.subheader("📈 Grafik Aktual vs Prediksi")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(hasil_prediksi.index, hasil_prediksi["Aktual"], marker="o", label="Aktual")
ax1.plot(hasil_prediksi.index, hasil_prediksi["Prediksi"], marker="s", label="Prediksi")
ax1.set_title(f"Aktual vs Prediksi - {target}")
ax1.set_xlabel("Data Uji")
ax1.set_ylabel(target)
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

st.download_button(
    "⬇️ Download Grafik Aktual vs Prediksi",
    data=fig_to_png(fig1),
    file_name="aktual_vs_prediksi_ml_smartcity.png",
    mime="image/png"
)


# ============================================================
# FEATURE IMPORTANCE / KOEFISIEN
# ============================================================
st.subheader("🧠 Pengaruh Fitur terhadap Model")

if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Fitur": fitur,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance, use_container_width=True)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.bar(importance["Fitur"], importance["Importance"])
    ax_imp.set_title("Feature Importance")
    ax_imp.set_xlabel("Fitur")
    ax_imp.set_ylabel("Importance")
    ax_imp.tick_params(axis="x", rotation=30)
    st.pyplot(fig_imp)

elif hasattr(model, "coef_"):
    coef = pd.DataFrame({
        "Fitur": fitur,
        "Koefisien": model.coef_
    }).sort_values("Koefisien", ascending=False)

    st.dataframe(coef, use_container_width=True)

    fig_coef, ax_coef = plt.subplots(figsize=(10, 5))
    ax_coef.bar(coef["Fitur"], coef["Koefisien"])
    ax_coef.set_title("Koefisien Linear Regression")
    ax_coef.set_xlabel("Fitur")
    ax_coef.set_ylabel("Koefisien")
    ax_coef.tick_params(axis="x", rotation=30)
    st.pyplot(fig_coef)


# ============================================================
# FORECAST MASA DEPAN
# ============================================================
st.subheader("🔮 Forecast Indikator Smart City")

tahun_terakhir = int(df_filter["Tahun"].max())
tahun_depan = list(range(tahun_terakhir + 1, tahun_terakhir + forecast_years + 1))

# Untuk forecast sederhana:
# nilai fitur selain Tahun memakai rata-rata tahun terakhir per seluruh wilayah terpilih.
data_terakhir = df_filter[df_filter["Tahun"] == tahun_terakhir].copy()
rata_fitur = data_terakhir[fitur].mean(numeric_only=True)

future_rows = []

for t in tahun_depan:
    row = {}
    for f in fitur:
        if f == "Tahun":
            row[f] = t
        else:
            row[f] = rata_fitur[f]
    future_rows.append(row)

future_X = pd.DataFrame(future_rows)
future_pred = model.predict(future_X)

forecast_df = future_X.copy()
forecast_df[target] = future_pred

st.dataframe(forecast_df, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 5))

data_hist = (
    df_filter
    .groupby("Tahun", as_index=False)[target]
    .mean()
    .sort_values("Tahun")
)

ax2.plot(data_hist["Tahun"], data_hist[target], marker="o", label="Historis")
ax2.plot(forecast_df["Tahun"], forecast_df[target], marker="s", linestyle="--", label="Forecast")
ax2.set_title(f"Forecast {target} {forecast_years} Tahun ke Depan")
ax2.set_xlabel("Tahun")
ax2.set_ylabel(target)
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

st.download_button(
    "⬇️ Download Grafik Forecast",
    data=fig_to_png(fig2),
    file_name="forecast_ml_smartcity.png",
    mime="image/png"
)

st.download_button(
    "⬇️ Download Hasil Forecast CSV",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="forecast_ml_smartcity.csv",
    mime="text/csv"
)


# ============================================================
# SMART INSIGHT
# ============================================================
st.subheader("🤖 Insight Otomatis")

insight = buat_insight(target, mae, rmse, r2, mape, forecast_df)
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
    - Data minimal sebaiknya memiliki beberapa tahun observasi.
    - Kolom `Kota` dapat dipakai untuk membandingkan beberapa wilayah.
    - Untuk prediksi time series yang lebih kuat, aplikasi berikutnya dapat dikembangkan dengan LSTM atau GRU.
    - Library yang diperlukan:

    ```bash
    pip install streamlit pandas numpy matplotlib scikit-learn requests
    ```
    """
)
