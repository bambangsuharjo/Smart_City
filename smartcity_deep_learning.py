# ============================================================
# Deep Learning Prediksi Smart City
# File: smartcity_deep_learning.py
#
# Fitur:
# 1. Data contoh
# 2. Upload CSV
# 3. Link API / CSV online
# 4. Pilih kota/wilayah
# 5. Pilih indikator target
# 6. Model Deep Learning:
#    - LSTM
#    - GRU
#    - Simple RNN
# 7. Sequence time series otomatis
# 8. Evaluasi model:
#    - MAE
#    - RMSE
#    - R2
#    - MAPE
# 9. Grafik training loss
# 10. Grafik aktual vs prediksi
# 11. Forecast 1–5 tahun ke depan
# 12. Insight otomatis
#
# Jalankan:
# streamlit run smartcity_deep_learning.py
#
# Install:
# pip install streamlit pandas numpy matplotlib scikit-learn tensorflow requests
# ============================================================

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    tf = None


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Deep Learning Prediksi Smart City",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# DATA CONTOH
# ============================================================
def buat_data_contoh():
    kota = ["Jakarta", "Bandung", "Surabaya", "Medan", "Makassar"]
    tahun = list(range(2006, 2026))

    rows = []
    np.random.seed(42)

    base = {
        "Jakarta":     [9200000, 4200, 410, 5200, 60],
        "Bandung":     [2100000, 1050, 130, 950, 42],
        "Surabaya":    [2500000, 1400, 170, 1250, 45],
        "Medan":       [2000000, 980, 125, 850, 38],
        "Makassar":    [1100000, 600, 75, 520, 32],
    }

    for k in kota:
        p0, e0, a0, s0, m0 = base[k]

        for i, t in enumerate(tahun):
            penduduk = p0 * (1 + 0.015) ** i + np.random.normal(0, p0 * 0.004)
            energi = e0 * (1 + 0.038) ** i + np.random.normal(0, e0 * 0.025)
            air = a0 * (1 + 0.026) ** i + np.random.normal(0, a0 * 0.025)
            sampah = s0 * (1 + 0.034) ** i + np.random.normal(0, s0 * 0.025)
            kemacetan = m0 + i * 1.15 + np.random.normal(0, 1.2)

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
# TIME SERIES SEQUENCE
# ============================================================
def buat_sequence(data, window_size):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def buat_model_dl(model_type, window_size, units, dropout, learning_rate):
    if tf is None:
        st.error(
            "TensorFlow belum terpasang. Jalankan:\n\n"
            "pip install tensorflow"
        )
        st.stop()

    model = Sequential()

    if model_type == "LSTM":
        model.add(LSTM(units, input_shape=(window_size, 1)))
    elif model_type == "GRU":
        model.add(GRU(units, input_shape=(window_size, 1)))
    else:
        model.add(SimpleRNN(units, input_shape=(window_size, 1)))

    model.add(Dropout(dropout))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


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


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def forecast_deep_learning(model, scaler, data_scaled, window_size, langkah):
    history = list(data_scaled.flatten())
    pred_scaled = []

    for _ in range(langkah):
        x_input = np.array(history[-window_size:]).reshape(1, window_size, 1)
        yhat = model.predict(x_input, verbose=0)[0][0]
        pred_scaled.append(yhat)
        history.append(yhat)

    pred_scaled = np.array(pred_scaled).reshape(-1, 1)
    pred = scaler.inverse_transform(pred_scaled).flatten()

    return pred


# ============================================================
# INSIGHT OTOMATIS
# ============================================================
def buat_insight(target, model_type, mae, rmse, r2, mape, forecast_df):
    insight = []

    insight.append(
        f"- Model **{model_type}** digunakan untuk mempelajari pola time series pada indikator **{target}**."
    )

    if r2 >= 0.80:
        kualitas = "sangat baik"
    elif r2 >= 0.60:
        kualitas = "cukup baik"
    elif r2 >= 0.30:
        kualitas = "sedang"
    else:
        kualitas = "masih perlu ditingkatkan"

    insight.append(
        f"- Nilai R² sebesar **{r2:.3f}**, sehingga kemampuan model dalam mengikuti pola data tergolong **{kualitas}**."
    )

    if not np.isnan(mape):
        insight.append(
            f"- Nilai MAPE sebesar **{mape:.2f}%** menunjukkan tingkat kesalahan relatif prediksi."
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
        "- Catatan: Deep Learning membutuhkan data time series yang cukup panjang. "
        "Semakin banyak data historis, semakin stabil model yang dihasilkan."
    )

    return "\n".join(insight)


# ============================================================
# JUDUL
# ============================================================
st.title("🧠 Deep Learning Prediksi Smart City")
st.markdown(
    "Aplikasi ini menunjukkan penggunaan model **LSTM, GRU, dan Simple RNN** "
    "untuk memprediksi indikator Smart City berdasarkan data time series."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Pengaturan Deep Learning")

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
st.sidebar.subheader("2. Pilihan Data")

kota_list = sorted(df["Kota"].dropna().astype(str).unique().tolist())

kota_pilih = st.sidebar.selectbox(
    "Pilih Kota / Wilayah",
    kota_list
)

target = st.sidebar.selectbox(
    "Pilih Indikator Target",
    indikator_list,
    index=4
)

df_kota = (
    df[df["Kota"].astype(str) == kota_pilih]
    .groupby("Tahun", as_index=False)[target]
    .mean()
    .sort_values("Tahun")
)

if len(df_kota) < 8:
    st.warning(
        "Data time series terlalu pendek. Gunakan minimal 8 titik waktu, "
        "lebih baik 15–30 titik waktu."
    )
    st.stop()


st.sidebar.divider()
st.sidebar.subheader("3. Parameter Model")

model_type = st.sidebar.selectbox(
    "Pilih Model Deep Learning",
    ["LSTM", "GRU", "Simple RNN"]
)

window_size = st.sidebar.slider(
    "Window Size",
    min_value=2,
    max_value=min(8, len(df_kota) - 3),
    value=min(4, len(df_kota) - 3)
)

units = st.sidebar.slider(
    "Jumlah Neuron",
    min_value=8,
    max_value=128,
    value=32,
    step=8
)

dropout = st.sidebar.slider(
    "Dropout",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05
)

epochs = st.sidebar.slider(
    "Epochs",
    min_value=20,
    max_value=300,
    value=100,
    step=10
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [4, 8, 16, 32],
    index=1
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.01, 0.005, 0.001, 0.0005],
    index=2
)

forecast_years = st.sidebar.slider(
    "Prediksi Tahun ke Depan",
    min_value=1,
    max_value=5,
    value=3
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    max_value=9999,
    value=42
)


# ============================================================
# TRAIN MODEL
# ============================================================
if tf is None:
    st.error(
        "TensorFlow belum terpasang. Jalankan:\n\n"
        "pip install tensorflow"
    )
    st.stop()

tf.random.set_seed(int(random_seed))
np.random.seed(int(random_seed))

st.subheader("📋 Data Time Series")
st.dataframe(df_kota, use_container_width=True)

data_values = df_kota[target].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_values)

X, y = buat_sequence(data_scaled, window_size)

if len(X) < 5:
    st.warning("Sequence terlalu sedikit. Kurangi window size atau tambah data historis.")
    st.stop()

split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = buat_model_dl(
    model_type=model_type,
    window_size=window_size,
    units=units,
    dropout=dropout,
    learning_rate=learning_rate
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

with st.spinner("Melatih model Deep Learning..."):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

y_pred_scaled = model.predict(X_test, verbose=0)

y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred_scaled).flatten()

mae, rmse, r2, mape = hitung_metrik(y_test_inv, y_pred_inv)


# ============================================================
# EVALUASI
# ============================================================
st.subheader("📊 Evaluasi Model Deep Learning")

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("R²", f"{r2:.3f}")
col4.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "-")

hasil_pred = pd.DataFrame({
    "Aktual": y_test_inv,
    "Prediksi": y_pred_inv
})

st.dataframe(hasil_pred, use_container_width=True)


# ============================================================
# TRAINING LOSS
# ============================================================
st.subheader("📉 Grafik Training Loss")

fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
ax_loss.plot(history.history["loss"], label="Training Loss")
ax_loss.plot(history.history["val_loss"], label="Validation Loss")
ax_loss.set_title("Training Loss dan Validation Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)
ax_loss.legend()
st.pyplot(fig_loss)

st.download_button(
    "⬇️ Download Grafik Loss",
    data=fig_to_png(fig_loss),
    file_name="loss_deep_learning_smartcity.png",
    mime="image/png"
)


# ============================================================
# AKTUAL VS PREDIKSI
# ============================================================
st.subheader("📈 Grafik Aktual vs Prediksi")

fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
ax_pred.plot(hasil_pred.index, hasil_pred["Aktual"], marker="o", label="Aktual")
ax_pred.plot(hasil_pred.index, hasil_pred["Prediksi"], marker="s", label="Prediksi")
ax_pred.set_title(f"Aktual vs Prediksi {target}")
ax_pred.set_xlabel("Data Uji")
ax_pred.set_ylabel(target)
ax_pred.grid(True)
ax_pred.legend()
st.pyplot(fig_pred)

st.download_button(
    "⬇️ Download Grafik Aktual vs Prediksi",
    data=fig_to_png(fig_pred),
    file_name="aktual_vs_prediksi_deep_learning.png",
    mime="image/png"
)


# ============================================================
# FORECAST
# ============================================================
st.subheader("🔮 Forecast Deep Learning")

pred_future = forecast_deep_learning(
    model=model,
    scaler=scaler,
    data_scaled=data_scaled,
    window_size=window_size,
    langkah=forecast_years
)

tahun_terakhir = int(df_kota["Tahun"].max())
tahun_future = list(range(tahun_terakhir + 1, tahun_terakhir + forecast_years + 1))

forecast_df = pd.DataFrame({
    "Tahun": tahun_future,
    target: pred_future
})

st.dataframe(forecast_df, use_container_width=True)

fig_forecast, ax_forecast = plt.subplots(figsize=(10, 5))
ax_forecast.plot(df_kota["Tahun"], df_kota[target], marker="o", label="Historis")
ax_forecast.plot(forecast_df["Tahun"], forecast_df[target], marker="s", linestyle="--", label="Forecast")
ax_forecast.set_title(f"Forecast {target} dengan {model_type}")
ax_forecast.set_xlabel("Tahun")
ax_forecast.set_ylabel(target)
ax_forecast.grid(True)
ax_forecast.legend()
st.pyplot(fig_forecast)

st.download_button(
    "⬇️ Download Grafik Forecast",
    data=fig_to_png(fig_forecast),
    file_name="forecast_deep_learning_smartcity.png",
    mime="image/png"
)

st.download_button(
    "⬇️ Download Forecast CSV",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="forecast_deep_learning_smartcity.csv",
    mime="text/csv"
)


# ============================================================
# SMART INSIGHT
# ============================================================
st.subheader("🤖 Insight Otomatis")

insight = buat_insight(
    target=target,
    model_type=model_type,
    mae=mae,
    rmse=rmse,
    r2=r2,
    mape=mape,
    forecast_df=forecast_df
)

st.markdown(insight)


# ============================================================
# PENJELASAN KONSEP
# ============================================================
st.divider()
st.subheader("📘 Penjelasan Singkat")

st.markdown(
    """
    **Deep Learning untuk prediksi Smart City** digunakan untuk mempelajari pola deret waktu.
    Model seperti LSTM dan GRU cocok untuk data yang memiliki ketergantungan antarwaktu,
    misalnya pertumbuhan penduduk, konsumsi energi, kebutuhan air, volume sampah,
    dan tingkat kemacetan.

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

st.markdown(
    """
    **Catatan:**
    - Data Deep Learning sebaiknya memiliki banyak titik waktu.
    - Untuk data tahunan, minimal 15–30 tahun lebih baik.
    - Untuk data bulanan atau harian, model dapat bekerja lebih kuat.
    - Jika data hanya sedikit, Machine Learning klasik sering lebih stabil daripada Deep Learning.
    """
)
