# ============================================================
# Smart City GIS Indonesia - Local Shapefile Version
# File: smartcity_gis_indonesia_local.py
#
# Solusi untuk masalah upload shapefile besar:
# - Shapefile ZIP tidak di-upload melalui Streamlit.
# - File indonesia.zip diletakkan satu folder dengan file Python ini.
# - Aplikasi membaca shapefile dari file lokal.
#
# Struktur folder:
# project/
# ├── smartcity_gis_indonesia_local.py
# ├── indonesia.zip
#
# Jalankan:
# streamlit run smartcity_gis_indonesia_local.py
# ============================================================

import os
import re
import zipfile
import tempfile
import requests
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None


st.set_page_config(
    page_title="Smart City GIS Indonesia Local",
    page_icon="🗺️",
    layout="wide"
)


# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = Path(__file__).parent
DEFAULT_SHAPEFILE_ZIP = BASE_DIR / "indonesia.zip"


# ============================================================
# FUNGSI BANTUAN
# ============================================================
def normalisasi_nama_wilayah(x):
    if pd.isna(x):
        return ""

    x = str(x).upper().strip()
    x = x.replace("KAB.", "KABUPATEN")
    x = x.replace("KAB ", "KABUPATEN ")
    x = x.replace("KOTA ADM.", "KOTA")
    x = x.replace("KOTA ADMINISTRASI", "KOTA")
    x = x.replace("ADM. ", "")
    x = x.replace("ADMINISTRASI ", "")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


@st.cache_data(show_spinner=False)
def buat_data_contoh():
    return pd.DataFrame({
        "Tahun": [2025] * 12,
        "Provinsi": [
            "DKI JAKARTA", "DKI JAKARTA", "DKI JAKARTA",
            "JAWA BARAT", "JAWA BARAT", "JAWA BARAT",
            "JAWA TIMUR", "JAWA TIMUR", "JAWA TIMUR",
            "SUMATERA UTARA", "SULAWESI SELATAN", "BALI"
        ],
        "Kabupaten_Kota": [
            "KOTA JAKARTA SELATAN",
            "KOTA JAKARTA TIMUR",
            "KOTA JAKARTA PUSAT",
            "KOTA BANDUNG",
            "KABUPATEN BANDUNG",
            "KOTA BEKASI",
            "KOTA SURABAYA",
            "KABUPATEN SIDOARJO",
            "KOTA MALANG",
            "KOTA MEDAN",
            "KOTA MAKASSAR",
            "KOTA DENPASAR"
        ],
        "Jumlah Penduduk": [
            2350000, 3050000, 1100000,
            2670000, 3700000, 2550000,
            3100000, 2100000, 890000,
            2610000, 1620000, 750000
        ],
        "Konsumsi Energi": [
            3200, 3600, 2100,
            2120, 2500, 2400,
            2770, 1850, 980,
            1960, 1200, 780
        ],
        "Konsumsi Air": [
            300, 350, 180,
            241, 300, 260,
            300, 220, 95,
            236, 154, 90
        ],
        "Volume Sampah": [
            2600, 3100, 1600,
            2070, 2500, 2200,
            2620, 1700, 760,
            1900, 1160, 700
        ],
        "Indeks Kemacetan": [
            85, 88, 80,
            72, 66, 78,
            75, 62, 58,
            65, 55, 60
        ],
    })


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


def validasi_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    kolom_wajib = [
        "Tahun", "Provinsi", "Kabupaten_Kota",
        "Jumlah Penduduk", "Konsumsi Energi", "Konsumsi Air",
        "Volume Sampah", "Indeks Kemacetan"
    ]

    kurang = [k for k in kolom_wajib if k not in df.columns]
    if kurang:
        st.error("Kolom berikut belum ada pada CSV: " + ", ".join(kurang))
        st.stop()

    indikator = [
        "Jumlah Penduduk", "Konsumsi Energi", "Konsumsi Air",
        "Volume Sampah", "Indeks Kemacetan"
    ]

    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df = df.dropna(subset=["Tahun"])
    df["Tahun"] = df["Tahun"].astype(int)

    for col in indikator:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["JOIN_PROV"] = df["Provinsi"].apply(normalisasi_nama_wilayah)
    df["JOIN_KABKOT"] = df["Kabupaten_Kota"].apply(normalisasi_nama_wilayah)

    return df


@st.cache_data(show_spinner=True)
def baca_shapefile_lokal(zip_path_str, tolerance=0.01):
    if gpd is None:
        st.error(
            "Geopandas belum terpasang. Jalankan:\n\n"
            "pip install geopandas pyogrio shapely fiona"
        )
        st.stop()

    zip_path = Path(zip_path_str)

    if not zip_path.exists():
        st.error(
            f"File shapefile tidak ditemukan: {zip_path}\n\n"
            "Letakkan file indonesia.zip satu folder dengan smartcity_gis_indonesia_local.py"
        )
        st.stop()

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    shp_files = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, file))

    if not shp_files:
        st.error("ZIP tidak berisi file .shp.")
        st.stop()

    gdf = gpd.read_file(shp_files[0])
    gdf.columns = [str(c).strip() for c in gdf.columns]

    if "WADMPR" not in gdf.columns or "WADMKK" not in gdf.columns:
        st.error("Shapefile harus memiliki kolom WADMPR dan WADMKK.")
        st.write("Kolom tersedia:", list(gdf.columns))
        st.stop()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf = gdf.to_crs(epsg=4326)

    # Ambil kolom penting saja agar ringan
    gdf = gdf[["WADMPR", "WADMKK", "geometry"]].copy()

    gdf["WADMPR"] = gdf["WADMPR"].astype(str)
    gdf["WADMKK"] = gdf["WADMKK"].astype(str)
    gdf["JOIN_PROV"] = gdf["WADMPR"].apply(normalisasi_nama_wilayah)
    gdf["JOIN_KABKOT"] = gdf["WADMKK"].apply(normalisasi_nama_wilayah)

    if tolerance > 0:
        gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)

    return gdf


def buat_peta(gdf_join, indikator):
    if folium is None or st_folium is None:
        st.error("Folium belum terpasang. Jalankan: pip install folium streamlit-folium")
        st.stop()

    gdf_map = gdf_join.copy()
    gdf_map = gdf_map[gdf_map[indikator].notna()].copy()

    if gdf_map.empty:
        return None

    gdf_map["MAP_ID"] = gdf_map.index.astype(str)
    gdf_map["MAP_PROV"] = gdf_map["WADMPR"].astype(str)
    gdf_map["MAP_KABKOT"] = gdf_map["WADMKK"].astype(str)

    center = gdf_map.geometry.unary_union.centroid

    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=6,
        tiles="cartodbpositron"
    )

    folium.Choropleth(
        geo_data=gdf_map.to_json(),
        data=gdf_map,
        columns=["MAP_ID", indikator],
        key_on="feature.properties.MAP_ID",
        fill_opacity=0.75,
        line_opacity=0.25,
        nan_fill_color="lightgray",
        legend_name=indikator,
    ).add_to(m)

    folium.GeoJson(
        gdf_map,
        tooltip=folium.GeoJsonTooltip(
            fields=["MAP_PROV", "MAP_KABKOT", indikator],
            aliases=["Provinsi", "Kabupaten/Kota", indikator],
            localize=True
        ),
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 0.4,
            "fillOpacity": 0,
        }
    ).add_to(m)

    return m


# ============================================================
# TAMPILAN UTAMA
# ============================================================
st.title("🗺️ Smart City GIS Indonesia")
st.markdown(
    "Versi ini **tidak memakai upload shapefile**. File `indonesia.zip` cukup "
    "diletakkan satu folder dengan file Python ini."
)

st.sidebar.header("⚙️ Pengaturan GIS")

st.sidebar.subheader("1. Data Smart City")
sumber_data = st.sidebar.selectbox(
    "Pilih sumber data",
    ["Data Contoh", "Upload CSV", "Link API / CSV Online"]
)

if sumber_data == "Data Contoh":
    df = buat_data_contoh()

elif sumber_data == "Upload CSV":
    file_csv = st.sidebar.file_uploader("Upload CSV indikator", type=["csv"])
    if file_csv is None:
        st.warning("Upload CSV terlebih dahulu.")
        st.stop()
    df = pd.read_csv(file_csv)

else:
    url = st.sidebar.text_input("Masukkan URL API atau CSV")
    if not url:
        st.info("Masukkan URL terlebih dahulu.")
        st.stop()
    df = ambil_data_online(url)

df = validasi_data(df)

st.sidebar.subheader("2. File Shapefile Lokal")
nama_file_zip = st.sidebar.text_input(
    "Nama file ZIP shapefile",
    value="indonesia.zip"
)

zip_lokal = BASE_DIR / nama_file_zip

st.sidebar.caption(
    "Letakkan file ZIP di folder yang sama dengan file Python. "
    "Jangan upload shapefile melalui aplikasi."
)

tolerance = st.sidebar.slider(
    "Tingkat penyederhanaan peta",
    min_value=0.001,
    max_value=0.050,
    value=0.010,
    step=0.001,
)

with st.spinner("Membaca shapefile lokal..."):
    gdf = baca_shapefile_lokal(str(zip_lokal), tolerance=tolerance)

indikator_list = [
    "Jumlah Penduduk", "Konsumsi Energi", "Konsumsi Air",
    "Volume Sampah", "Indeks Kemacetan"
]

st.sidebar.subheader("3. Filter")

tahun_list = sorted(df["Tahun"].unique().tolist())
tahun = st.sidebar.selectbox("Tahun", tahun_list, index=len(tahun_list)-1)

# Provinsi berasal dari shapefile, jadi bisa memilih provinsi meskipun data contoh belum semua ada
provinsi_peta = sorted(gdf["WADMPR"].dropna().astype(str).unique().tolist())

provinsi = st.sidebar.multiselect(
    "Provinsi pada Peta",
    provinsi_peta,
    default=["JAWA BARAT"] if "JAWA BARAT" in provinsi_peta else provinsi_peta[:1]
)

indikator = st.sidebar.selectbox("Indikator peta", indikator_list)

if not provinsi:
    st.warning("Pilih minimal satu provinsi.")
    st.stop()

provinsi_norm = [normalisasi_nama_wilayah(p) for p in provinsi]

df_filter = df[
    (df["Tahun"] == tahun) &
    (df["JOIN_PROV"].isin(provinsi_norm))
].copy()

gdf_filter = gdf[gdf["JOIN_PROV"].isin(provinsi_norm)].copy()

data_agregat = (
    df_filter
    .groupby(["JOIN_PROV", "JOIN_KABKOT"], as_index=False)[indikator_list]
    .mean()
)

gdf_join = gdf_filter.merge(
    data_agregat,
    on=["JOIN_PROV", "JOIN_KABKOT"],
    how="left"
)

terpetakan = int(gdf_join[indikator].notna().sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Provinsi dipilih", len(provinsi))
col2.metric("Wilayah pada peta", len(gdf_filter))
col3.metric("Wilayah terpetakan", terpetakan)
col4.metric("Tahun", tahun)

st.subheader(f"🗺️ Peta Tematik {indikator}")

if terpetakan == 0:
    st.warning(
        "Peta berhasil dibaca, tetapi belum ada data CSV yang cocok dengan wilayah peta. "
        "Coba gunakan Data Contoh dan pilih provinsi Jawa Barat, DKI Jakarta, Jawa Timur, "
        "Sumatera Utara, Sulawesi Selatan, atau Bali."
    )
else:
    peta = buat_peta(gdf_join, indikator)
    if peta is not None:
        st_folium(peta, width=None, height=650)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data CSV",
    "Data Shapefile",
    "Hasil Join",
    "Wilayah Belum Cocok"
])

with tab1:
    st.dataframe(df_filter, use_container_width=True)

with tab2:
    st.dataframe(
        gdf_filter[["WADMPR", "WADMKK"]].sort_values(["WADMPR", "WADMKK"]),
        use_container_width=True
    )

with tab3:
    kolom = ["WADMPR", "WADMKK"] + indikator_list
    st.dataframe(gdf_join[kolom], use_container_width=True)

with tab4:
    shp_key = set(zip(gdf_filter["JOIN_PROV"], gdf_filter["JOIN_KABKOT"]))
    csv_key = set(zip(df_filter["JOIN_PROV"], df_filter["JOIN_KABKOT"]))
    belum = sorted(csv_key - shp_key)

    if not belum:
        st.success("Semua wilayah CSV cocok dengan shapefile.")
    else:
        st.warning("Ada wilayah CSV yang belum cocok dengan shapefile.")
        st.dataframe(
            pd.DataFrame(belum, columns=["Provinsi_Normalisasi", "Kabupaten_Kota_Normalisasi"]),
            use_container_width=True
        )

st.divider()
st.subheader("🧾 Cara Menjalankan")

st.code(
    """
project/
├── smartcity_gis_indonesia_local.py
├── indonesia.zip

streamlit run smartcity_gis_indonesia_local.py
    """,
    language="bash"
)

st.subheader("Format CSV yang Disarankan")
st.dataframe(pd.DataFrame({
    "Tahun": [2025, 2025],
    "Provinsi": ["JAWA BARAT", "JAWA TIMUR"],
    "Kabupaten_Kota": ["KOTA BANDUNG", "KOTA SURABAYA"],
    "Jumlah Penduduk": [2670000, 3100000],
    "Konsumsi Energi": [2120, 2770],
    "Konsumsi Air": [241, 300],
    "Volume Sampah": [2070, 2620],
    "Indeks Kemacetan": [72, 75],
}), use_container_width=True)
