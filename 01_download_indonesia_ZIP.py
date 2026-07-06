# ============================================================
# Download Indonesia ZIP - Streamlit
# File: download_indonesia_streamlit.py
#
# Fungsi:
# - Mengunduh indonesia.zip dari Google Drive
# - Menyimpan otomatis ke folder data/indonesia.zip
# - Cocok sebagai langkah awal sebelum menjalankan aplikasi GIS
# ============================================================

import os
import re
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="Download Indonesia ZIP",
    page_icon="📦",
    layout="centered"
)

# Link Google Drive milik pengguna
DEFAULT_DRIVE_LINK = "https://drive.google.com/file/d/1KRhCI26tRyysKWqav83wxg___Px2hXIQ/view?usp=sharing"
DEFAULT_FILE_ID = "1KRhCI26tRyysKWqav83wxg___Px2hXIQ"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "indonesia.zip"


def extract_file_id(link_or_id: str) -> str:
    """Ambil File ID dari link Google Drive atau langsung ID."""
    text = str(link_or_id).strip()
    if not text:
        return ""

    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"id=([A-Za-z0-9_-]+)",
        r"/d/([A-Za-z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return text


def get_confirm_token(response):
    """Ambil token konfirmasi Google Drive untuk file besar."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def download_google_drive_file(file_id: str, destination: Path):
    """Download file besar dari Google Drive menggunakan requests."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    base_url = "https://docs.google.com/uc?export=download"

    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=60)
    token = get_confirm_token(response)

    if token:
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=60,
        )

    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024

    progress_bar = st.progress(0)
    status_text = st.empty()

    temp_file = destination.with_suffix(".download")

    with open(temp_file, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(downloaded / total, 1.0)
                    progress_bar.progress(pct)
                    status_text.write(
                        f"Mengunduh... {downloaded / 1024 / 1024:.1f} MB dari {total / 1024 / 1024:.1f} MB"
                    )
                else:
                    status_text.write(f"Mengunduh... {downloaded / 1024 / 1024:.1f} MB")

    os.replace(temp_file, destination)
    progress_bar.progress(1.0)
    status_text.success("Download selesai.")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


st.title("📦 Download indonesia.zip")
st.markdown(
    "Aplikasi kecil ini digunakan untuk mengunduh file **indonesia.zip** "
    "ke folder `data/`, sehingga dapat dipakai oleh aplikasi GIS Smart City."
)

st.info("File akan disimpan sebagai: `data/indonesia.zip`")

link_or_id = st.text_input(
    "Link Google Drive atau File ID",
    value=DEFAULT_DRIVE_LINK,
)

file_id = extract_file_id(link_or_id)

st.caption(f"File ID terdeteksi: `{file_id}`")

if OUTPUT_FILE.exists():
    st.success(f"File sudah tersedia: `{OUTPUT_FILE}`")
    st.write(f"Ukuran file: **{file_size_mb(OUTPUT_FILE):.2f} MB**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download ulang / Timpa file"):
            try:
                download_google_drive_file(file_id, OUTPUT_FILE)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal download: {e}")
    with col2:
        if st.button("Hapus file lokal"):
            OUTPUT_FILE.unlink(missing_ok=True)
            st.warning("File lokal sudah dihapus.")
            st.rerun()
else:
    st.warning("File `data/indonesia.zip` belum tersedia.")
    if st.button("Mulai Download indonesia.zip"):
        try:
            download_google_drive_file(file_id, OUTPUT_FILE)
            st.success("File berhasil diunduh dan siap dipakai aplikasi GIS.")
            st.rerun()
        except Exception as e:
            st.error(f"Gagal download: {e}")

st.divider()
st.subheader("Cara menjalankan")
st.code("streamlit run download_indonesia_streamlit.py", language="bash")

st.subheader("Setelah selesai")
st.markdown(
    "Jalankan aplikasi GIS Anda. Aplikasi GIS akan memakai file `data/indonesia.zip` "
    "yang sudah diunduh oleh halaman ini."
)
