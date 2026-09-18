"""
Download and extract the EuroSAT (RGB) dataset.

EuroSAT: Land Use and Land Cover Classification with Sentinel-2
Helber et al., 2019 (IEEE JSTARS) / IGARSS 2018
Official repo: https://github.com/phelber/eurosat
License: MIT (repository/labels) + Copernicus Sentinel Data Terms
         (Sentinel-2 imagery is free, full and open under the EU's
         Copernicus data policy).

This script expects the official archive (~90-95 MB) to already be present
at data/EuroSAT.zip. If it is not there, it tries to download it from the
Zenodo long-term archive; if that is blocked on your network, download it
yourself first with one of:

    curl -L -o data/EuroSAT.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
    curl -L -o data/EuroSAT.zip "https://madm.dfki.de/files/sentinel/EuroSAT.zip"

or via torchvision, which fetches the same imagery automatically:

    from torchvision.datasets import EuroSAT
    EuroSAT(root="data", download=True)

After extraction you will have:
    data/raw/EuroSAT_RGB/<ClassName>/<image>.jpg   (10 class subfolders, 27,000 images)
"""
import os
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "EuroSAT.zip")
URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
RAW_DIR = os.path.join(DATA_DIR, "raw")


def download(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract(zip_path: str, dest_dir: str) -> None:
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        try:
            download(URL, ZIP_PATH)
        except Exception as e:
            print(f"Automatic download failed ({e}).")
            print("Please download it manually and place it at:")
            print(f"  {ZIP_PATH}")
            print(f"URL: {URL}")
            sys.exit(1)

    target = os.path.join(RAW_DIR, "EuroSAT_RGB")
    if not os.path.isdir(target) or not os.listdir(target):
        extract(ZIP_PATH, RAW_DIR)
    else:
        print("Data already extracted, skipping.")

    print("Done. Class folders are under:", target)


if __name__ == "__main__":
    main()
