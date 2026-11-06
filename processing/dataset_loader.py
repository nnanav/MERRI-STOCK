
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

EKSTENSI = {".png", ".jpg", ".jpeg"}

LABEL_SIMBOL = {
    "koma": ",",
    "titik": ".",
    "strip": "-",
    "minus": "-",
    "slash": "/",
    "persen": "%",
    "petik": '"',
    "kurung_buka": "(",
    "kurung_tutup": ")",
    "samadengan": "=",
    "titik_dua": ":"
}


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)

    gambar = cv2.imdecode(
        data,
        cv2.IMREAD_GRAYSCALE
    )

    return gambar


def normalisasi_karakter(gambar, ukuran=32):
    """
    Normalisasi karakter menjadi gambar 32x32.
    Karakter putih di atas latar belakang hitam.
    """

    if gambar is None or gambar.size == 0:
        return None

    _, binary = cv2.threshold(
        gambar,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    koordinat = cv2.findNonZero(binary)

    if koordinat is None:
        return None

    x, y, w, h = cv2.boundingRect(koordinat)

    karakter = binary[y:y+h, x:x+w]

    ruang = ukuran - 8

    skala = min(
        ruang / max(1, w),
        ruang / max(1, h)
    )

    w_baru = max(1, round(w * skala))
    h_baru = max(1, round(h * skala))

    karakter = cv2.resize(
        karakter,
        (w_baru, h_baru),
        interpolation=cv2.INTER_NEAREST
    )

    kanvas = np.zeros(
        (ukuran, ukuran),
        dtype=np.uint8
    )

    px = (ukuran - w_baru) // 2
    py = (ukuran - h_baru) // 2

    kanvas[
        py:py+h_baru,
        px:px+w_baru
    ] = karakter

    return kanvas


def daftar_folder_kelas():
    """
    Menghasilkan pasangan (folder, label).
    """

    hasil = []

    for jenis, alfabet in (
        ("uppercase", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("lowercase", "abcdefghijklmnopqrstuvwxyz")
    ):
        for karakter in alfabet:
            folder = (
                DATASET_DIR / "huruf" /
                jenis / karakter
            )
            hasil.append((folder, karakter))

    for angka in "0123456789":
        folder = DATASET_DIR / "angka" / angka
        hasil.append((folder, angka))

    for nama_folder, simbol in LABEL_SIMBOL.items():
        folder = DATASET_DIR / "simbol" / nama_folder
        hasil.append((folder, simbol))

    return hasil


def muat_dataset():
    """
    Membaca dataset gabungan.

    Return:
        X      : array (N, 32, 32, 1)
        y      : label numerik
        labels : daftar nama karakter
        info   : informasi file sumber
    """

    gambar_semua = []
    label_semua = []
    info = []

    for folder, label in daftar_folder_kelas():

        if not folder.exists():
            continue

        for file in sorted(folder.iterdir()):

            if not file.is_file():
                continue

            if file.suffix.lower() not in EKSTENSI:
                continue

            gambar = baca_gambar(file)

            if gambar is None:
                print("Gambar rusak:", file)
                continue

            normal = normalisasi_karakter(gambar)

            if normal is None:
                print("Gambar kosong:", file)
                continue

            gambar_semua.append(normal)
            label_semua.append(label)

            info.append({
                "file": str(file),
                "label": label
            })

    if not gambar_semua:
        return (
            np.empty((0, 32, 32, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
            []
        )

    labels = sorted(set(label_semua))

    label_ke_indeks = {
        label: i for i, label in enumerate(labels)
    }

    X = np.array(
        gambar_semua,
        dtype=np.float32
    )[..., np.newaxis] / 255.0

    y = np.array(
        [label_ke_indeks[label] for label in label_semua],
        dtype=np.int64
    )

    return X, y, labels, info


if __name__ == "__main__":

    X, y, labels, info = muat_dataset()

    print("\n=== DATASET BERHASIL DIBACA ===")
    print("Jumlah gambar:", len(X))
    print("Jumlah kelas:", len(labels))
    print("Ukuran X:", X.shape)
    print("Ukuran y:", y.shape)

    hitung = Counter(
        item["label"] for item in info
    )

    print("\n=== JUMLAH PER KARAKTER ===")

    for label in labels:
        jumlah = hitung[label]

        status = (
            "CUKUP" if jumlah >= 50
            else "KURANG"
        )

        print(f"{repr(label):5} : {jumlah:3d} - {status}")
