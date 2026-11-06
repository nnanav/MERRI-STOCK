import json
from pathlib import Path

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent.parent

ACUAN = BASE / "data_nota_asli" / "nota_01.jpeg"
KONFIG = BASE / "hasil_evaluasi" / "area_field.json"


def baca_gambar(path):
    path = Path(path)

    if not path.is_file():
        return None

    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


def transformasi_kotak(
    kotak,
    matriks,
    ukuran,
    margin=(0, 0, 0, 0)
):
    x, y, w, h = map(float, kotak)

    kiri, atas, kanan, bawah = margin

    titik = np.array([
        [x - kiri, y - atas],
        [x + w + kanan, y - atas],
        [x + w + kanan, y + h + bawah],
        [x - kiri, y + h + bawah]
    ], dtype=np.float32).reshape(-1, 1, 2)

    hasil = cv2.perspectiveTransform(
        titik,
        matriks
    ).reshape(-1, 2)

    if not np.isfinite(hasil).all():
        return None

    H, W = ukuran[:2]

    x1 = max(
        0,
        int(np.floor(hasil[:, 0].min()))
    )

    y1 = max(
        0,
        int(np.floor(hasil[:, 1].min()))
    )

    x2 = min(
        W,
        int(np.ceil(hasil[:, 0].max()))
    )

    y2 = min(
        H,
        int(np.ceil(hasil[:, 1].max()))
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def cari_area_field(gambar_baru):
    """
    Pencocokan lokasi nota menggunakan
    ORB dan homografi.

    Mengembalikan area kandidat field.
    Belum melakukan segmentasi karakter.
    """

    if gambar_baru is None or gambar_baru.size == 0:
        return None, "Gambar nota kosong."

    if not ACUAN.is_file():
        return None, f"Gambar acuan tidak ada: {ACUAN}"

    if not KONFIG.is_file():
        return None, f"Kalibrasi tidak ada: {KONFIG}"

    acuan = baca_gambar(ACUAN)

    if acuan is None:
        return None, "Gambar acuan gagal dibaca."

    try:
        with open(
            KONFIG,
            encoding="utf-8-sig"
        ) as file:
            konfigurasi = json.load(file)

    except (OSError, ValueError) as error:
        return None, f"Kalibrasi rusak: {error}"

    gray1 = cv2.cvtColor(
        acuan,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        gambar_baru,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=5000
    )

    k1, d1 = orb.detectAndCompute(
        gray1,
        None
    )

    k2, d2 = orb.detectAndCompute(
        gray2,
        None
    )

    if d1 is None or d2 is None:
        return None, "Fitur ORB tidak ditemukan."

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pasangan = matcher.knnMatch(
        d1,
        d2,
        k=2
    )

    cocok = []

    for item in pasangan:
        if len(item) != 2:
            continue

        m, n = item

        if m.distance < 0.70 * n.distance:
            cocok.append(m)

    if len(cocok) < 20:
        return None, (
            f"Kecocokan ORB kurang: {len(cocok)}"
        )

    titik1 = np.float32([
        k1[m.queryIdx].pt
        for m in cocok
    ]).reshape(-1, 1, 2)

    titik2 = np.float32([
        k2[m.trainIdx].pt
        for m in cocok
    ]).reshape(-1, 1, 2)

    matriks, inlier = cv2.findHomography(
        titik1,
        titik2,
        cv2.RANSAC,
        5.0
    )

    if matriks is None or inlier is None:
        return None, "Homografi gagal."

    jumlah_inlier = int(inlier.sum())
    rasio_inlier = jumlah_inlier / len(cocok)

    if (
        jumlah_inlier < 15
        or rasio_inlier < 0.38
    ):
        return None, (
            "Kecocokan homografi belum meyakinkan."
        )

    # Periksa apakah ukuran transformasi masuk akal
    h0, w0 = acuan.shape[:2]

    sudut = np.float32([
        [0, 0],
        [w0, 0],
        [w0, h0],
        [0, h0]
    ]).reshape(-1, 1, 2)

    quad = cv2.perspectiveTransform(
        sudut,
        matriks
    ).reshape(-1, 2)

    if not np.isfinite(quad).all():
        return None, "Transformasi tidak valid."

    luas = abs(
        cv2.contourArea(
            quad.astype(np.float32)
        )
    )

    luas_gambar = float(
        gambar_baru.shape[0] * gambar_baru.shape[1]
    )

    if not (
        0.18 * luas_gambar
        <= luas
        <= 2.0 * luas_gambar
    ):
        return None, "Skala transformasi tidak wajar."

    lokasi = {}

    for nama, kotak in konfigurasi.items():

        if not isinstance(kotak, list) or len(kotak) != 4:
            continue

        # Area tanggal diperluas secara konservatif
        # karena posisi tulisan dapat bergeser
        # relatif terhadap kertas dan tanda tangan.
        if nama == "tanggal":
            margin = (65, 22, 15, 10)
        else:
            margin = (0, 0, 0, 0)

        hasil = transformasi_kotak(
            kotak,
            matriks,
            gambar_baru.shape,
            margin
        )

        if hasil is None:
            continue

        x1, y1, x2, y2 = hasil

        lokasi[nama] = {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "gambar": gambar_baru[
                y1:y2,
                x1:x2
            ].copy()
        }

    if "tanggal" not in lokasi:
        return None, (
            "Wilayah tanggal berada di luar gambar."
        )

    return lokasi, None
