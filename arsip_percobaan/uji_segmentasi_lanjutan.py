
from pathlib import Path
import sys
import csv

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "hasil_segmentasi_lanjutan"

# Koordinat khusus untuk nota_52.jpeg
# Format: (x1, y1, x2, y2)
ROI_PRODUK = [
    (525, 252, 870, 280),
    (525, 276, 875, 303)
]


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan_gambar(path, gambar):
    sukses, buffer = cv2.imencode(".png", gambar)

    if sukses:
        buffer.tofile(str(path))


def binarisasi(gambar):
    abu = cv2.cvtColor(
        gambar, cv2.COLOR_BGR2GRAY
    )

    # Perbesar untuk membantu pemisahan
    abu = cv2.resize(
        abu,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    abu = cv2.GaussianBlur(
        abu, (3, 3), 0
    )

    _, biner = cv2.threshold(
        abu,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return biner


def cari_kelompok(biner):
    """
    Cari kelompok piksel karakter berdasarkan
    proyeksi kolom.
    """

    proyeksi = np.sum(
        biner > 0,
        axis=0
    )

    aktif = proyeksi > 0

    kelompok = []
    mulai = None

    for x, ada in enumerate(aktif):

        if ada and mulai is None:
            mulai = x

        elif not ada and mulai is not None:
            kelompok.append((mulai, x))
            mulai = None

    if mulai is not None:
        kelompok.append((mulai, len(aktif)))

    return kelompok, proyeksi


def potong_kelompok_lebar(
    awal,
    akhir,
    proyeksi,
    lebar_target=34
):
    """
    Mencari batas pemisahan pada kelompok
    yang terlalu lebar.

    Ini masih pendekatan heuristik.
    """

    lebar = akhir - awal

    if lebar <= lebar_target * 1.4:
        return [(awal, akhir)]

    jumlah_potong = max(
        2,
        round(lebar / lebar_target)
    )

    titik = [awal]

    for i in range(1, jumlah_potong):

        estimasi = awal + (
            lebar * i / jumlah_potong
        )

        radius = max(
            4,
            int(lebar_target * 0.35)
        )

        kiri = max(
            awal + 2,
            int(estimasi - radius)
        )

        kanan = min(
            akhir - 2,
            int(estimasi + radius)
        )

        if kanan <= kiri:
            continue

        kandidat = np.argmin(
            proyeksi[kiri:kanan]
        )

        posisi = kiri + int(kandidat)

        if posisi - titik[-1] >= 5:
            titik.append(posisi)

    titik.append(akhir)

    hasil = []

    for a, b in zip(titik[:-1], titik[1:]):
        if b - a >= 3:
            hasil.append((a, b))

    return hasil


def segmentasi_karakter(biner):
    kelompok, proyeksi = cari_kelompok(biner)

    hasil = []

    for awal, akhir in kelompok:
        lebar = akhir - awal

        if lebar < 3:
            continue

        bagian = potong_kelompok_lebar(
            awal,
            akhir,
            proyeksi
        )

        hasil.extend(bagian)

    return sorted(hasil)


def buat_visualisasi(gambar, segmen):
    hasil = gambar.copy()

    tinggi = hasil.shape[0]

    for awal, akhir in segmen:

        cv2.rectangle(
            hasil,
            (awal, 0),
            (akhir, tinggi - 1),
            (0, 0, 255),
            2
        )

    return hasil


def proses_produk(gambar, nomor):
    crop = gambar.copy()

    biner = binarisasi(crop)

    # Gambar crop diperbesar sesuai ukuran biner
    besar = cv2.resize(
        crop,
        (biner.shape[1], biner.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    segmen = segmentasi_karakter(biner)

    visual = buat_visualisasi(
        besar,
        segmen
    )

    simpan_gambar(
        OUTPUT / f"produk_{nomor}_crop.png",
        besar
    )

    simpan_gambar(
        OUTPUT / f"produk_{nomor}_biner.png",
        biner
    )

    simpan_gambar(
        OUTPUT / f"produk_{nomor}_batas.png",
        visual
    )

    return segmen


def main():
    if len(sys.argv) < 2:
        print(
            'Penggunaan: python uji_segmentasi_lanjutan.py '
            '"path_nota.jpeg"'
        )
        return

    path = Path(sys.argv[1])

    gambar = baca_gambar(path)

    if gambar is None:
        print("Gagal membaca gambar:", path)
        return

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    semua = []

    for nomor, (x1, y1, x2, y2) in enumerate(
        ROI_PRODUK,
        start=1
    ):
        crop = gambar[y1:y2, x1:x2]

        if crop.size == 0:
            print("Crop kosong pada produk", nomor)
            continue

        segmen = proses_produk(
            crop,
            nomor
        )

        print(
            f"Produk {nomor}: "
            f"{len(segmen)} calon segmen"
        )

        for urutan, (awal, akhir) in enumerate(
            segmen,
            start=1
        ):
            semua.append({
                "produk": nomor,
                "segmen": urutan,
                "x_awal": awal,
                "x_akhir": akhir
            })

    with open(
        OUTPUT / "calon_karakter.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "produk",
                "segmen",
                "x_awal",
                "x_akhir"
            ]
        )

        writer.writeheader()
        writer.writerows(semua)

    print("\nHasil disimpan di:", OUTPUT)
    print(
        "Periksa produk_1_batas.png "
        "dan produk_2_batas.png"
    )


if __name__ == "__main__":
    main()
