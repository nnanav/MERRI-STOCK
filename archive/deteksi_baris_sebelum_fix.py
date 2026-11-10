from pathlib import Path
import sys
import csv

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "hasil_deteksi_baris"
OUTPUT.mkdir(exist_ok=True)


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan(path, gambar):
    sukses, data = cv2.imencode(".png", gambar)
    if sukses:
        data.tofile(str(path))


def deteksi_baris(gambar):
    tinggi_asli, lebar_asli = gambar.shape[:2]

    # Samakan lebar pemrosesan agar parameter
    # relatif konsisten pada resolusi berbeda.
    skala = 1400 / lebar_asli

    kecil = cv2.resize(
        gambar,
        (1400, max(1, round(tinggi_asli * skala))),
        interpolation=(
            cv2.INTER_AREA if skala < 1
            else cv2.INTER_CUBIC
        )
    )

    abu = cv2.cvtColor(
        kecil, cv2.COLOR_BGR2GRAY
    )

    abu = cv2.GaussianBlur(
        abu, (3, 3), 0
    )

    biner = cv2.adaptiveThreshold(
        abu,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13
    )

    # Hilangkan garis horizontal panjang
    # seperti garis pembatas pada nota.
    panjang_garis = max(80, kecil.shape[1] // 8)

    kernel_garis = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (panjang_garis, 1)
    )

    garis = cv2.morphologyEx(
        biner,
        cv2.MORPH_OPEN,
        kernel_garis
    )

    bersih = cv2.subtract(biner, garis)

    # Sambungkan huruf dalam satu baris.
    # Ini untuk deteksi baris, bukan
    # untuk segmentasi karakter.
    kernel_baris = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (13, 3)
    )

    gabung = cv2.dilate(
        bersih,
        kernel_baris,
        iterations=1
    )

    jumlah, label, stats, pusat = (
        cv2.connectedComponentsWithStats(
            gabung,
            connectivity=8
        )
    )

    kandidat = []

    H, W = kecil.shape[:2]

    for i in range(1, jumlah):
        x, y, w, h, luas = stats[i]

        if w < 25 or h < 5:
            continue

        if h > H * 0.08:
            continue

        if w > W * 0.90:
            continue

        if luas < 60:
            continue

        # Tambah sedikit ruang agar huruf
        # tidak terpotong saat disimpan.
        margin_x = 5
        margin_y = 4

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(W, x + w + margin_x)
        y2 = min(H, y + h + margin_y)

        kandidat.append((x1, y1, x2, y2))

    kandidat.sort(key=lambda item: (item[1], item[0]))

    visual = kecil.copy()

    for nomor, (x1, y1, x2, y2) in enumerate(
        kandidat, start=1
    ):
        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            (0, 180, 0),
            2
        )

        cv2.putText(
            visual,
            str(nomor),
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

    return kecil, bersih, visual, kandidat


def main():
    if len(sys.argv) < 2:
        print(
            'Gunakan: python uji_deteksi_baris.py '
            '"lokasi_nota.jpeg"'
        )
        return

    path = Path(sys.argv[1])

    gambar = baca_gambar(path)

    if gambar is None:
        print("Gambar gagal dibaca:", path)
        return

    kecil, biner, visual, kandidat = (
        deteksi_baris(gambar)
    )

    simpan(OUTPUT / "biner.png", biner)
    simpan(OUTPUT / "deteksi_baris.png", visual)

    with open(
        OUTPUT / "daftar_baris.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["nomor", "x1", "y1", "x2", "y2"]
        )

        for nomor, kotak in enumerate(
            kandidat, start=1
        ):
            writer.writerow([nomor, *kotak])

            x1, y1, x2, y2 = kotak
            crop = kecil[y1:y2, x1:x2]

            simpan(
                OUTPUT / f"baris_{nomor:03d}.png",
                crop
            )

    print("\n=== DETEKSI BARIS SELESAI ===")
    print("Kandidat terdeteksi:", len(kandidat))
    print("Hasil:", OUTPUT)
    print("Periksa deteksi_baris.png")


if __name__ == "__main__":
    main()
