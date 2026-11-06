
from pathlib import Path
import sys
import cv2
import numpy as np

from uji_deteksi_baris import deteksi_baris
from processing.deteksi_field import deteksi_produk


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "hasil_uji_produk"


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan(path, gambar):
    sukses, data = cv2.imencode(".png", gambar)
    if sukses:
        data.tofile(str(path))


def main():
    if len(sys.argv) < 2:
        print(
            'Gunakan: python uji_produk_otomatis.py '
            '"path_nota.jpeg"'
        )
        return

    path = Path(sys.argv[1]).resolve()
    gambar = baca_gambar(path)

    if gambar is None:
        print("Gagal membaca:", path)
        return

    # Semua tahapan memakai gambar yang sama
    kecil, biner, visual_baris, kotak = deteksi_baris(
        gambar
    )

    H, W = kecil.shape[:2]

    kandidat = []

    for x1, y1, x2, y2 in kotak:
        kandidat.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "w": x2 - x1,
            "h": y2 - y1
        })

    produk = deteksi_produk(kandidat, W, H)

    # Folder berbeda untuk setiap nota
    folder = OUTPUT / path.stem
    folder.mkdir(parents=True, exist_ok=True)

    visual = kecil.copy()

    for k in kandidat:
        cv2.rectangle(
            visual,
            (k["x1"], k["y1"]),
            (k["x2"], k["y2"]),
            (180, 180, 180),
            1
        )

    for nomor, p in enumerate(produk, 1):
        x1, y1 = p["x1"], p["y1"]
        x2, y2 = p["x2"], p["y2"]

        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2
        )

        cv2.putText(
            visual,
            f"Produk {nomor}",
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        crop = kecil[y1:y2, x1:x2]
        simpan(
            folder / f"nama_produk_{nomor}.png",
            crop
        )

        print(
            f"Produk {nomor}: "
            f"x={x1}-{x2}, y={y1}-{y2}"
        )

    simpan(folder / "deteksi_baris.png", visual_baris)
    simpan(folder / "deteksi_produk.png", visual)

    print("\n=== HASIL PENGUJIAN ===")
    print("Nota:", path.name)
    print("Kandidat baris:", len(kandidat))
    print("Calon produk:", len(produk))
    print("Hasil disimpan:", folder)


if __name__ == "__main__":
    main()
