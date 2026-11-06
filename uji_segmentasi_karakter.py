from pathlib import Path

import cv2

from processing.pemrosesan_field import (
    baca_gambar, simpan_gambar, segmentasi_per_karakter,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "diagnostik_v3"
OUT.mkdir(exist_ok=True)

DAFTAR = [
    "diagnostik_v2/nota_52/produk_1_sesudah.png",
    "diagnostik_v2/nota_52/jumlah_1.png",
    "diagnostik_v2/nota_52/satuan_1.png",
]


def main():
    for rel in DAFTAR:
        path = BASE / rel
        if not path.exists():
            print("TIDAK ADA:", rel)
            continue
        img = baca_gambar(path)
        if img is None:
            print("GAGAL BACA:", rel)
            continue
        cimg = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hasil = segmentasi_per_karakter(cimg)

        vis = cimg.copy()
        for i, k in enumerate(hasil["karakter"]):
            warna = (0, 170, 0) if i % 2 == 0 else (0, 0, 255)
            cv2.rectangle(vis, (k["x1"], k["y1"]), (k["x2"], k["y2"]), warna, 1)

        simpan_gambar(OUT / ("segkar_" + path.stem + ".png"), vis)
        print(f"{rel} -> jumlah karakter = {len(hasil['karakter'])}")


if __name__ == "__main__":
    main()