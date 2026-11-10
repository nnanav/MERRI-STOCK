import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2

import app
from processing.pemrosesan_field import baca_gambar, simpan_gambar

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

OUT = BASE / "output" / "diagnostik"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    for nama, path in NOTA:
        if not path.exists():
            print(nama, "tidak ada")
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        kecil = det["kecil"]
        H, W = kecil.shape[:2]
        print(f"\n===== {nama}  W={W} H={H} =====")
        for k in sorted(det["kandidat"], key=lambda z: z["cy"]):
            if k["cy"] < 0.45 * H:
                print("x=%4d-%4d cy=%4.0f w=%3d h=%2d" %
                      (k["x1"], k["x2"], k["cy"], k["w"], k["h"]))
        # crop area label penerima/proyek untuk dilihat
        reg = kecil[int(0.14 * H):int(0.28 * H), int(0.20 * W):int(0.85 * W)]
        simpan_gambar(OUT / f"{nama}_area_penerima_proyek.png", reg)
        # crop area kolom harga kanan
        reg2 = kecil[int(0.17 * H):int(0.62 * H), int(0.68 * W):W]
        simpan_gambar(OUT / f"{nama}_area_harga.png", reg2)


if __name__ == "__main__":
    main()