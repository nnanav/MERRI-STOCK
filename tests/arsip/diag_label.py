import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def main():
    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        H = det["kecil"].shape[0]
        W = det["kecil"].shape[1]
        print(f"\n===== {nama} W={W} H={H} =====")
        band = [k for k in det["kandidat"]
                if 0.12 * H <= k["cy"] <= 0.27 * H]
        for k in sorted(band, key=lambda z: (z["cy"], z["x1"])):
            print("cy=%4.0f x=%4d-%4d w=%3d h=%2d  cy%%=%.2f x%%=%.2f" %
                  (k["cy"], k["x1"], k["x2"], k["w"], k["h"],
                   k["cy"] / H, k["x1"] / W))


if __name__ == "__main__":
    main()