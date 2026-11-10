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
        W = det["kecil"].shape[1]
        H = det["kecil"].shape[0]
        print(f"\n===== {nama} W={W} H={H} =====")
        for i, p in enumerate(det["produk"], 1):
            print(f"  produk {i}: x={p['x1']}-{p['x2']} cy={p['cy']:.0f} h={p['h']}")
            kanan = [k for k in det["kandidat"]
                     if abs(k["cy"] - p["cy"]) <= max(8, 0.5 * p["h"])
                     and k["x1"] >= 0.70 * W]
            for k in sorted(kanan, key=lambda z: z["x1"]):
                print("     kanan: x=%4d-%4d cy=%4.0f w=%3d" %
                      (k["x1"], k["x2"], k["cy"], k["w"]))


if __name__ == "__main__":
    main()