import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.segmentasi_karakter import segmentasi_karakter

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def main():
    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            if f["jenis"] not in ("produk", "proyek", "penerima"):
                continue
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)
            biner, hasil = segmentasi_karakter(img)
            if not hasil:
                continue
            tinggi = [s["y2"] - s["y1"] for s in hasil]
            ta = float(np.median(tinggi))
            lebar = [(s["x2"] - s["x1"]) / ta for s in hasil]
            nempel = [z for z in lebar if z > 1.05]
            print(f"  {f['label']:16s} n={len(hasil):3d} maks={max(lebar):.2f} "
                  f"nempel={len(nempel)}")
            # tampilkan lebar tiap segmen untuk melihat mana yang menempel
            if nempel:
                detail = ", ".join(f"{z:.2f}" for z in lebar)
                print(f"     lebar: {detail}")


if __name__ == "__main__":
    main()