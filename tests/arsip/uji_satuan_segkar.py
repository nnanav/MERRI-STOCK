import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import (
    baca_gambar, simpan_gambar, segmentasi_per_karakter,
)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]
OUT = BASE / "output" / "segmentasi_perchar"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} : {len(det['fields'])} field =====")
        for f in det["fields"]:
            if f["jenis"] in ("jumlah", "satuan", "harga"):
                crop = f["crop_show"]
                ada = "ADA" if crop is not None and getattr(crop, "size", 0) else "TIDAK"
                print(f"  {f['label']:16s} crop={ada} bbox={f['bbox']}")
                if crop is not None and getattr(crop, "size", 0):
                    simpan_gambar(OUT / f"{nama}_{f['id']}.png", crop)

        for f in det["fields"]:
            if f["jenis"] in ("proyek", "penerima", "produk"):
                crop = f["crop_show"]
                if crop is None or not getattr(crop, "size", 0):
                    continue
                cimg = app.ke_tigakanal(crop)
                hasil = segmentasi_per_karakter(cimg)
                kar = hasil["karakter"]
                vis = cimg.copy()
                for i, k in enumerate(kar):
                    c = (0, 170, 0) if i % 2 == 0 else (0, 0, 255)
                    cv2.rectangle(vis, (k["x1"], k["y1"]), (k["x2"], k["y2"]), c, 1)
                simpan_gambar(OUT / f"{nama}_{f['id']}_box.png", vis)
                tinggi = [k["y2"] - k["y1"] for k in kar]
                ta = float(np.median(tinggi)) if tinggi else 1.0
                lebar = [(k["x2"] - k["x1"]) / ta for k in kar]
                nempel = sum(1 for z in lebar if z > 1.05)
                print(f"  SEGKAR {f['label']:16s} n={len(kar):3d} "
                      f"maks={max(lebar) if lebar else 0:.2f} nempel={nempel}")


if __name__ == "__main__":
    main()