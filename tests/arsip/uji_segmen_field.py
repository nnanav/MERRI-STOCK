import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar, simpan_gambar

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]
OUT = BASE / "output" / "segmentasi_final"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            if f["jenis"] not in ("proyek", "penerima", "produk", "satuan"):
                continue
            crop = f["crop_show"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            h, w = crop.shape[:2]
            vis, kar, biner, teks, _ = app.segmen_field(crop)
            if not kar:
                print(f"  {f['label']:16s} TIDAK ADA KOTAK")
                continue
            xs = [k["x2"] for k in kar]
            tinggi = [k["y2"] - k["y1"] for k in kar]
            ta = float(np.median(tinggi)) if tinggi else 1
            lebar = [(k["x2"] - k["x1"]) / ta for k in kar]
            nempel = sum(1 for z in lebar if z > 1.05)
            print(f"  {f['label']:16s} n={len(kar):3d} maks={max(lebar):.2f} "
                  f"nempel={nempel} x2maks={max(xs)}/{w} teks={teks!r}")
            simpan_gambar(OUT / f"{nama}_{f['id']}_box.png", vis)


if __name__ == "__main__":
    main()