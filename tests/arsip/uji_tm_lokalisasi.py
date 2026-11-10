import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import baca_gambar, simpan_gambar
from processing.lokalisasi_template import (
    muat_template_manual, cari_kandidat_karakter,
)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]
OUT = BASE / "output" / "uji_tm_lokalisasi"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    templates, labels = muat_template_manual()
    print("template manual termuat:", len(templates))
    from collections import Counter
    print("jumlah kelas:", len(set(labels)))

    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            if f["jenis"] not in ("nomor", "jumlah", "satuan", "harga", "produk"):
                continue
            crop = f["crop_show"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            cimg = app.ke_tigakanal(crop)
            kand = cari_kandidat_karakter(cimg)
            vis = cimg.copy()
            for k in kand:
                cv2.rectangle(vis, (k["x1"], k["y1"]), (k["x2"], k["y2"]),
                              (0, 200, 0), 1)
            simpan_gambar(OUT / f"{nama}_{f['id']}.png", vis)
            label_tm = "".join(k["template"] for k in kand)
            print(f"  {f['label']:16s} kandidat={len(kand):3d} "
                  f"tm={label_tm!r}")


if __name__ == "__main__":
    main()