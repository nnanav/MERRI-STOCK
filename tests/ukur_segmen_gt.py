"""Ukur objektif: jumlah kotak segmentasi vs jumlah karakter ground truth.

Bila jumlah kotak < jumlah karakter (tanpa spasi), ada karakter yang
tergabung. Bila lebih banyak, ada karakter yang terpotong.
"""

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))

import app  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402
from processing.segmentasi_karakter import segmentasi_utama  # noqa: E402
from processing.pengenalan_karakter import muat_cnn, muat_templates  # noqa: E402
from tests_gt import GT  # noqa: E402

model, labels = muat_cnn()
templates, label_tm = muat_templates()

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

total_gt = 0
total_seg = 0
gabung = 0
lebih = 0
pas = 0

for nama, path in NOTA:
    if not path.exists():
        continue
    det = app.jalankan_deteksi(baca_gambar(path))
    print(f"\n=== {nama} ===")
    for f in det["fields"]:
        if f.get("jenis") not in ("penerima", "proyek", "produk"):
            continue
        gt = GT.get((nama, f["id"]))
        crop = f.get("crop_ocr")
        if not gt or crop is None or not getattr(crop, "size", 0):
            continue
        _, hasil, _ = segmentasi_utama(
            app.ke_tigakanal(crop), model, labels, templates, label_tm
        )
        n_gt = len(gt.replace(" ", ""))
        n_seg = len(hasil)
        total_gt += n_gt
        total_seg += n_seg
        beda = n_seg - n_gt
        if beda < 0:
            status = f"GABUNG {-beda}"
            gabung += -beda
        elif beda > 0:
            status = f"LEBIH {beda}"
            lebih += beda
        else:
            status = "PAS"
            pas += 1
        print(f"  {f['label']:15s} GT={n_gt:3d} kotak={n_seg:3d} -> {status}")

print("\n=== RINGKASAN ===")
print(f"Total karakter GT (tanpa spasi) : {total_gt}")
print(f"Total kotak segmentasi          : {total_seg}")
print(f"Selisih (kotak - GT)            : {total_seg - total_gt}")
print(f"Karakter hilang (tergabung)     : {gabung}")
print(f"Kotak berlebih (terpotong)      : {lebih}")
print(f"Field yang jumlahnya pas        : {pas}")
