"""
Diagnosis per karakter pada field Satuan (kasus 'Pcs' -> 'fscs').

Menampilkan: jumlah segmen, ukuran tiap segmen, prediksi CNN, prediksi TM.
Menyimpan gambar tiap segmen untuk diperiksa.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import baca_gambar, simpan_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter, SKALA

OUT = BASE / "output" / "diag_karakter"
OUT.mkdir(parents=True, exist_ok=True)

LAPORAN = []
def catat(t=""):
    LAPORAN.append(str(t))

model, labels = muat_cnn()
templates, label_tm = muat_templates()

det = app.jalankan_deteksi(baca_gambar(BASE / "data_uji" / "nota_52.jpeg"))

TARGET = ("satuan_1", "satuan_2", "jumlah_1", "jumlah_2", "harga_1")

for f in det["fields"]:
    if f["id"] not in TARGET:
        continue
    crop = f["crop_ocr"]
    if crop is None or not getattr(crop, "size", 0):
        continue

    catat(f"\n{'='*66}")
    catat(f"{f['label']}  (bbox={f['bbox']})")
    catat(f"{'='*66}")

    img = app.ke_tigakanal(crop)
    biner, segmen = segmentasi_karakter(
        img, model, labels, templates, label_tm
    )
    simpan_gambar(OUT / f"{f['id']}_crop.png", crop)
    simpan_gambar(OUT / f"{f['id']}_biner.png", biner)

    besar = cv2.resize(
        img, None, fx=SKALA, fy=SKALA, interpolation=cv2.INTER_CUBIC
    )

    catat(f"Jumlah segmen: {len(segmen)}")
    catat(f"{'no':3s} {'x1-x2':12s} {'y1-y2':12s} {'w':3s} {'h':3s} "
          f"{'cnn':4s} {'tm':4s} {'pc':5s} {'pt':5s}")

    for i, s in enumerate(segmen, 1):
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        w, h = x2 - x1, y2 - y1
        catat(f"{i:<3d} {x1:4d}-{x2:<6d} {y1:4d}-{y2:<6d} {w:<3d} {h:<3d} "
              f"{s.get('cnn',''):4s} {s.get('tm',''):4s} "
              f"{s.get('pc',0):<5.2f} {s.get('pt',0):<5.2f}")

        # simpan potongan per karakter
        aa = max(0, min(besar.shape[0] - 1, y1))
        bb = max(aa + 1, min(besar.shape[0], y2))
        cc = max(0, min(besar.shape[1] - 1, x1))
        dd = max(cc + 1, min(besar.shape[1], x2))
        potong = besar[aa:bb, cc:dd]
        if potong.size:
            simpan_gambar(OUT / f"{f['id']}_kar{i:02d}.png", potong)

Path("hasil_diag_karakter.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("selesai -> hasil_diag_karakter.txt")