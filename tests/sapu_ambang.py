"""Menyapu nilai ambang lebar komponen untuk mencari CER terbaik."""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "tests") not in sys.path:
    sys.path.insert(0, str(BASE / "tests"))

import app
import processing.segmentasi_karakter as sk
from processing.pemrosesan_field import baca_gambar
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

# Deteksi field sekali saja; crop tidak berubah saat ambang diubah.
data = []
for nama, path in NOTA:
    if not path.exists():
        continue
    det = app.jalankan_deteksi(baca_gambar(path))
    for f in det["fields"]:
        gt = GT.get((nama, f["id"]))
        if not gt:
            continue
        data.append((nama, f["label"], gt, f["crop_ocr"], f.get("jenis")))

print(f"{'ambang':>8} {'CER':>8} {'benar':>7}")
for nilai in (1.08, 1.10, 1.12):
    sk.AMBANG_LEBAR_KOMPONEN = nilai
    total = 0
    jumlah_cer = 0.0
    benar = 0
    for _nama, _label, gt, crop, jenis in data:
        if crop is None or not getattr(crop, "size", 0):
            pred = ""
        else:
            # cache dimatikan agar segmentasi dijalankan ulang
            pred = app.kenali_crop(crop, pakai_clahe=False, jenis=jenis)["teks"]
        cer = app.hitung_cer(gt, pred)
        total += 1
        jumlah_cer += cer
        benar += int(pred.strip() == gt.strip())
    print(f"{nilai:>8.2f} {jumlah_cer / max(1, total):>8.4f} {benar:>7}")
