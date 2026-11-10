"""
Diagnosis sisa kesalahan pada nota_52:
1. Apakah crop Jumlah sudah memuat "3." (masalah deteksi/crop)?
2. Karakter apa yang salah dibaca CNN (masalah model)?
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

OUT = BASE / "output" / "diagnosis_sisa"
OUT.mkdir(parents=True, exist_ok=True)

LAPORAN = []
def catat(teks=""):
    LAPORAN.append(str(teks))

det = app.jalankan_deteksi(baca_gambar(BASE / "data_uji" / "nota_52.jpeg"))
kecil = det["kecil"]
W = kecil.shape[1]

catat("=== FIELD YANG DIPERIKSA ===")
for f in det["fields"]:
    if f["jenis"] not in ("jumlah", "satuan", "harga", "produk"):
        continue
    crop = f["crop_ocr"]
    ada = "ADA" if crop is not None and getattr(crop, "size", 0) else "TIDAK"
    bb = f["bbox"]
    catat(f"\n{f['label']:16s} bbox={bb} crop={ada}")
    if crop is None or not getattr(crop, "size", 0):
        continue
    simpan_gambar(OUT / f"{f['id']}.png", crop)
    catat(f"   crop w={crop.shape[1]} h={crop.shape[0]}")

catat("\n=== TOKEN DI SEKITAR BARIS PRODUK 1 ===")
produk = det["produk"][0]
catat(f"produk_1: x={produk['x1']}-{produk['x2']} cy={produk['cy']:.0f}")

dekat = [k for k in det["kandidat"] if abs(k["cy"] - produk["cy"]) <= 20]
for k in sorted(dekat, key=lambda z: z["x1"]):
    catat(f"   x={k['x1']:4d}-{k['x2']:4d} cy={k['cy']:4.0f} w={k['w']:3d}")

catat("\n=== AREA KIRI CROP JUMLAH ===")
for f in det["fields"]:
    if f["jenis"] != "jumlah":
        continue
    x1, y1, x2, y2 = f["bbox"]
    catat(f"{f['label']}: bbox x={x1}-{x2}")
    kiri = kecil[max(0, y1 - 3):y2 + 3, max(0, x1 - 70):x1]
    if kiri.size:
        simpan_gambar(OUT / f"kiri_{f['id']}.png", kiri)
        catat(f"   disimpan kiri_{f['id']}.png")

Path("hasil_diagnosis.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("selesai -> hasil_diagnosis.txt")