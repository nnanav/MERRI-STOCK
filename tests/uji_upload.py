"""Uji jalur upload: decode normal, decode gagal, dan alur penuh."""

import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402
from processing.ground_truth import gt_field  # noqa: E402

# 1. Decode gambar valid (meniru st.file_uploader -> np.frombuffer -> imdecode)
path = BASE / "data_uji" / "nota_26.jpeg"
mentah = path.read_bytes()
data = np.frombuffer(mentah, np.uint8)
gambar = cv2.imdecode(data, cv2.IMREAD_COLOR)
print(f"1. decode valid   : {'OK' if gambar is not None else 'GAGAL'} "
      f"shape={None if gambar is None else gambar.shape}")
assert gambar is not None

# 2. Decode data rusak harus mengembalikan None (bukan crash)
data_rusak = np.frombuffer(b"bukan gambar sama sekali", np.uint8)
hasil_rusak = cv2.imdecode(data_rusak, cv2.IMREAD_COLOR)
print(f"2. decode rusak   : {hasil_rusak} (harus None, tanpa crash)")
assert hasil_rusak is None

# 3. Alur penuh: deteksi -> segmentasi -> OCR pada gambar hasil upload
det = app.jalankan_deteksi(gambar)
print(f"3. deteksi        : {len(det['fields'])} field")
assert det["fields"]

n_ok = 0
for f in det["fields"]:
    crop = f.get("crop_ocr")
    if crop is None or not getattr(crop, "size", 0):
        continue
    r = app.kenali_crop(crop, pakai_clahe=False, jenis=f.get("jenis"))
    if r.get("status") == "ok":
        n_ok += 1
print(f"4. OCR field ok   : {n_ok}/{len(det['fields'])}")
assert n_ok >= 1

# 5. Ground truth untuk nota ini tetap bisa dievaluasi
gt = gt_field("nota_26.jpeg", "nomor")
r = app.hitung_cer_rinci(gt, "0099")
print(f"5. CER nomor      : {r['cer']*100:.2f}% (GT={gt!r})")
assert r["cer"] == 0.0

print("\nSemua pemeriksaan upload lolos tanpa error.")
