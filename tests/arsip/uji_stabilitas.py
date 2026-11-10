"""
Menguji kestabilan hasil OCR: menjalankan pengenalan beberapa kali
pada crop yang sama untuk melihat apakah hasilnya konsisten.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar

LAP = []

det = app.jalankan_deteksi(baca_gambar(BASE / "data_uji" / "nota_52.jpeg"))
f = next(x for x in det["fields"] if x["id"] == "satuan_1")
crop = f["crop_ocr"]

LAP.append("Uji 5 kali pada crop Satuan 1 nota_52:")
for i in range(5):
    r = app.kenali_crop(crop, pakai_clahe=False)
    LAP.append(f"  run {i+1}: {r['teks']!r}  n={r['n']}")

# Uji juga field lain yang rawan
for fid in ("satuan_2", "jumlah_2", "harga_1"):
    f2 = next((x for x in det["fields"] if x["id"] == fid), None)
    if f2 is None or f2["crop_ocr"] is None:
        continue
    LAP.append(f"\n{f2['label']}:")
    for i in range(3):
        r = app.kenali_crop(f2["crop_ocr"], pakai_clahe=False)
        LAP.append(f"  run {i+1}: {r['teks']!r}")

Path("hasil_stabil.txt").write_text("\n".join(LAP), encoding="utf-8")
print("selesai")