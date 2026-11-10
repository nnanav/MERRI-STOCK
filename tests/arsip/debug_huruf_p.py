"""
Debug mengapa huruf P pada Satuan 1 terpecah menjadi 2 segmen.
Menampilkan: batas kandidat, pasangan yang dinilai, dan jalur DP terpilih.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import (
    binarisasi, deteksi_komponen, calon_batas, skor_potongan, pilih_batas,
)

LAP = []
def c(t=""):
    LAP.append(str(t))

model, labels = muat_cnn()
templates, label_tm = muat_templates()

det = app.jalankan_deteksi(baca_gambar(BASE / "data_uji" / "nota_52.jpeg"))
f = next(x for x in det["fields"] if x["id"] == "satuan_1")
img = app.ke_tigakanal(f["crop_ocr"])

biner = binarisasi(img)
komponen = deteksi_komponen(biner)

tinggi = [h for _, _, _, h in komponen]
h_max = max(tinggi)
tinggi_acuan = float(np.median([h for h in tinggi if h >= h_max * 0.45]))

c(f"tinggi_acuan = {tinggi_acuan:.1f}")
c(f"jumlah komponen = {len(komponen)}")
c("\n=== KOMPONEN ===")
for x, y, w, h in komponen:
    c(f"  x={x:3d} y={y:3d} w={w:3d} h={h:3d}  (w/tinggi={w/tinggi_acuan:.2f})")

# Proses komponen pertama (huruf P)
x, y, w, h = komponen[0]
c(f"\n=== KOMPONEN PERTAMA (diduga 'P') ===")
c(f"x={x} y={y} w={w} h={h}")

blob = biner[y:y + h, x:x + w]
is_lebar = (w > 1.25 * tinggi_acuan and h > 0.40 * tinggi_acuan)
c(f"is_lebar = {is_lebar}  (w={w}, 1.25*tinggi={1.25*tinggi_acuan:.1f})")

batas = calon_batas(blob, tinggi_acuan) if is_lebar else [0, w]
c(f"\nBATAS KANDIDAT: {batas}")
c(f"  -> apakah ada batas di w={w} (tepi kanan)? {w in batas}")

rasio = w / max(tinggi_acuan, 1.0)
if rasio > 1.5:
    penalti = 0.15
elif rasio > 1.2:
    penalti = 0.35
else:
    penalti = 0.55
c(f"penalti_blob = {penalti} (rasio_lebar={rasio:.2f})")

nilai = skor_potongan(blob, batas, tinggi_acuan, model, labels,
                      templates, label_tm, penalti=penalti)

c("\n=== SEMUA PASANGAN YANG DINILAI (i,j) ===")
for (i, j), info in sorted(nilai.items()):
    skor, cc, tt, pc, pt = info
    c(f"  ({i},{j}) lebar={batas[j]-batas[i]:3d} "
      f"skor={skor:7.3f} cnn={cc!r} tm={tt!r} pc={pc:.2f} pt={pt:.2f}")

potong = pilih_batas(batas, nilai)
c("\n=== JALUR DP TERPILIH ===")
total = 0.0
for a, b, info in potong:
    skor, cc, tt, pc, pt = info
    total += skor
    c(f"  {a}-{b} (lebar {b-a:3d}) cnn={cc!r} skor={skor:.3f}")
c(f"  TOTAL SKOR = {total:.3f}")

Path("hasil_debug_p.txt").write_text("\n".join(LAP), encoding="utf-8")
print("selesai")