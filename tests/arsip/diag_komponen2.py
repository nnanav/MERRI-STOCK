"""
Mencari semua komponen di sekitar jumlah 1 nota_52 (tanpa batas x),
untuk mengetahui di mana "3." sebenarnya berada.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

from processing.deteksi_baris import baca_gambar, simpan

LAPORAN = []
def catat(t=""):
    LAPORAN.append(str(t))

gambar = baca_gambar(BASE / "data_uji" / "nota_52.jpeg")
tinggi_asli, lebar_asli = gambar.shape[:2]
skala = 1400 / lebar_asli
kecil = cv2.resize(
    gambar, (1400, max(1, round(tinggi_asli * skala))),
    interpolation=cv2.INTER_AREA if skala < 1 else cv2.INTER_CUBIC,
)
abu = cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY)
abu = cv2.GaussianBlur(abu, (3, 3), 0)
biner = cv2.adaptiveThreshold(
    abu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 31, 13
)

n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)

catat("=== SEMUA KOMPONEN pada y=248..290, x=850..1080 ===")
for i in range(1, n):
    x, y, w, h, luas = stats[i]
    if not (240 <= y <= 300):
        continue
    if not (850 <= x <= 1080):
        continue
    if luas < 20:
        continue
    catat(f"  x={x:4d}-{x+w:4d} y={y:3d}-{y+h:3d} w={w:3d} h={h:2d} luas={luas:4d}")

# Simpan visual area luas untuk melihat tata letak sebenarnya
potong = kecil[240:305, 850:1080]
simpan(BASE / "output" / "diagnosis_sisa" / "area_luas_jumlah1.png", potong)
catat("\nvisual: area_luas_jumlah1.png (x=850..1080, y=240..305)")

Path("hasil_komponen2.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("selesai")