"""
Mencari token "3." pada baris jumlah 1 nota_52 dengan memindai
seluruh lebar pada band y yang sama.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

from processing.deteksi_baris import baca_gambar

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

n, _, stats, pusat = cv2.connectedComponentsWithStats(biner, connectivity=8)

# Band dua baris: y 255..280 (baris atas) dan 278..300 (baris bawah)
for label_band, (y_lo, y_hi) in (
    ("BARIS ATAS  y=255..280", (255, 280)),
    ("BARIS BAWAH y=278..300", (278, 300)),
):
    catat(f"\n=== {label_band} : semua komponen x>700 ===")
    baris = []
    for i in range(1, n):
        x, y, w, h, luas = stats[i]
        if luas < 25:
            continue
        cy = y + h / 2
        if not (y_lo <= cy <= y_hi):
            continue
        if x < 700:
            continue
        baris.append((x, x + w, y, h, luas))

    baris.sort()
    for x1, x2, y, h, luas in baris:
        catat(f"  x={x1:4d}-{x2:4d} y={y:3d} h={h:2d} luas={luas:4d}")

Path("hasil_komponen3.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("selesai")