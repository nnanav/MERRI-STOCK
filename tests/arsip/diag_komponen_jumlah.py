"""
Mencari mengapa token "3." pada Jumlah 1 nota_52 tidak terdeteksi.
Melihat semua komponen connected-component di sekitar baris produk 1,
sebelum dan sesudah filter.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

from processing.deteksi_baris import (
    baca_gambar as baca, simpan, deteksi_baris,
)

LAPORAN = []


def catat(t=""):
    LAPORAN.append(str(t))


gambar = baca(BASE / "data_uji" / "nota_52.jpeg")
tinggi_asli, lebar_asli = gambar.shape[:2]

skala = 1400 / lebar_asli
kecil = cv2.resize(
    gambar,
    (1400, max(1, round(tinggi_asli * skala))),
    interpolation=cv2.INTER_AREA if skala < 1 else cv2.INTER_CUBIC,
)

abu = cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY)
abu = cv2.GaussianBlur(abu, (3, 3), 0)
biner = cv2.adaptiveThreshold(
    abu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 31, 13
)

H, W = kecil.shape[:2]

# Baris produk 1 berada di sekitar y=256..277 (dari diagnosis sebelumnya)
y_atas, y_bawah = 250, 285
catat(f"Lebar citra = {W}, area baris y={y_atas}..{y_bawah}")

# Komponen mentah (tanpa filter) di area tersebut
n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
catat("\n=== KOMPONEN MENTAH di area baris produk 1 (x 850..1050) ===")
for i in range(1, n):
    x, y, w, h, luas = stats[i]
    if not (y_atas <= y <= y_bawah):
        continue
    if not (850 <= x <= 1050):
        continue
    lulus_w = w >= 25
    lulus_h = h >= 5
    lulus_luas = luas >= 60
    status = "LOLOS" if (lulus_w and lulus_h and lulus_luas) else "DIBUANG"
    catat(f"  x={x:4d} y={y:4d} w={w:3d} h={h:2d} luas={luas:4d} -> {status}"
          f"  (w>=25:{lulus_w} h>=5:{lulus_h} luas>=60:{lulus_luas})")

# Simpan potongan visual area tersebut
potong = kecil[y_atas:y_bawah, 850:1050]
simpan(BASE / "output" / "diagnosis_sisa" / "baris_jumlah1.png", potong)
catat("\nvisual disimpan: baris_jumlah1.png")

Path("hasil_komponen.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("selesai")