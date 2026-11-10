import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import baca_gambar

det = app.jalankan_deteksi(baca_gambar(BASE / "data_uji" / "nota_26.jpeg"))
f = next(x for x in det["fields"] if x["id"] == "proyek")
crop = f["crop_show"]
print("crop shape:", None if crop is None else crop.shape)

skala = 2
abu = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
up = cv2.resize(abu, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
_, biner = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
H, W = biner.shape
print("upsampled H,W:", H, W)

n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
ambang_luas = max(4, int(0.0006 * H * W))
for i in range(1, n):
    if stats[i, 4] < ambang_luas:
        x, y, w, h = stats[i, :4]
        biner[y:y + h, x:x + w] = 0

ys = np.where(np.count_nonzero(biner, axis=1) > 0)[0]
tinggi = int(ys.max() - ys.min() + 1)
print("tinggi:", tinggi, "ambang_luas:", ambang_luas)

lebar_komp = []
for i in range(1, n):
    w_komp, h_komp, a_komp = stats[i, 2], stats[i, 3], stats[i, 4]
    if h_komp >= 0.55 * tinggi and a_komp >= ambang_luas and w_komp <= 1.3 * tinggi:
        lebar_komp.append(int(w_komp))
print("lebar_komp sample:", sorted(lebar_komp)[:20], "n=", len(lebar_komp))
char_w = float(np.median(lebar_komp)) if len(lebar_komp) >= 3 else 0.5 * tinggi
print("char_w:", char_w, "1.15*char_w:", 1.15 * char_w)

proj = np.count_nonzero(biner, axis=0)
ambang_ink = max(1, int(0.02 * tinggi))
runs = []
mulai = None
for x in range(W):
    if proj[x] > ambang_ink:
        if mulai is None:
            mulai = x
    else:
        if mulai is not None:
            runs.append((mulai, x - 1)); mulai = None
if mulai is not None:
    runs.append((mulai, W - 1))

print("jumlah runs:", len(runs))
for a, b in runs:
    lebar = b - a + 1
    n_sel = max(1, int(np.ceil(lebar / (1.15 * char_w))))
    print(f"  run {a:4d}-{b:4d} lebar={lebar:3d} n_sel={n_sel}")