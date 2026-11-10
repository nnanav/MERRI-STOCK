"""
Diagnosis kualitas segmentasi pada semua field ber-GT.

Mendeteksi tiga jenis masalah:
1. Bergabung (lebar segmen > 1.15 x tinggi teks)
2. Terpotong/noise (lebar segmen < 0.30 x tinggi teks)
3. Karakter pengganggu (':', '-', '.', ' ' palsu di awal/akhir)
"""

import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

LAP = []
def c(t=""):
    LAP.append(str(t))

model, labels = muat_cnn()
templates, label_tm = muat_templates()

t_bergabung = 0
t_terpotong = 0
t_pengganggu = 0
t_total = 0

for nama, path in NOTA:
    det = app.jalankan_deteksi(baca_gambar(path))
    c(f"\n{'='*70}")
    c(f"{nama}")
    c(f"{'='*70}")

    for f in det["fields"]:
        crop = f["crop_ocr"]
        if crop is None or not getattr(crop, "size", 0):
            continue

        img = app.ke_tigakanal(crop)
        _, seg = segmentasi_karakter(img, model, labels, templates, label_tm)
        if not seg:
            continue

        ta = float(np.median([s["y2"] - s["y1"] for s in seg]))

        bergabung = []
        terpotong = []
        pengganggu = []

        for i, s in enumerate(seg):
            w = s["x2"] - s["x1"]
            rasio = w / max(1.0, ta)
            ch = s.get("cnn", "")

            if rasio > 1.15:
                bergabung.append((i + 1, round(rasio, 2), ch))
            if rasio < 0.30:
                terpotong.append((i + 1, round(rasio, 2), ch))
            if i == 0 or i == len(seg) - 1:
                if ch in (":", "-", ".", " ", "'", "`", "="):
                    pengganggu.append((i + 1, ch))

        t_total += 1
        if bergabung:
            t_bergabung += 1
        if terpotong:
            t_terpotong += 1
        if pengganggu:
            t_pengganggu += 1

        tanda = []
        if bergabung:
            tanda.append("BERGABUNG")
        if terpotong:
            tanda.append("TERPOTONG")
        if pengganggu:
            tanda.append("PENGGANGGU")

        status = " / ".join(tanda) if tanda else "OK"
        c(f"\n{f['label']:16s} n={len(seg):3d} tinggi={ta:.0f}  [{status}]")

        if bergabung:
            c(f"   bergabung : {bergabung}")
        if terpotong:
            c(f"   terpotong : {terpotong}")
        if pengganggu:
            c(f"   pengganggu: {pengganggu}")

        teks = "".join(s.get("cnn", "") for s in seg)
        c(f"   teks      : {teks!r}")

c(f"\n\n{'='*70}")
c("RINGKASAN")
c(f"{'='*70}")
c(f"Field diperiksa      : {t_total}")
c(f"Ada segmen bergabung : {t_bergabung}")
c(f"Ada segmen terpotong : {t_terpotong}")
c(f"Ada karakter pengganggu: {t_pengganggu}")

Path("hasil_seg_diag.txt").write_text("\n".join(LAP), encoding="utf-8")
print("selesai")