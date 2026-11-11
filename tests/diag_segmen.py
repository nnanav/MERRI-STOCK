"""Diagnostik segmentasi: menampilkan lebar/tinggi tiap segmen agar
terlihat mana yang tergabung (lebar > 1 karakter) atau terpotong."""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "tests") not in sys.path:
    sys.path.insert(0, str(BASE / "tests"))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import gabung_segmen, muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_utama
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

model, labels = muat_cnn()
templates, label_tm = muat_templates()

lebar = []


def catat(teks):
    lebar.append(teks)


for nama, path in NOTA:
    if not path.exists():
        continue
    det = app.jalankan_deteksi(baca_gambar(path))
    catat("=" * 100)
    catat(nama)
    catat("=" * 100)

    for f in det["fields"]:
        gt = GT.get((nama, f["id"]))
        if not gt:
            continue
        crop = f["crop_ocr"]
        if crop is None or not getattr(crop, "size", 0):
            catat(f"\n{f['label']:18s} GT={gt!r}  CROP KOSONG")
            continue

        img = app.ke_tigakanal(crop)
        _, hasil, metode = segmentasi_utama(
            img, model, labels, templates, label_tm
        )
        if not hasil:
            catat(f"\n{f['label']:18s} GT={gt!r}  TIDAK ADA SEGMEN")
            continue

        ta = float(np.median([s["y2"] - s["y1"] for s in hasil]))
        teks = app.kenali_crop(
            crop, pakai_clahe=False, jenis=f.get("jenis")
        )["teks"]

        catat(f"\n{f['label']:18s} GT={gt!r}")
        catat(f"{'':18s} pred={teks!r}  segmen={len(hasil)} "
              f"panjangGT={len(gt)} metode={metode}")

        for i, s in enumerate(hasil, 1):
            w = s["x2"] - s["x1"]
            h = s["y2"] - s["y1"]
            catat(f"   {i:02d} x={s['x1']:4d}-{s['x2']:4d} "
                  f"w={w:3d} h={h:3d} w/ta={w / ta:4.2f} "
                  f"cnn={s.get('cnn', '')!r:5s} pc={s.get('pc', 0):.2f} "
                  f"{s.get('status', '')}")

Path(BASE / "output" / "hasil_diag_segmen.txt").write_text(
    "\n".join(lebar), encoding="utf-8"
)
print("selesai")
