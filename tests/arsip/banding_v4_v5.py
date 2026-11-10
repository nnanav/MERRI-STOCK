"""
Membandingkan dua model pada semua field ber-GT:
- v4 (CER 0,2454)
- v5 (CER 0,2579)

Tujuan: memilih model terbaik berdasarkan CER dan jumlah field benar.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "tests") not in sys.path:
    sys.path.insert(0, str(BASE / "tests"))

import numpy as np
import tensorflow as tf

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_templates, gabung_segmen
from processing.segmentasi_karakter import segmentasi_karakter, susun_teks
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

import json

MODEL_V4 = BASE / "archive" / "cnn_karakter_TERBAIK_cer0245.keras"
LABEL_V4 = BASE / "archive" / "labels_TERBAIK_cer0245.json"
MODEL_V5 = BASE / "models" / "cnn_karakter.keras"


def evaluasi(model, labels, templates, label_tm, det_semua):
    total = 0.0
    n = 0
    benar = 0
    rincian = {}

    for nama, fields in det_semua:
        for f in fields:
            gt = GT.get((nama, f["id"]))
            if not gt or f["crop_ocr"] is None:
                continue
            img = app.ke_tigakanal(f["crop_ocr"])
            _, hasil = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            if not hasil:
                continue
            ta = float(np.median([s["y2"] - s["y1"] for s in hasil]))
            final, _ = gabung_segmen(hasil)
            teks = susun_teks(final, "final", ta)
            cer = app.hitung_cer(gt, teks)
            total += cer
            n += 1
            benar += int(teks.strip() == gt.strip())
            rincian[(nama, f["id"])] = (teks, cer)

    return total / max(1, n), benar, n, rincian


def main():
    templates, label_tm = muat_templates()

    det_semua = []
    for nama, path in NOTA:
        if not path.exists():
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        det_semua.append((nama, det["fields"]))

    m4 = tf.keras.models.load_model(str(MODEL_V4), compile=False)
    l4 = json.loads(LABEL_V4.read_text(encoding="utf-8"))
    m5 = tf.keras.models.load_model(str(MODEL_V5), compile=False)
    l5 = json.loads((BASE / "models" / "labels.json").read_text(encoding="utf-8"))

    cer4, b4, n4, r4 = evaluasi(m4, l4, templates, label_tm, det_semua)
    cer5, b5, n5, r5 = evaluasi(m5, l5, templates, label_tm, det_semua)

    print(f"{'':10s} {'CER':>8s} {'akurasi':>9s} {'benar':>7s}")
    print(f"{'v4':10s} {cer4:8.4f} {(1-cer4)*100:8.2f}% {b4:4d}/{n4}")
    print(f"{'v5':10s} {cer5:8.4f} {(1-cer5)*100:8.2f}% {b5:4d}/{n5}")

    print("\nField yang BERBEDA antara v4 dan v5:")
    for kunci in r4:
        t4, c4 = r4[kunci]
        t5, c5 = r5.get(kunci, ("", 0))
        if t4 != t5:
            tanda = "v5 lebih baik" if c5 < c4 else "v4 lebih baik"
            print(f"  {kunci[0]} {kunci[1]}")
            print(f"     v4: {t4!r} (CER {c4:.3f})")
            print(f"     v5: {t5!r} (CER {c5:.3f})  -> {tanda}")


if __name__ == "__main__":
    main()