"""
Menguji beberapa aturan override TM terhadap CNN untuk melihat
seberapa besar manfaatnya terhadap CER, dan pada ambang berapa.
"""

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
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter, susun_teks
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

# Aturan: (nama, syarat_cnn_ragu, ambang_pt)
ATURAN = [
    ("A. CNN murni", None, None),
    ("B. TM bila pc<0.60 & pt>=0.80", 0.60, 0.80),
    ("C. TM bila pc<0.90 & pt>=0.80", 0.90, 0.80),
    ("D. TM bila pt>=0.85 (abaikan pc)", None, 0.85),
    ("E. TM bila pt>=0.92 (abaikan pc)", None, 0.92),
]


def terapkan(hasil, batas_pc, ambang_pt):
    if batas_pc is None and ambang_pt is None:
        return [dict(s, final=s.get("cnn", "")) for s in hasil]

    keluaran = []
    for s in hasil:
        cnn = s.get("cnn", "")
        tm = s.get("tm", "")
        pc = float(s.get("pc", 0.0) or 0.0)
        pt = float(s.get("pt", 0.0) or 0.0)

        final = cnn
        if tm and tm != cnn and pt >= ambang_pt:
            if batas_pc is None or pc < batas_pc:
                final = tm

        keluaran.append(dict(s, final=final))
    return keluaran


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    data = []
    for nama, path in NOTA:
        if not path.exists():
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        for f in det["fields"]:
            gt = GT.get((nama, f["id"]))
            crop = f["crop_ocr"]
            if not gt or crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)
            _, hasil = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            if not hasil:
                continue
            ta = float(np.median([s["y2"] - s["y1"] for s in hasil]))
            data.append((nama, f["label"], gt, hasil, ta))

    print(f"Field ber-GT: {len(data)}\n")
    for nama_aturan, batas_pc, ambang_pt in ATURAN:
        total_cer = 0.0
        benar = 0
        for nama, label, gt, hasil, ta in data:
            final = terapkan(hasil, batas_pc, ambang_pt)
            teks = susun_teks(final, "final", ta)
            cer = app.hitung_cer(gt, teks)
            total_cer += cer
            benar += int(teks.strip() == gt.strip())
        rata = total_cer / max(1, len(data))
        print(f"{nama_aturan:34s} CER={rata:.4f}  akurasi={(1-rata)*100:5.2f}%  "
              f"benar={benar}/{len(data)}")


if __name__ == "__main__":
    main()