"""
Mengukur CER hasil OCR pada field yang punya ground truth.
Dipakai untuk membandingkan model lama vs model baru.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "tests") not in sys.path:
    sys.path.insert(0, str(BASE / "tests"))

import app
from processing.pemrosesan_field import baca_gambar
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def main():
    total = 0
    total_cer = 0.0
    benar = 0
    rincian = []

    for nama, path in NOTA:
        if not path.exists():
            print(f"{nama}: file tidak ada")
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        for f in det["fields"]:
            gt = GT.get((nama, f["id"]))
            if not gt:
                continue
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                pred = ""
            else:
                pred = app.kenali_crop(crop, pakai_clahe=False)["teks"]

            cer = app.hitung_cer(gt, pred)
            total += 1
            total_cer += cer
            benar += int(pred.strip() == gt.strip())
            rincian.append((nama, f["label"], gt, pred, cer))

    print(f"{'nota':10s} {'field':16s} {'GT':18s} {'prediksi':18s} CER")
    for nama, label, gt, pred, cer in rincian:
        print(f"{nama:10s} {label:16s} {gt!r:18s} {pred!r:18s} {cer:.3f}")

    print(f"\nJumlah field = {total}")
    print(f"Benar penuh = {benar}")
    print(f"Rata-rata CER = {total_cer / max(1, total):.4f}")
    print(f"Akurasi karakter (1-CER) = {(1 - total_cer / max(1, total)) * 100:.2f}%")


if __name__ == "__main__":
    main()