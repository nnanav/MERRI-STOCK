"""
Membandingkan 3 strategi keputusan label pada field yang punya ground truth:

A. final = CNN           (kondisi sekarang, Opsi B')
B. final = TM            (jika TM yang menentukan)
C. final = CNN + TM menimpa bila CNN ragu (aturan lama, sebelum Opsi B')

Tujuan: mengukur berapa besar kerugian/keuntungan tiap strategi
terhadap CER, bukan menebak.
"""

import importlib.util
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
from processing.pengenalan_karakter import (
    muat_cnn, muat_templates, gabung_segmen,
)
from processing.segmentasi_karakter import segmentasi_karakter, susun_teks
from tests_gt import GT

_spes = importlib.util.spec_from_file_location(
    "peng_lama",
    str(BASE / "archive" / "pengenalan_karakter_sebelum_opsiB.py"),
)
mod_lama = importlib.util.module_from_spec(_spes)
_spes.loader.exec_module(mod_lama)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    total = 0
    jumlah = {"A": 0.0, "B": 0.0, "C": 0.0}
    menang = {"A": 0, "B": 0, "C": 0}

    for nama, path in NOTA:
        if not path.exists():
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        for f in det["fields"]:
            gt = GT.get((nama, f["id"]))
            if not gt:
                continue
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue

            img = app.ke_tigakanal(crop)
            _, hasil = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            if not hasil:
                continue
            ta = float(np.median([s["y2"] - s["y1"] for s in hasil]))

            # A. CNN murni
            final_a, _ = gabung_segmen(hasil)
            teks_a = susun_teks(final_a, "final", ta)

            # B. TM murni
            teks_b = susun_teks(hasil, "tm", ta)

            # C. aturan lama (TM boleh menimpa)
            final_c, _ = mod_lama.gabung_segmen(hasil)
            teks_c = susun_teks(final_c, "final", ta)

            ca = app.hitung_cer(gt, teks_a)
            cb = app.hitung_cer(gt, teks_b)
            cc = app.hitung_cer(gt, teks_c)

            total += 1
            jumlah["A"] += ca
            jumlah["B"] += cb
            jumlah["C"] += cc

            terbaik = min(ca, cb, cc)
            if abs(ca - terbaik) < 1e-9:
                menang["A"] += 1
            if abs(cb - terbaik) < 1e-9:
                menang["B"] += 1
            if abs(cc - terbaik) < 1e-9:
                menang["C"] += 1

            if cb < ca - 1e-9 or cc < ca - 1e-9:
                print(f"{nama} {f['label']}:")
                print(f"   GT   = {gt!r}")
                print(f"   A CNN= {teks_a!r}  CER {ca:.3f}")
                print(f"   B TM = {teks_b!r}  CER {cb:.3f}")
                print(f"   C mix= {teks_c!r}  CER {cc:.3f}")

    print(f"\nField diuji = {total}")
    for k in ("A", "B", "C"):
        print(f"  Strategi {k}: rata-rata CER = {jumlah[k] / max(1, total):.4f}"
              f" | jadi yang terbaik di {menang[k]} field")


if __name__ == "__main__":
    main()