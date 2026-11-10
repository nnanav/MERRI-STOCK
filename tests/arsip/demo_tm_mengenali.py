"""
Membuktikan dua hal:
1. Template Matching memang MENGENALI karakter (bukan cuma menemukan posisi)
   -> dengan template berlabel, TM mengembalikan label 'A' dst.
2. Seberapa sering TM sepakat dengan CNN pada karakter nota,
   dan seberapa besar pengaruh kualitas template.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.dataset_loader import baca_gambar, normalisasi_karakter
from processing.pemrosesan_field import baca_gambar as baca_nota
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def bagian1_tm_mengenali(templates, label_tm):
    print("=" * 70)
    print("BAGIAN 1: TM mengenali karakter berdasarkan template berlabel")
    print("=" * 70)

    contoh = [
        ("dataset/huruf/uppercase/A", "A"),
        ("dataset/angka/0", "0"),
        ("dataset/angka/5", "5"),
        ("dataset/simbol/koma", ","),
    ]

    for folder, harap in contoh:
        path_folder = BASE / folder
        if not path_folder.exists():
            continue
        berkas = sorted(p for p in path_folder.iterdir() if p.is_file())
        if not berkas:
            continue

        gambar = baca_gambar(berkas[0])
        normal = normalisasi_karakter(gambar)
        if normal is None:
            continue

        vektor = normal.astype(np.float32).flatten()
        norma = np.linalg.norm(vektor)
        if norma == 0:
            continue
        skor = templates @ (vektor / norma)

        urut = np.argsort(skor)[::-1][:3]
        teks = ", ".join(
            f"{label_tm[i]!r}={skor[i]:.3f}" for i in urut
        )
        print(f"  gambar {folder.split('/')[-1]:6s} (harus {harap!r}) "
              f"-> TM top-3: {teks}")


def bagian2_kesepakatan(model, labels, templates, label_tm):
    print()
    print("=" * 70)
    print("BAGIAN 2: pada karakter nota, TM vs CNN")
    print("=" * 70)

    total = 0
    sepakat = 0
    pt_tm = []
    pc_cnn = []

    for nama, path in NOTA:
        if not path.exists():
            continue
        det = app.jalankan_deteksi(baca_nota(path))
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)
            _, hasil = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            for s in hasil:
                total += 1
                if s.get("cnn") == s.get("tm"):
                    sepakat += 1
                pt_tm.append(float(s.get("pt", 0)))
                pc_cnn.append(float(s.get("pc", 0)))

    if total:
        print(f"  Total karakter nota     : {total}")
        print(f"  TM sepakat dengan CNN   : {sepakat} "
              f"({100.0 * sepakat / total:.1f}%)")
        print(f"  Rata-rata keyakinan TM  : {np.mean(pt_tm):.3f}")
        print(f"  Rata-rata keyakinan CNN : {np.mean(pc_cnn):.3f}")


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    print(f"Template dimuat: {len(templates)} dari {len(set(label_tm))} kelas\n")

    bagian1_tm_mengenali(templates, label_tm)
    bagian2_kesepakatan(model, labels, templates, label_tm)


if __name__ == "__main__":
    main()