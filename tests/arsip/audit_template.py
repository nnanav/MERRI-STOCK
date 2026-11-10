"""
Mengukur kualitas template Template Matching pada proyek ini.

1. Komposisi template (asli vs augmentasi).
2. Konsistensi intra-kelas vs antar-kelas.
3. Kemiripan template terhadap karakter nyata dari nota.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.dataset_loader import (
    daftar_folder_kelas, normalisasi_karakter,
)
from processing.pemrosesan_field import baca_gambar as baca_nota
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter, SKALA

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def bagian1_komposisi():
    print("=" * 72)
    print("1. KOMPOSISI TEMPLATE")
    print("=" * 72)

    total = Counter()
    per_kelas = defaultdict(Counter)

    for folder, label in daftar_folder_kelas():
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            if p.name.startswith("asli_"):
                total["asli"] += 1
                per_kelas[label]["asli"] += 1
            elif p.name.startswith("aug_"):
                total["augmentasi"] += 1
                per_kelas[label]["augmentasi"] += 1
            else:
                total["lain"] += 1

    jumlah = sum(total.values())
    print(f"  Total template   : {jumlah}")
    for k in ("asli", "augmentasi", "lain"):
        if total[k]:
            print(f"  {k:14s}: {total[k]:5d} "
                  f"({100.0 * total[k] / jumlah:.1f}%)")

    print("\n  Contoh per kelas (asli / augmentasi):")
    for c in ["0", "5", ",", "P", "c", "K"]:
        if c in per_kelas:
            print(f"    {c!r:4s} -> {per_kelas[c]['asli']:3d} asli / "
                  f"{per_kelas[c]['augmentasi']:3d} augmentasi")


def bagian2_konsistensi(templates, label_tm):
    print()
    print("=" * 72)
    print("2. KONSISTENSI TEMPLATE (intra-kelas vs antar-kelas)")
    print("=" * 72)

    T = np.array(templates, dtype=np.float32)
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-8)
    label_tm = np.array(label_tm)

    mirip = T @ T.T
    np.fill_diagonal(mirip, -1.0)

    sama = mirip.copy()
    beda = mirip.copy()
    for i in range(len(label_tm)):
        sama[i, label_tm != label_tm[i]] = np.nan
        beda[i, label_tm == label_tm[i]] = np.nan

    r_sama = np.nanmean(sama, axis=1)
    r_beda = np.nanmax(beda, axis=1)

    m_sama = float(np.nanmean(r_sama))
    m_beda = float(np.nanmean(r_beda))
    print(f"  Kemiripan ke kelas SENDIRI (rata-rata): {m_sama:.3f}")
    print(f"  Kemiripan ke kelas LAIN    (maks)     : {m_beda:.3f}")
    margin = m_sama - m_beda
    print(f"  Margin pemisahan                      : {margin:.3f}")
    if margin < 0.15:
        print("  -> Margin kecil: template antar-kelas saling mirip,")
        print("     TM mudah tertukar.")
    else:
        print("  -> Margin cukup: template masih bisa dibedakan.")

    skor = []
    for i in range(len(label_tm)):
        j = int(np.nanargmax(beda[i]))
        skor.append((r_sama[i] - beda[i, j], label_tm[i],
                     label_tm[j], beda[i, j]))
    skor.sort()
    print("\n  Kelas paling rawan tertukar:")
    for s, a, b, v in skor[:6]:
        print(f"    {a!r:4s} mirip dengan {b!r:4s} skor {v:.3f}")


def bagian3_terhadap_nota(templates, label_tm, model, labels):
    print()
    print("=" * 72)
    print("3. KEMIRIPAN TEMPLATE TERHADAP KARAKTER NOTA ASLI")
    print("=" * 72)

    T = np.array(templates, dtype=np.float32)
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-8)
    daftar_label = list(label_tm)

    semua = []

    for nama, path in NOTA:
        if not path.exists():
            continue
        det = app.jalankan_deteksi(baca_nota(path))
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)
            _, segmen = segmentasi_karakter(
                img, model, labels, T, daftar_label
            )
            if not segmen:
                continue

            besar = cv2.resize(img, None, fx=SKALA, fy=SKALA,
                               interpolation=cv2.INTER_CUBIC)
            for s in segmen:
                y1 = max(0, min(besar.shape[0] - 1, int(s["y1"])))
                y2 = max(y1 + 1, min(besar.shape[0], int(s["y2"])))
                x1 = max(0, min(besar.shape[1] - 1, int(s["x1"])))
                x2 = max(x1 + 1, min(besar.shape[1], int(s["x2"])))
                potong = besar[y1:y2, x1:x2]
                if potong.size == 0 or potong.shape[0] < 3 or potong.shape[1] < 2:
                    continue
                if potong.ndim == 3:
                    potong = cv2.cvtColor(potong, cv2.COLOR_BGR2GRAY)
                normal = normalisasi_karakter(potong)
                if normal is None:
                    continue

                v = normal.astype(np.float32).flatten()
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                semua.append(float((T @ (v / n)).max()))

    if semua:
        print(f"  Karakter nota diuji          : {len(semua)}")
        print(f"  Skor cocok tertinggi (rata2) : {np.mean(semua):.3f}")
        print(f"  Skor 10% terendah            : {np.percentile(semua, 10):.3f}")
        rendah = sum(1 for z in semua if z < 0.80)
        print(f"  Karakter dengan skor < 0.80  : {rendah} "
              f"({100.0 * rendah / len(semua):.1f}%)")


def main():
    bagian1_komposisi()

    templates, label_tm = muat_templates()
    bagian2_konsistensi(templates, label_tm)

    model, labels = muat_cnn()
    bagian3_terhadap_nota(templates, label_tm, model, labels)


if __name__ == "__main__":
    main()

