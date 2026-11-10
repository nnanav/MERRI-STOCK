"""
Membuat contact sheet untuk banyak kelas sekaligus,
agar pemeriksaan label bisa dilakukan cepat.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from alat_label_manual import buat_contact_sheet, kumpulkan_berkas

KELUAR = BASE / "output" / "label_manual"


def main():
    KELUAR.mkdir(parents=True, exist_ok=True)

    target = []

    # Simbol
    for nama in ("koma", "titik", "slash", "strip", "persen",
                 "petik", "kurung_buka", "kurung_tutup"):
        target.append(("simbol", nama))

    # Angka
    for d in "0123456789":
        target.append(("angka", d))

    # Huruf kecil yang sering muncul di nota
    for h in ("p", "c", "s", "r", "i", "t", "e", "a", "n", "o"):
        target.append(("huruf/lowercase", h))

    # Huruf besar yang sering muncul
    for h in "KMBPC":
        target.append(("huruf/uppercase", h))

    ringkas = []

    for jenis, nama in target:
        folder = BASE / "dataset" / Path(jenis) / nama
        if not folder.exists():
            continue

        berkas = kumpulkan_berkas(folder)
        if not berkas:
            continue

        keluar = KELUAR / f"sheet_{jenis.replace('/', '_')}_{nama}.png"
        hasil, daftar = buat_contact_sheet(folder, keluar)

        asli = sum(1 for p in daftar if p.name.startswith("asli_"))
        aug = sum(1 for p in daftar if p.name.startswith("aug_"))
        nota = sum(1 for p in daftar if p.name.startswith("nota_"))

        ringkas.append((jenis, nama, len(daftar), asli, aug, nota))

    baris = ["jenis              kelas      total  asli  aug  nota"]
    baris.append("-" * 58)
    for jenis, nama, total, asli, aug, nota in ringkas:
        baris.append(
            f"{jenis:18s} {nama:9s} {total:6d} {asli:5d} {aug:4d} {nota:5d}"
        )

    Path("rekap_sheet.txt").write_text("\n".join(baris), encoding="utf-8")
    print("\n".join(baris))
    print(f"\nSemua sheet disimpan di: {KELUAR}")


if __name__ == "__main__":
    main()