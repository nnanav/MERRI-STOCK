"""
Membuat contact sheet FOKUS: hanya file yang perlu diperiksa manual.

File augmentasi (aug_) dilewati karena hanya variasi dari gambar asli
yang sudah ada. Yang ditampilkan: asli_ dan nota_.

Pemakaian:
    python alat_sheet_fokus.py koma
    python alat_sheet_fokus.py titik
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

PER_BARIS = 10
UKURAN = 84
MARGIN = 8
JUDUL = 26

EKSTENSI = {".png", ".jpg", ".jpeg"}


def ambil_fokus(folder):
    """Ambil asli_ dan nota_ saja (lewati aug_)."""
    hasil = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EKSTENSI:
            continue
        if p.name.startswith("aug_"):
            continue
        hasil.append(p)
    return hasil


def buat_sheet(folder, keluaran):
    berkas = ambil_fokus(folder)
    if not berkas:
        return None, []

    n = len(berkas)
    baris = (n + PER_BARIS - 1) // PER_BARIS
    sel = UKURAN + MARGIN
    lebar = PER_BARIS * sel + MARGIN
    tinggi = baris * (sel + JUDUL) + MARGIN

    kanvas = np.full((tinggi, lebar, 3), 255, dtype=np.uint8)

    for i, path in enumerate(berkas):
        r, c = i // PER_BARIS, i % PER_BARIS
        x0 = MARGIN + c * sel
        y0 = MARGIN + r * (sel + JUDUL)

        cv2.rectangle(
            kanvas, (x0, y0), (x0 + UKURAN, y0 + UKURAN), (190, 190, 190), 1
        )

        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if img is not None and img.size:
            h, w = img.shape[:2]
            skala = min(UKURAN / max(1, w), UKURAN / max(1, h))
            w2, h2 = max(1, int(w * skala)), max(1, int(h * skala))
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            px = x0 + (UKURAN - w2) // 2
            py = y0 + (UKURAN - h2) // 2
            kanvas[py:py + h2, px:px + w2] = img3

        warna = (180, 0, 0) if path.name.startswith("asli_") else (0, 130, 0)
        cv2.putText(
            kanvas, str(i + 1), (x0 + 2, y0 + UKURAN + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 1
        )

    cv2.imwrite(str(keluaran), kanvas)
    return keluaran, berkas


def main():
    if len(sys.argv) < 2:
        print("Gunakan: python alat_sheet_fokus.py <kelas>")
        print("Contoh : python alat_sheet_fokus.py koma")
        return

    kelas = sys.argv[1]
    folder = BASE / "dataset" / "simbol" / kelas

    if not folder.exists():
        print(f"Folder tidak ada: {folder}")
        return

    keluar = BASE / "output" / "label_manual"
    keluar.mkdir(parents=True, exist_ok=True)

    keluaran, berkas = buat_sheet(folder, keluar / f"fokus_{kelas}.png")
    if keluaran is None:
        print(f"Tidak ada file asli_/nota_ di {folder}")
        return

    asli = sum(1 for p in berkas if p.name.startswith("asli_"))
    nota = sum(1 for p in berkas if p.name.startswith("nota_"))

    print(f"Kelas         : {kelas}")
    print(f"File diperiksa: {len(berkas)}  (asli={asli}, nota={nota})")
    print(f"Sheet         : {keluaran}")
    print()
    print("Nomor biru = file asli, nomor hijau = crop dari nota.")
    print("Sebutkan nomor yang BUKAN koma/titik, misal: '3, 7, 12'")


if __name__ == "__main__":
    main()