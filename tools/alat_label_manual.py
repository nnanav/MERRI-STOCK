"""
Membuat contact sheet (lembar pemeriksa) untuk SATU kelas.

Semua jenis file dipakai: asli_*, aug_*, nota_* (tidak hanya nota_*).

Pemakaian:
    python alat_label_manual.py koma
    python alat_label_manual.py titik
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

PER_BARIS = 12
UKURAN = 64
MARGIN = 6
JUDUL = 22

EKSTENSI = {".png", ".jpg", ".jpeg"}


def kumpulkan_berkas(folder):
    """
    Mengambil SEMUA gambar pada folder, apa pun awalan namanya
    (asli_, aug_, nota_, atau nama lain).
    """
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EKSTENSI
    )


def ambil_fokus(folder):
    """
    Mengambil hanya file yang perlu diperiksa manual:
    asli_ dan nota_ (aug_ dilewati karena hanya variasi dari asli_).
    """
    return [
        p for p in kumpulkan_berkas(folder)
        if not p.name.startswith("aug_")
    ]


def buat_contact_sheet(folder, keluaran):
    berkas = kumpulkan_berkas(folder)
    if not berkas:
        return None, []

    n = len(berkas)
    baris = (n + PER_BARIS - 1) // PER_BARIS

    sel = UKURAN + MARGIN
    lebar = PER_BARIS * sel + MARGIN
    tinggi = baris * (sel + JUDUL) + MARGIN

    kanvas = np.full((tinggi, lebar, 3), 255, dtype=np.uint8)

    for i, path in enumerate(berkas):
        r = i // PER_BARIS
        c = i % PER_BARIS
        x0 = MARGIN + c * sel
        y0 = MARGIN + r * (sel + JUDUL)

        cv2.rectangle(
            kanvas, (x0, y0), (x0 + UKURAN, y0 + UKURAN), (200, 200, 200), 1
        )

        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if img is not None and img.size:
            h, w = img.shape[:2]
            skala = min(UKURAN / max(1, w), UKURAN / max(1, h))
            w2 = max(1, int(w * skala))
            h2 = max(1, int(h * skala))
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            px = x0 + (UKURAN - w2) // 2
            py = y0 + (UKURAN - h2) // 2
            kanvas[py:py + h2, px:px + w2] = img3

        cv2.putText(
            kanvas, str(i + 1), (x0 + 2, y0 + UKURAN + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1
        )

    cv2.imwrite(str(keluaran), kanvas)
    return keluaran, berkas


def pindahkan(kelas_asal, tujuan, daftar_nomor, hanya_fokus=True):
    """
    Memindahkan crop nomor tertentu dari satu kelas ke kelas lain.

    hanya_fokus=True  -> penomoran mengikuti sheet fokus (asli_ + nota_)
    hanya_fokus=False -> penomoran mengikuti semua file

    kelas_asal : nama folder sumber (mis. 'titik')
    tujuan     : nama folder tujuan (mis. 'koma')
    daftar_nomor : list nomor pada contact sheet, mis. [3, 7, 12]
    """
    folder_asal = BASE / "dataset" / "simbol" / kelas_asal
    folder_tujuan = BASE / "dataset" / "simbol" / tujuan

    if not folder_asal.exists():
        print(f"Folder asal tidak ada: {folder_asal}")
        return
    if not folder_tujuan.exists():
        print(f"Folder tujuan tidak ada: {folder_tujuan}")
        return

    if hanya_fokus:
        berkas = ambil_fokus(folder_asal)
    else:
        berkas = kumpulkan_berkas(folder_asal)

    dipindah = 0
    for nomor in daftar_nomor:
        idx = nomor - 1
        if idx < 0 or idx >= len(berkas):
            print(f"  nomor {nomor}: di luar rentang (1-{len(berkas)})")
            continue

        sumber = berkas[idx]
        target = folder_tujuan / sumber.name

        if target.exists():
            print(f"  nomor {nomor}: sudah ada di tujuan, dilewati")
            continue

        sumber.rename(target)
        dipindah += 1
        print(f"  nomor {nomor}: {sumber.name} -> {tujuan}/")

    print(f"\nSelesai. {dipindah} file dipindahkan ke '{tujuan}'.")


def main():
    if len(sys.argv) < 2:
        print("Gunakan salah satu:")
        print("  python alat_label_manual.py <kelas>")
        print("      -> membuat contact sheet, mis. python alat_label_manual.py titik")
        print()
        print("  python alat_label_manual.py <asal> <tujuan> <nomor...>")
        print("      -> memindahkan crop, mis. python alat_label_manual.py titik koma 3 7 12")
        return

    # Mode pindah: ada 3 argumen atau lebih
    if len(sys.argv) >= 4 and sys.argv[2].isalpha():
        asal = sys.argv[1]
        tujuan = sys.argv[2]
        try:
            nomor = [int(x) for x in sys.argv[3:]]
        except ValueError:
            print("Nomor harus berupa angka. Contoh: titik koma 3 7 12")
            return
        pindahkan(asal, tujuan, nomor)
        return

    # Mode buat contact sheet
    kelas = sys.argv[1]
    folder = BASE / "dataset" / "simbol" / kelas

    if not folder.exists():
        print(f"Folder tidak ada: {folder}")
        return

    keluar = BASE / "output" / "label_manual"
    keluar.mkdir(parents=True, exist_ok=True)

    keluaran, berkas = buat_contact_sheet(folder, keluar / f"sheet_{kelas}.png")

    if keluaran is None:
        print(f"Tidak ada file nota_* di {folder}")
        return

    print(f"Kelas          : {kelas}")
    print(f"Jumlah crop    : {len(berkas)}")
    print(f"Contact sheet  : {keluaran}")
    print()
    print("Cara pakai:")
    print("  1. Buka gambar contact sheet")
    print("  2. Perhatikan nomor di bawah tiap kotak")
    print(f"  3. Sebut nomor yang BUKAN {kelas}")
    print("     contoh: 'nomor 3 dan 7 sebenarnya koma'")
    print("  4. Jalankan perintah pindah, format:")
    print(f"     python alat_label_manual.py {kelas} <tujuan> <nomor...>")
    print(f"     contoh: python alat_label_manual.py {kelas} titik 3 7")


if __name__ == "__main__":
    main()