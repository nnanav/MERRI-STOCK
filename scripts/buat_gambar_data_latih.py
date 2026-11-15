"""
Membuat gambar "Data Latih Karakter" dari dataset ASLI program ini.

Setiap kotak menampilkan satu contoh gambar karakter hasil normalisasi
yang dipakai saat melatih CNN (32x32, tinta putih di atas latar hitam),
beserta nomor kelas dan kodenya.

Kode label mengikuti gaya gambar laporan:
    angka   -> "7"
    besar   -> "A.U"  (U = uppercase)
    kecil   -> "a.L"  (L = lowercase)
    simbol  -> ", S"  (S = symbol)

Jalankan:
    python scripts/buat_gambar_data_latih.py
    python scripts/buat_gambar_data_latih.py --kolom 10 --keluar output/gambar.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from processing.dataset_loader import (  # noqa: E402
    DATASET_DIR,
    EKSTENSI,
    LABEL_SIMBOL,
    baca_gambar,
    normalisasi_karakter,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # backend tanpa jendela, aman di server
import matplotlib.pyplot as plt  # noqa: E402


# =========================================
# PENGUMPULAN KELAS
# =========================================

# Urutan tampil mengikuti gambar laporan: angka, huruf besar, huruf kecil,
# lalu simbol. Hanya folder yang benar-benar ada di dataset yang dipakai.
KELOMPOK = [
    ("angka", "D", "0123456789"),
    ("huruf/uppercase", "U", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ("huruf/lowercase", "L", "abcdefghijklmnopqrstuvwxyz"),
    ("simbol", "S", None),
]


def daftar_kelas():
    """
    Kumpulkan (folder, nama_kelas, karakter, kode_jenis) tiap kelas.

    Hanya kelas yang foldernya ada yang dikembalikan.
    """
    hasil = []

    for sub, kode_jenis, urutan in KELOMPOK:
        akar = DATASET_DIR / sub
        if not akar.exists():
            continue

        if urutan is not None:
            # Angka/huruf: urutan karakter sudah pasti.
            for karakter in urutan:
                folder = akar / karakter
                if folder.is_dir():
                    hasil.append((folder, karakter, karakter, kode_jenis))
        else:
            # Simbol: nama folder = nama simbol; labelnya dibaca dari
            # LABEL_SIMBOL agar konsisten dengan pipeline pelatihan.
            for nama_folder in sorted(LABEL_SIMBOL):
                folder = akar / nama_folder
                if folder.is_dir():
                    simbol = LABEL_SIMBOL[nama_folder]
                    hasil.append((folder, nama_folder, simbol, kode_jenis))

    return hasil


def kode_tampilan(karakter, kode_jenis):
    """Bentuk teks kode kelas, mis. 'A.U', 'a.L', ', S', atau '7'."""
    if kode_jenis == "D":
        return karakter
    if kode_jenis == "S":
        return f"{karakter} S"
    return f"{karakter}.{kode_jenis}"


def pilih_contoh(folder):
    """
    Ambil satu contoh dari folder kelas.

    Prioritas file asli (berawalan 'asli_') agar yang tampil adalah data
    asli, bukan hasil augmentasi.
    """
    berkas = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in EKSTENSI
    ]
    if not berkas:
        return None, 0

    asli = [p for p in berkas if p.name.lower().startswith("asli_")]
    pilihan = (asli or berkas)[0]
    return pilihan, len(berkas)


def muat_contoh(folder):
    """Muat satu contoh ternormalisasi 32x32 (seperti data latih CNN)."""
    path, jumlah = pilih_contoh(folder)
    if path is None:
        return None, 0, None
    gambar = baca_gambar(path)
    if gambar is None:
        return None, jumlah, path.name
    normal = normalisasi_karakter(gambar)
    return normal, jumlah, path.name


# =========================================
# PEMBUATAN GAMBAR
# =========================================

def buat_gambar(kolom=8, keluar=None, judul="B. Data Latih Karakter",
                gaya="terang"):
    """
    Susun grid contoh karakter menjadi satu berkas PNG.

    gaya:
      "terang" -> karakter gelap di latar putih (seperti gambar laporan)
      "latih"  -> persis input CNN: tinta putih di latar hitam 32x32
    """
    if keluar is None:
        akar = "data_latih_karakter" if gaya == "terang" else "data_latih_latih"
        keluar = BASE / "output" / f"{akar}.png"
    keluar = Path(keluar)
    keluar.parent.mkdir(parents=True, exist_ok=True)

    kelas = daftar_kelas()
    if not kelas:
        raise RuntimeError(
            f"Tidak ada folder kelas ditemukan di {DATASET_DIR}"
        )

    # Muat semua contoh lebih dulu agar ukuran grid bisa dihitung pasti.
    contoh = []
    total_gambar = 0
    for folder, nama_kelas, karakter, kode_jenis in kelas:
        normal, jumlah, nama_file = muat_contoh(folder)
        total_gambar += jumlah
        if normal is None:
            print(f"  lewati (gagal dibaca): {folder}")
            continue
        contoh.append({
            "kelas": nama_kelas,
            "kode": kode_tampilan(karakter, kode_jenis),
            "gambar": normal,
            "jumlah": jumlah,
            "berkas": nama_file,
        })

    n = len(contoh)
    if n == 0:
        raise RuntimeError("Tidak ada contoh yang berhasil dimuat.")
    baris = int(np.ceil(n / kolom))

    # Ukuran tiap sel disamakan agar kotak terlihat rapi.
    lebar_sel, tinggi_sel = 1.55, 1.72
    fig, axes = plt.subplots(
        baris, kolom,
        figsize=(kolom * lebar_sel, baris * tinggi_sel),
        squeeze=False,
    )

    tinggi = lebar = 32
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue

        data = contoh[i]
        img = data["gambar"]
        tinggi, lebar = img.shape[:2]

        # Gaya tampilan: "terang" membalik warna agar karakter gelap di
        # latar putih seperti gambar laporan. Nilai piksel tetap berasal
        # dari data asli, hanya warnanya dibalik untuk keperluan cetak.
        if gaya == "terang":
            img = 255 - img
            latar, tinta = "white", "black"
        else:
            latar, tinta = "black", "white"

        ax.set_facecolor(latar)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_xticks([0, lebar - 1])
        ax.set_yticks([0, tinggi - 1])
        ax.tick_params(axis="both", labelsize=5, length=2, pad=1,
                       colors=tinta)

        # Nomor kelas dan kode di bawah kotak, seperti gambar laporan.
        ax.set_xlabel(f"{i}\n{data['kode']}", fontsize=7.5, labelpad=1.5)

        for sisi in ax.spines.values():
            sisi.set_linewidth(0.6)
            sisi.set_color("#888888")

    fig.suptitle(judul, x=0.012, y=0.995, ha="left", va="top",
                 fontsize=15, fontweight="bold")

    fig.text(
        0.012, 0.003,
        f"Sumber: {DATASET_DIR.name}/ (dataset asli program)  |  "
        f"{n} kelas  |  {total_gambar} gambar  |  "
        f"tiap kotak = 1 contoh hasil normalisasi {tinggi}x{lebar}",
        ha="left", va="bottom", fontsize=7.5, color="#444444",
    )

    fig.tight_layout(rect=(0, 0.012, 1, 0.975))
    fig.savefig(keluar, dpi=170, facecolor="white")
    plt.close(fig)

    print(f"Kelas ditampilkan : {n}")
    print(f"Total gambar      : {total_gambar}")
    print(f"Grid              : {baris} baris x {kolom} kolom")
    print(f"Disimpan ke       : {keluar}")
    return keluar


def main():
    parser = argparse.ArgumentParser(
        description="Buat gambar data latih karakter dari dataset asli."
    )
    parser.add_argument("--kolom", type=int, default=8,
                        help="jumlah kolom grid (default 8)")
    parser.add_argument("--keluar", type=str, default=None,
                        help="path berkas PNG keluaran")
    parser.add_argument("--judul", type=str,
                        default="B. Data Latih Karakter",
                        help="judul gambar")
    parser.add_argument("--gaya", type=str, default="terang",
                        choices=("terang", "latih"),
                        help="terang = karakter gelap di latar putih; "
                             "latih = persis input CNN (putih di hitam)")
    args = parser.parse_args()

    buat_gambar(kolom=args.kolom, keluar=args.keluar, judul=args.judul,
                gaya=args.gaya)


if __name__ == "__main__":
    main()
