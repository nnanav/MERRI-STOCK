from pathlib import Path
import csv
import sys

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "hasil_deteksi_field"
LEBAR_PROSES = 1400


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan_gambar(path, gambar):
    sukses, data = cv2.imencode(".png", gambar)
    if sukses:
        data.tofile(str(path))


def baca_kandidat(csv_path):

    hasil = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for baris in reader:
            x1 = int(baris["x1"])
            y1 = int(baris["y1"])
            x2 = int(baris["x2"])
            y2 = int(baris["y2"])

            if x2 <= x1 or y2 <= y1:
                continue

            hasil.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "w": x2 - x1,
                "h": y2 - y1
            })

    return hasil

def samakan_ukuran(gambar):
    h, w = gambar.shape[:2]
    skala = LEBAR_PROSES / w

    return cv2.resize(
        gambar,
        (LEBAR_PROSES, max(1, round(h * skala))),
        interpolation=(
            cv2.INTER_AREA if skala < 1
            else cv2.INTER_CUBIC
        )
    )

def kelompokkan_sejajar(kandidat):

    if not kandidat:
        return []

    kandidat = sorted(kandidat, key=lambda k: k["cy"])

    tinggi_median = np.median([k["h"] for k in kandidat])
    toleransi = max(5, min(9, tinggi_median * 0.4))

    kelompok = []

    for kotak in kandidat:
        cocok = None
        jarak_terkecil = float("inf")

        for grup in kelompok:
            tengah = np.median([k["cy"] for k in grup])
            jarak = abs(kotak["cy"] - tengah)

            if jarak <= toleransi and jarak < jarak_terkecil:
                cocok = grup
                jarak_terkecil = jarak

        if cocok is None:
            kelompok.append([kotak])
        else:
            cocok.append(kotak)

    return kelompok

def deteksi_produk(kandidat, lebar, tinggi):

    # Kandidat kolom jumlah
    calon_jumlah = [
        k for k in kandidat
        if 0.57 * lebar <= k["x1"] < 0.70 * lebar
        and k["w"] >= 35
        and 0.17 * tinggi <= k["cy"] <= 0.60 * tinggi
    ]

    # Kandidat nama produk di kiri jumlah
    calon_nama = [
        k for k in kandidat
        if 0.26 * lebar <= k["x1"] < 0.56 * lebar
        and k["w"] >= 75
        and 0.17 * tinggi <= k["cy"] <= 0.60 * tinggi
    ]

    # Kandidat harga di kanan jumlah
    calon_harga = [
        k for k in kandidat
        if k["x1"] >= 0.69 * lebar
        and k["w"] >= 30
        and 0.17 * tinggi <= k["cy"] <= 0.60 * tinggi
    ]

    hasil = []
    nama_terpakai = set()

    # Proses jumlah dari atas ke bawah
    calon_jumlah.sort(key=lambda k: k["cy"])

    for jumlah in calon_jumlah:

        # Harus ada kolom harga di sebelah kanan
        harga_valid = [
            h for h in calon_harga
            if h is not jumlah
            and h["x1"] >= jumlah["x2"] + 8
            and abs(h["cy"] - jumlah["cy"]) <= 22
        ]

        if not harga_valid:
            continue

        # Cari nama produk di kiri jumlah
        nama_valid = []

        for nama in calon_nama:
            identitas = (
                nama["x1"], nama["y1"],
                nama["x2"], nama["y2"]
            )

            if identitas in nama_terpakai:
                continue

            if nama["x2"] >= jumlah["x1"] - 8:
                continue

            selisih_y = abs(
                nama["cy"] - jumlah["cy"]
            )

            if selisih_y > 24:
                continue

            nama_valid.append(nama)

        if not nama_valid:
            continue

        # Pilih nama dengan posisi Y terdekat.
        # Jarak horizontal digunakan sebagai pembeda jika posisi Y hampir sama.
        nama = min(
            nama_valid,
            key=lambda n: (
                abs(n["cy"] - jumlah["cy"]),
                jumlah["x1"] - n["x2"]
            )
        )

        identitas = (
            nama["x1"], nama["y1"],
            nama["x2"], nama["y2"]
        )

        nama_terpakai.add(identitas)
        hasil.append(nama)

    return sorted(
        hasil,
        key=lambda k: k["cy"]
    )


# DETEKSI FIELD TAMBAHAN


def deteksi_field_lain(kandidat, lebar, tinggi, produk):
    """
    Deteksi field tambahan.

    Nomor nota dan tanggal menggunakan
    area pencarian khusus agar tidak
    salah memilih baris lain.

    Deteksi nama produk tetap menggunakan
    fungsi deteksi_produk() yang sudah ada.
    """

    def buat_kotak(x1, y1, x2, y2):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(lebar, int(x2))
        y2 = min(tinggi, int(y2))

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "w": x2 - x1,
            "h": y2 - y1
        }

    # --------------------------------------
    # 1. AREA NOMOR NOTA
    # --------------------------------------
    # Mencakup tulisan NO dan angka di
    # kanan atas, bukan baris "Lembar ke".

    nomor_nota = buat_kotak(
        0.78 * lebar,
        0.055 * tinggi,
        0.95 * lebar,
        0.115 * tinggi
    )

    # --------------------------------------
    # 2. AREA TANGGAL
    # --------------------------------------
    # Mencakup wilayah tanggal di atas
    # tanda tangan, bukan nama direktur.

    tanggal = buat_kotak(
        0.77 * lebar,
        0.68 * tinggi,
        0.97 * lebar,
        0.76 * tinggi
    )

    # --------------------------------------
    # 3. JUMLAH DAN SATUAN
    # --------------------------------------

    kandidat_jumlah = [
        k for k in kandidat
        if (
            0.55 * lebar <= k["x1"] < 0.80 * lebar
            and k["w"] >= 25
        )
    ]

    jumlah_terpakai = set()
    rincian_produk = []

    for nomor, nama in enumerate(produk, start=1):
        calon = []

        for k in kandidat_jumlah:
            identitas = (
                k["x1"], k["y1"],
                k["x2"], k["y2"]
            )

            if identitas in jumlah_terpakai:
                continue

            if k["x1"] <= nama["x2"]:
                continue

            selisih_y = abs(
                k["cy"] - nama["cy"]
            )

            toleransi = max(
                16,
                0.8 * max(nama["h"], k["h"])
            )

            if selisih_y <= toleransi:
                calon.append(k)

        kotak_jumlah = None

        if calon:
            kotak_jumlah = min(
                calon,
                key=lambda k: (
                    abs(k["cy"] - nama["cy"]),
                    k["x1"]
                )
            )

            identitas = (
                kotak_jumlah["x1"],
                kotak_jumlah["y1"],
                kotak_jumlah["x2"],
                kotak_jumlah["y2"]
            )

            jumlah_terpakai.add(identitas)

        rincian_produk.append({
            "produk": nomor,
            "nama_produk": nama,
            "jumlah_satuan": kotak_jumlah
        })

    return {
        "nomor_nota": nomor_nota,
        "tanggal": tanggal,
        "rincian_produk": rincian_produk
    }

def crop_field(gambar, kotak, margin=4):

    if kotak is None:
        return None

    H, W = gambar.shape[:2]

    x1 = max(0, int(kotak["x1"]) - margin)
    y1 = max(0, int(kotak["y1"]) - margin)

    x2 = min(W, int(kotak["x2"]) + margin)
    y2 = min(H, int(kotak["y2"]) + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    return gambar[y1:y2, x1:x2].copy()

def proses(gambar_path, csv_path):
    gambar = baca_gambar(gambar_path)

    if gambar is None:
        raise ValueError("Gambar nota tidak dapat dibaca.")

    gambar = samakan_ukuran(gambar)
    H, W = gambar.shape[:2]

    kandidat = baca_kandidat(csv_path)
    produk = deteksi_produk(kandidat, W, H)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    visual = gambar.copy()

    # Semua kandidat berwarna abu-abu
    for k in kandidat:
        cv2.rectangle(
            visual,
            (k["x1"], k["y1"]),
            (k["x2"], k["y2"]),
            (170, 170, 170),
            1
        )

    data_produk = []

    for nomor, p in enumerate(produk, start=1):
        # Tambah margin kecil agar huruf tidak terpotong
        x1 = max(0, p["x1"] - 4)
        y1 = max(0, p["y1"] - 4)
        x2 = min(W, p["x2"] + 4)
        y2 = min(H, p["y2"] + 4)

        crop = gambar[y1:y2, x1:x2]

        simpan_gambar(
            OUTPUT / f"nama_produk_{nomor}.png",
            crop
        )

        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            (0, 180, 0),
            2
        )

        cv2.putText(
            visual,
            f"Produk {nomor}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        data_produk.append({
            "produk": nomor,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    simpan_gambar(
        OUTPUT / "deteksi_produk_otomatis.png",
        visual
    )

    with open(
        OUTPUT / "koordinat_produk.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["produk", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(data_produk)

    print("\n=== DETEKSI FIELD OTOMATIS ===")
    print("Jumlah kandidat teks:", len(kandidat))
    print("Calon nama produk:", len(produk))
    print("Folder hasil:", OUTPUT)

    if not produk:
        print(
            "PERINGATAN: Tidak ditemukan baris "
            "yang memenuhi pola kolom produk."
        )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            'Penggunaan: python -m processing.deteksi_field '
            '"path_nota" "path_csv"'
        )
        sys.exit(1)

    proses(Path(sys.argv[1]), Path(sys.argv[2]))
