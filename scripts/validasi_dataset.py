
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import random

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset"
LAPORAN = BASE / "output" / "laporan_dataset"

EKSTENSI = {".png", ".jpg", ".jpeg"}
SEED = 42

random.seed(SEED)


def baca_gambar(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except (OSError, ValueError, cv2.error):
        return None


def semua_kelas():
    hasil = []

    for jenis, daftar in (
        ("uppercase", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("lowercase", "abcdefghijklmnopqrstuvwxyz")
    ):
        for label in daftar:
            folder = DATASET / "huruf" / jenis / label
            hasil.append((folder, label))

    for label in "0123456789":
        hasil.append((DATASET / "angka" / label, label))

    simbol = {
        "koma": ",",
        "titik": ".",
        "strip": "-",
        "slash": "/",
        "persen": "%",
        "petik": '"',
        "kurung_buka": "(",
        "kurung_tutup": ")"
    }

    for nama, label in simbol.items():
        hasil.append(
            (DATASET / "simbol" / nama, label)
        )

    return hasil


def mask_karakter(gambar):
    """
    Deteksi piksel karakter untuk pemeriksaan.
    Bukan fungsi normalisasi untuk pelatihan.
    """
    _, biner = cv2.threshold(
        gambar, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = gambar.shape

    tepi = np.concatenate([
        gambar[0, :],
        gambar[-1, :],
        gambar[:, 0],
        gambar[:, -1]
    ])

    # Background terang -> karakter gelap
    if np.median(tepi) >= 127:
        mask = cv2.bitwise_not(biner)
    else:
        mask = biner

    return mask


def periksa_gambar(path):
    gambar = baca_gambar(path)

    if gambar is None or gambar.size == 0:
        return "RUSAK", 0.0, None

    h, w = gambar.shape

    if h < 5 or w < 5:
        return "TERLALU_KECIL", 0.0, gambar

    mask = mask_karakter(gambar)

    rasio = cv2.countNonZero(mask) / mask.size

    if rasio < 0.005:
        return "CEK_KOSONG", rasio, gambar

    if rasio > 0.80:
        return "CEK_BACKGROUND", rasio, gambar

    return "OK", rasio, gambar


def buat_preview(kelompok, nama):
    """
    Menampilkan enam contoh acak dari setiap kelas.
    Setiap kelas menjadi satu baris.
    """
    lebar_sel = 105
    tinggi_sel = 110

    canvas = np.full(
        (
            max(1, len(kelompok)) * tinggi_sel,
            7 * lebar_sel,
            3
        ),
        255,
        dtype=np.uint8
    )

    for baris, (label, daftar_file) in enumerate(kelompok):
        y = baris * tinggi_sel

        cv2.putText(
            canvas,
            label,
            (12, y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        contoh = random.sample(
            daftar_file,
            min(6, len(daftar_file))
        )

        for kolom, path in enumerate(contoh, start=1):
            gambar = baca_gambar(path)

            if gambar is None:
                continue

            h, w = gambar.shape
            skala = min(70 / w, 70 / h)

            ukuran = (
                max(1, round(w * skala)),
                max(1, round(h * skala))
            )

            kecil = cv2.resize(
                gambar,
                ukuran,
                interpolation=cv2.INTER_AREA
                if skala < 1
                else cv2.INTER_NEAREST
            )

            x = kolom * lebar_sel + 8
            py = y + 12

            canvas[
                py:py + kecil.shape[0],
                x:x + kecil.shape[1]
            ] = cv2.cvtColor(
                kecil,
                cv2.COLOR_GRAY2BGR
            )

    target = LAPORAN / f"preview_{nama}.png"
    cv2.imwrite(str(target), canvas)
    print("Preview dibuat:", target.name)


def jalankan():
    LAPORAN.mkdir(parents=True, exist_ok=True)

    catatan = []
    ringkasan = Counter()
    kategori_preview = defaultdict(list)

    for folder, label in semua_kelas():
        if not folder.exists():
            print("FOLDER BELUM ADA:", folder)
            continue

        daftar = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in EKSTENSI
        )

        if "uppercase" in folder.parts:
            kelompok = "uppercase"
        elif "lowercase" in folder.parts:
            kelompok = "lowercase"
        elif "angka" in folder.parts:
            kelompok = "angka"
        else:
            kelompok = "simbol"

        if daftar:
            kategori_preview[kelompok].append(
                (label, daftar)
            )

        # Deteksi gambar identik dalam satu kelas
        hash_terlihat = set()

        for path in daftar:
            status, rasio, gambar = periksa_gambar(path)

            if status == "OK" and gambar is not None:
                sidik = hashlib.sha256(
                    gambar.tobytes()
                ).hexdigest()

                if sidik in hash_terlihat:
                    status = "DUPLIKAT_IDENTIK"
                else:
                    hash_terlihat.add(sidik)

            nama = path.name.lower()

            if nama.startswith("aug_"):
                sumber = "augmentasi"
            elif nama.startswith("asli_"):
                sumber = "asli"
            else:
                sumber = "belum_diketahui"

            catatan.append({
                "label": label,
                "kategori": kelompok,
                "file": str(path.relative_to(DATASET)),
                "sumber": sumber,
                "status": status,
                "rasio_tinta": round(rasio, 4)
            })

            ringkasan[status] += 1

    csv_path = LAPORAN / "hasil_validasi.csv"

    with open(
        csv_path, "w",
        newline="", encoding="utf-8-sig"
    ) as f:
        penulis = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "kategori",
                "file",
                "sumber",
                "status",
                "rasio_tinta"
            ]
        )

        penulis.writeheader()
        penulis.writerows(catatan)

    for kategori, kelompok in kategori_preview.items():
        buat_preview(kelompok, kategori)

    print("\n=== HASIL VALIDASI DATASET ===")
    print("Total gambar diperiksa:", len(catatan))

    for status, jumlah in sorted(ringkasan.items()):
        print(f"{status}: {jumlah}")

    print("\nLaporan CSV:", csv_path)
    print("Folder preview:", LAPORAN)

    print(
        "\nCATATAN: Status OK hanya berarti gambar "
        "lolos pemeriksaan teknis awal."
    )
    print(
        "Label dan bentuk karakter tetap perlu "
        "diperiksa dari gambar preview."
    )


if __name__ == "__main__":
    jalankan()
