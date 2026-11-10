
from pathlib import Path
import cv2
import numpy as np
import random

# =========================================
# KONFIGURASI
# =========================================

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / "dataset" / "original"
AUGMENTED = BASE / "dataset" / "augmented"

TARGET = 50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

EKSTENSI = {".jpg", ".jpeg", ".png"}


# =========================================
# BACA GAMBAR
# =========================================

def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)

    gambar = cv2.imdecode(
        data,
        cv2.IMREAD_GRAYSCALE
    )

    return gambar


# =========================================
# NORMALISASI KANVAS
# =========================================

def siapkan_kanvas(gambar):
    """
    Menyiapkan gambar karakter pada
    kanvas putih 64x64 secara proporsional.
    """

    if gambar is None or gambar.size == 0:
        return None

    tinggi, lebar = gambar.shape[:2]

    if tinggi < 2 or lebar < 2:
        return None

    ukuran = 64
    ruang = 46

    skala = min(
        ruang / lebar,
        ruang / tinggi
    )

    w_baru = max(1, round(lebar * skala))
    h_baru = max(1, round(tinggi * skala))

    resize = cv2.resize(
        gambar,
        (w_baru, h_baru),
        interpolation=cv2.INTER_CUBIC
    )

    kanvas = np.full(
        (ukuran, ukuran),
        255,
        dtype=np.uint8
    )

    x = (ukuran - w_baru) // 2
    y = (ukuran - h_baru) // 2

    kanvas[
        y:y + h_baru,
        x:x + w_baru
    ] = resize

    return kanvas


# =========================================
# AUGMENTASI
# =========================================

def augmentasi(gambar):
    """
    Variasi ringan agar bentuk asli
    karakter tetap dipertahankan.
    """

    h, w = gambar.shape

    # 1. Rotasi kecil
    sudut = random.uniform(-3, 3)

    # 2. Translasi kecil
    geser_x = random.uniform(-2, 2)
    geser_y = random.uniform(-2, 2)

    matriks = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        sudut,
        1.0
    )

    matriks[0, 2] += geser_x
    matriks[1, 2] += geser_y

    hasil = cv2.warpAffine(
        gambar,
        matriks,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    # 3. Kecerahan dan kontras
    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-12, 12)

    hasil = np.clip(
        hasil.astype(np.float32) * alpha + beta,
        0, 255
    ).astype(np.uint8)

    # 4. Sesekali blur ringan
    if random.random() < 0.25:
        hasil = cv2.GaussianBlur(
            hasil,
            (3, 3),
            random.uniform(0.1, 0.5)
        )

    # 5. Sesekali noise ringan
    if random.random() < 0.35:
        noise = np.random.normal(
            0,
            random.uniform(1, 4),
            hasil.shape
        )

        hasil = np.clip(
            hasil.astype(np.float32) + noise,
            0, 255
        ).astype(np.uint8)

    return hasil


# =========================================
# PROSES SEMUA KELAS
# =========================================

def proses_dataset():

    if not ORIGINAL.exists():
        print("Folder original tidak ditemukan.")
        return

    total_dibuat = 0
    kelas_kosong = []

    for folder in sorted(ORIGINAL.rglob("*")):

        if not folder.is_dir():
            continue

        # Hanya folder yang berisi gambar langsung
        daftar_file = [
            p for p in sorted(folder.iterdir())
            if p.is_file()
            and p.suffix.lower() in EKSTENSI
        ]

        if not daftar_file:
            continue

        relatif = folder.relative_to(ORIGINAL)
        tujuan = AUGMENTED / relatif

        gambar_asli = []

        for path in daftar_file:
            gambar = baca_gambar(path)
            kanvas = siapkan_kanvas(gambar)

            if kanvas is not None:
                gambar_asli.append(
                    (path, kanvas)
                )

        if not gambar_asli:
            kelas_kosong.append(str(relatif))
            continue

        tujuan.mkdir(
            parents=True,
            exist_ok=True
        )

        sudah_aug = len([
            p for p in tujuan.iterdir()
            if p.is_file()
            and p.suffix.lower() in EKSTENSI
        ])

        jumlah_asli = len(gambar_asli)

        kurang = max(
            0,
            TARGET - jumlah_asli - sudah_aug
        )

        print(
            f"\n{relatif}: "
            f"{jumlah_asli} asli, "
            f"{sudah_aug} augmentasi, "
            f"perlu {kurang}"
        )

        dibuat = 0
        nomor = 1

        while dibuat < kurang:

            sumber, gambar = gambar_asli[
                dibuat % jumlah_asli
            ]

            hasil = augmentasi(gambar)

            nama = (
                f"aug_{sumber.stem}_"
                f"{nomor:04d}.png"
            )

            target = tujuan / nama

            nomor += 1

            if target.exists():
                continue

            berhasil, buffer = cv2.imencode(
                ".png",
                hasil
            )

            if not berhasil:
                continue

            buffer.tofile(str(target))

            dibuat += 1
            total_dibuat += 1

        print(
            f"TOTAL: {jumlah_asli + sudah_aug + dibuat}/50"
        )

    print("\n============================")
    print("AUGMENTASI SELESAI")
    print("Gambar baru:", total_dibuat)
    print("============================")

    if kelas_kosong:
        print("Kelas gagal dibaca:", kelas_kosong)


if __name__ == "__main__":
    proses_dataset()
