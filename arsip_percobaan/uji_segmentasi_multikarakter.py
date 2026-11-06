
from pathlib import Path
import sys
import csv

import cv2
import numpy as np

from processing.dataset_loader import (
    baca_gambar,
    normalisasi_karakter
)

from processing.pengenalan_karakter import (
    muat_cnn,
    muat_templates
)


# ====================================
# KONFIGURASI
# ====================================

SKALA = 3
MAX_BATAS = 24
MAKS_LEBAR_RASIO = 1.25
MIN_LEBAR_RASIO = 0.17


def simpan(path, gambar):
    ok, data = cv2.imencode(".png", gambar)
    if ok:
        data.tofile(str(path))


def binarisasi(gambar):
    besar = cv2.resize(
        gambar, None,
        fx=SKALA, fy=SKALA,
        interpolation=cv2.INTER_CUBIC
    )

    besar = cv2.GaussianBlur(
        besar, (3, 3), 0
    )

    _, biner = cv2.threshold(
        besar, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return besar, biner


def cari_komponen(biner):
    jumlah, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            biner, connectivity=8
        )
    )

    hasil = []

    for i in range(1, jumlah):
        x, y, w, h, area = stats[i]

        if area < 10 or w < 2 or h < 5:
            continue

        hasil.append((x, y, w, h))

    return sorted(hasil, key=lambda k: k[0])


# ====================================
# KANDIDAT BATAS
# ====================================

def cari_batas(bagian):
    h, w = bagian.shape

    if w < 10:
        return [0, w]

    proyeksi = np.count_nonzero(
        bagian, axis=0
    ).astype(np.float32)

    halus = np.convolve(
        proyeksi,
        np.ones(3) / 3,
        mode="same"
    )

    margin = max(3, int(h * 0.12))
    kandidat = []

    # Lembah tinta
    for x in range(margin, w - margin):
        if (
            halus[x] <= halus[x - 1]
            and halus[x] <= halus[x + 1]
        ):
            kandidat.append(x)

    # Titik tambahan agar kelompok
    # tanpa celah tetap bisa dicoba.
    langkah = max(5, round(h * 0.48))

    for x in range(langkah, w, langkah):
        kiri = max(1, x - langkah // 3)
        kanan = min(w - 1, x + langkah // 3)

        if kanan > kiri:
            titik = kiri + int(
                np.argmin(halus[kiri:kanan])
            )
            kandidat.append(titik)

    # Prioritaskan titik yang tintanya sedikit
    kandidat = sorted(
        set(kandidat),
        key=lambda x: halus[x]
    )

    terpilih = []

    jarak_min = max(3, round(h * 0.12))

    for x in kandidat:
        if all(
            abs(x - p) >= jarak_min
            for p in terpilih
        ):
            terpilih.append(x)

        if len(terpilih) >= MAX_BATAS:
            break

    return [0] + sorted(terpilih) + [w]


# ====================================
# NORMALISASI POTONGAN
# ====================================

def siapkan_potongan(bagian, kiri, kanan):
    potongan = bagian[:, kiri:kanan]

    if potongan.size == 0:
        return None

    titik = cv2.findNonZero(potongan)

    if titik is None:
        return None

    x, y, w, h = cv2.boundingRect(titik)

    if w < 2 or h < 4:
        return None

    potongan = potongan[
        y:y+h, x:x+w
    ]

    # Normalisasi membutuhkan
    # karakter hitam, latar putih.
    putih = cv2.bitwise_not(potongan)

    return normalisasi_karakter(putih)


# ====================================
# NILAI SEMUA CALON POTONGAN
# ====================================

def nilai_potongan(
    bagian, batas,
    model, labels,
    templates, template_labels
):
    h, w = bagian.shape

    min_lebar = max(
        3, round(h * MIN_LEBAR_RASIO)
    )
    max_lebar = max(
        min_lebar + 1,
        round(h * MAKS_LEBAR_RASIO)
    )

    pasangan = []
    gambar_batch = []

    for i, kiri in enumerate(batas[:-1]):
        for j in range(i + 1, len(batas)):
            kanan = batas[j]
            lebar = kanan - kiri

            if lebar < min_lebar:
                continue

            if lebar > max_lebar:
                break

            normal = siapkan_potongan(
                bagian, kiri, kanan
            )

            if normal is None:
                continue

            pasangan.append((i, j))
            gambar_batch.append(normal)

    if not gambar_batch:
        return {}

    X = np.array(
        gambar_batch, dtype=np.float32
    )[..., np.newaxis] / 255.0

    # Prediksi CNN sekaligus
    probs = model.predict(
        X, batch_size=64, verbose=0
    )

    # Template Matching memakai
    # beberapa template per kelas.
    vectors = X[..., 0].reshape(len(X), -1)

    norma = np.linalg.norm(
        vectors, axis=1, keepdims=True
    )

    vectors = vectors / np.maximum(
        norma, 1e-8
    )

    skor_tm = vectors @ templates.T

    hasil = {}

    lebar_acuan = max(3, h * 0.55)

    for idx, (i, j) in enumerate(pasangan):
        ic = int(np.argmax(probs[idx]))
        it = int(np.argmax(skor_tm[idx]))

        label_cnn = labels[ic]
        label_tm = template_labels[it]

        pc = float(probs[idx, ic])
        pt = float(skor_tm[idx, it])

        panjang = batas[j] - batas[i]

        # Penalti bentuk geometris:
        # terlalu sempit / lebar kurang disukai.
        penalti_lebar = 0.28 * abs(
            np.log(max(1, panjang) / lebar_acuan)
        )

        # Skor eksplorasi, bukan probabilitas
        # kebenaran segmentasi.
        skor = (
            0.65 * np.log(max(pc, 1e-6))
            + 0.35 * np.log(max(pt, 1e-6))
            - penalti_lebar
            - 0.25
        )

        if label_cnn == label_tm:
            skor += 0.10

        hasil[(i, j)] = {
            "skor": skor,
            "cnn": label_cnn,
            "tm": label_tm,
            "confidence_cnn": pc,
            "similarity_tm": pt
        }

    return hasil


# ====================================
# DYNAMIC PROGRAMMING
# ====================================

def pilih_kombinasi(batas, nilai):
    n = len(batas)

    dp = [-float("inf")] * n
    jejak = [None] * n

    dp[0] = 0.0

    for j in range(1, n):
        for i in range(j):
            info = nilai.get((i, j))

            if info is None:
                continue

            skor = dp[i] + info["skor"]

            if skor > dp[j]:
                dp[j] = skor
                jejak[j] = i

    if jejak[-1] is None:
        return []

    hasil = []
    sekarang = n - 1

    while sekarang > 0:
        sebelumnya = jejak[sekarang]

        if sebelumnya is None:
            return []

        hasil.append((
            batas[sebelumnya],
            batas[sekarang],
            nilai[(sebelumnya, sekarang)]
        ))

        sekarang = sebelumnya

    return list(reversed(hasil))


# ====================================
# PROSES UTAMA
# ====================================

def proses(path_gambar):
    path = Path(path_gambar)

    gambar = baca_gambar(path)

    if gambar is None:
        raise ValueError(
            f"Gagal membaca gambar: {path}"
        )

    output = path.parent / (
        "multikarakter_" + path.stem
    )

    output.mkdir(
        parents=True, exist_ok=True
    )

    print("Memuat model CNN dan templates...")

    model, labels = muat_cnn()
    semua_templates, semua_label_tm = muat_templates()

    # Batasi template per kelas agar
    # percobaan lebih ringan.
    indeks = []
    hitungan = {}

    for i, label in enumerate(semua_label_tm):
        jumlah = hitungan.get(label, 0)

        if jumlah < 5:
            indeks.append(i)
            hitungan[label] = jumlah + 1

    templates = semua_templates[indeks]
    labels_tm = [
        semua_label_tm[i] for i in indeks
    ]

    besar, biner = binarisasi(gambar)
    komponen = cari_komponen(biner)

    visual = cv2.cvtColor(
        besar, cv2.COLOR_GRAY2BGR
    )

    laporan = []
    urutan_global = 0

    for nomor, (x, y, w, h) in enumerate(
        komponen, start=1
    ):
        bagian = biner[y:y+h, x:x+w]

        # Komponen kecil tetap satu bagian.
        if w <= h * 1.15:
            batas = [0, w]
        else:
            batas = cari_batas(bagian)

        nilai = nilai_potongan(
            bagian, batas,
            model, labels,
            templates, labels_tm
        )

        hasil = pilih_kombinasi(
            batas, nilai
        )

        # Jika tidak ada kombinasi valid,
        # simpan sebagai komponen belum terurai.
        if not hasil:
            cv2.rectangle(
                visual,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                1
            )
            continue

        for kiri, kanan, info in hasil:
            urutan_global += 1

            x1 = x + kiri
            x2 = x + kanan

            cv2.rectangle(
                visual,
                (x1, y),
                (x2, y+h),
                (0, 170, 0),
                1
            )

            laporan.append({
                "komponen": nomor,
                "urutan": urutan_global,
                "x1": x1,
                "y1": y,
                "x2": x2,
                "y2": y+h,
                "cnn": info["cnn"],
                "tm": info["tm"],
                "skor_cnn": round(
                    info["confidence_cnn"], 4
                ),
                "skor_tm": round(
                    info["similarity_tm"], 4
                )
            })

    simpan(output / "biner.png", biner)
    simpan(
        output / "hasil_multikarakter.png",
        visual
    )

    csv_path = output / "hasil_prediksi.csv"

    with open(
        csv_path, "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        kolom = [
            "komponen", "urutan",
            "x1", "y1", "x2", "y2",
            "cnn", "tm",
            "skor_cnn", "skor_tm"
        ]

        writer = csv.DictWriter(
            f, fieldnames=kolom
        )

        writer.writeheader()
        writer.writerows(laporan)

    print("\n=== SEGMENTASI MULTIKARAKTER ===")
    print("Gambar:", path.name)
    print("Komponen awal:", len(komponen))
    print("Calon potongan terpilih:", len(laporan))
    print("Output:", output)

    print(
        "\nKotak hijau = kandidat segmentasi "
        "yang dipilih algoritma."
    )
    print(
        "Belum berarti semua kotak "
        "berisi satu karakter yang benar."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Gunakan: python uji_segmentasi_multikarakter.py '
            '"path_gambar"'
        )
        sys.exit(1)

    proses(sys.argv[1])
