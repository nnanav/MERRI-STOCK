
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


# ==========================================
# KONFIGURASI
# ==========================================

SKALA = 3
MAKS_KANDIDAT = 12

# Hanya komponen yang cukup lebar
# dibandingkan tinggi yang diperiksa.
RASIO_LEBAR = 1.15


def simpan(path, gambar):
    berhasil, data = cv2.imencode(".png", gambar)

    if berhasil:
        data.tofile(str(path))


def binarisasi(gambar):
    besar = cv2.resize(
        gambar,
        None,
        fx=SKALA,
        fy=SKALA,
        interpolation=cv2.INTER_CUBIC
    )

    abu = cv2.GaussianBlur(
        besar, (3, 3), 0
    )

    _, biner = cv2.threshold(
        abu,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return besar, biner


def cari_komponen(biner):
    jumlah, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            biner,
            connectivity=8
        )
    )

    hasil = []

    for i in range(1, jumlah):
        x, y, w, h, luas = stats[i]

        if luas < 10 or w < 2 or h < 5:
            continue

        hasil.append((x, y, w, h))

    return sorted(hasil, key=lambda k: k[0])


def cari_titik_potong(bagian):
    """
    Cari kandidat batas berdasarkan
    lembah proyeksi vertikal.
    """

    h, w = bagian.shape

    if w < 12:
        return []

    proyeksi = np.count_nonzero(
        bagian,
        axis=0
    ).astype(np.float32)

    proyeksi = np.convolve(
        proyeksi,
        np.ones(3) / 3,
        mode="same"
    )

    margin = max(4, int(w * 0.12))

    if margin * 2 >= w:
        return []

    calon = []

    for posisi in range(margin, w - margin):
        kiri = proyeksi[posisi - 1]
        tengah = proyeksi[posisi]
        kanan = proyeksi[posisi + 1]

        if tengah <= kiri and tengah <= kanan:
            calon.append((tengah, posisi))

    # Ambil titik dengan tinta paling sedikit
    calon.sort(key=lambda item: item[0])

    terpilih = []

    for _, posisi in calon:
        if all(abs(posisi - p) >= 5 for p in terpilih):
            terpilih.append(posisi)

        if len(terpilih) >= MAKS_KANDIDAT:
            break

    return sorted(terpilih)


def normalisasi_potongan(biner, x1, x2):
    """
    Potongan masih berupa karakter putih
    di atas latar hitam.

    Dibalik agar sesuai input normalisasi
    dataset_loader.py.
    """

    potongan = biner[:, x1:x2]

    if potongan.size == 0:
        return None

    titik = cv2.findNonZero(potongan)

    if titik is None:
        return None

    x, y, w, h = cv2.boundingRect(titik)

    if w < 2 or h < 4:
        return None

    potongan = potongan[y:y+h, x:x+w]

    # Latar kembali putih, karakter hitam
    gambar = cv2.bitwise_not(potongan)

    return normalisasi_karakter(gambar)


def nilai_kandidat(biner, x1, x2,
                   model, labels,
                   templates, template_labels):

    normal = normalisasi_potongan(
        biner, x1, x2
    )

    if normal is None:
        return None

    X = normal.astype(np.float32)
    X = X.reshape(1, 32, 32, 1) / 255.0

    prediksi = model.predict(
        X,
        verbose=0
    )[0]

    idx_cnn = int(np.argmax(prediksi))

    vektor = normal.astype(np.float32).flatten()
    norma = np.linalg.norm(vektor)

    if norma <= 0:
        return None

    vektor /= norma

    kemiripan = templates @ vektor
    idx_tm = int(np.argmax(kemiripan))

    skor_cnn = float(prediksi[idx_cnn])
    skor_tm = float(kemiripan[idx_tm])

    label_cnn = labels[idx_cnn]
    label_tm = template_labels[idx_tm]

    # Skor eksperimen untuk mengurutkan
    # calon potongan, BUKAN ukuran akurasi.
    skor = 0.5 * skor_cnn + 0.5 * skor_tm

    return {
        "cnn": label_cnn,
        "tm": label_tm,
        "skor_cnn": skor_cnn,
        "skor_tm": skor_tm,
        "skor": skor
    }


def proses(path_gambar):
    path = Path(path_gambar)

    gambar = baca_gambar(path)

    if gambar is None:
        raise ValueError(
            f"Gambar tidak bisa dibaca: {path}"
        )

    output = path.parent / (
        "analisis_cnn_tm_" + path.stem
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Memuat CNN dan Template Matching...")

    model, labels = muat_cnn()
    templates, template_labels = muat_templates()

    besar, biner = binarisasi(gambar)

    komponen = cari_komponen(biner)

    visual = cv2.cvtColor(
        besar, cv2.COLOR_GRAY2BGR
    )

    laporan = []

    for nomor, (x, y, w, h) in enumerate(
        komponen,
        start=1
    ):

        cv2.rectangle(
            visual,
            (x, y),
            (x+w, y+h),
            (0, 0, 255),
            1
        )

        # Jangan paksa pemisahan
        # pada komponen yang sempit.
        if w < h * RASIO_LEBAR:
            continue

        bagian = biner[y:y+h, x:x+w]

        titik = cari_titik_potong(bagian)

        hasil_kandidat = []

        for posisi in titik:

            kiri = nilai_kandidat(
                bagian,
                0,
                posisi,
                model,
                labels,
                templates,
                template_labels
            )

            kanan = nilai_kandidat(
                bagian,
                posisi,
                w,
                model,
                labels,
                templates,
                template_labels
            )

            if kiri is None or kanan is None:
                continue

            # Kedua sisi perlu dinilai bersama.
            skor = (
                kiri["skor"] + kanan["skor"]
            ) / 2

            # Penalti kecil jika garis potong
            # melewati terlalu banyak tinta.
            tinta_potong = np.count_nonzero(
                bagian[:, max(0, posisi-1):posisi+2]
            )

            penalti = (
                tinta_potong /
                max(1, 3 * h)
            ) * 0.15

            skor -= penalti

            hasil_kandidat.append({
                "posisi": posisi,
                "skor": skor,
                "kiri": kiri,
                "kanan": kanan
            })

        if not hasil_kandidat:
            continue

        terbaik = max(
            hasil_kandidat,
            key=lambda item: item["skor"]
        )

        potong_x = x + terbaik["posisi"]

        # Garis biru = kandidat terbaik.
        # Ini BELUM batas final.
        cv2.line(
            visual,
            (potong_x, y),
            (potong_x, y+h),
            (255, 0, 0),
            2
        )

        laporan.append({
            "komponen": nomor,
            "x": x,
            "y": y,
            "lebar": w,
            "tinggi": h,
            "calon_potong_x": potong_x,
            "cnn_kiri": terbaik["kiri"]["cnn"],
            "cnn_kanan": terbaik["kanan"]["cnn"],
            "tm_kiri": terbaik["kiri"]["tm"],
            "tm_kanan": terbaik["kanan"]["tm"],
            "skor": round(terbaik["skor"], 4)
        })

    simpan(output / "biner.png", biner)
    simpan(
        output / "kandidat_cnn_tm.png",
        visual
    )

    csv_path = output / "hasil_kandidat.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        kolom = [
            "komponen",
            "x",
            "y",
            "lebar",
            "tinggi",
            "calon_potong_x",
            "cnn_kiri",
            "cnn_kanan",
            "tm_kiri",
            "tm_kanan",
            "skor"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=kolom
        )

        writer.writeheader()
        writer.writerows(laporan)

    print("\n=== ANALISIS CNN + TM ===")
    print("Gambar:", path.name)
    print("Komponen tinta:", len(komponen))
    print("Komponen dengan kandidat:", len(laporan))
    print("Hasil:", output)

    print(
        "\nGaris biru masih kandidat, "
        "belum hasil segmentasi final."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Gunakan: python uji_segmentasi_cnn_tm.py '
            '"path_gambar"'
        )
        sys.exit(1)

    proses(sys.argv[1])
