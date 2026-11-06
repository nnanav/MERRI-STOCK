
from pathlib import Path
import sys
import csv

import cv2
import numpy as np

from processing.dataset_loader import normalisasi_karakter
from processing.pengenalan_karakter import (
    muat_cnn,
    muat_templates
)


SKALA = 3


# =========================================
# BACA DAN SIMPAN GAMBAR
# =========================================

def baca_gambar(path):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan_gambar(path, gambar):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, data = cv2.imencode(".png", gambar)

    if not ok:
        raise ValueError(f"Gagal encoding: {path}")

    with open(path, "wb") as f:
        f.write(data.tobytes())


# =========================================
# PREPROCESSING
# =========================================

def binarisasi(gambar):
    abu = cv2.cvtColor(
        gambar, cv2.COLOR_BGR2GRAY
    )

    abu = cv2.resize(
        abu,
        None,
        fx=SKALA,
        fy=SKALA,
        interpolation=cv2.INTER_CUBIC
    )

    abu = cv2.GaussianBlur(
        abu, (3, 3), 0
    )

    _, biner = cv2.threshold(
        abu, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return biner


# =========================================
# DETEKSI KOMPONEN
# =========================================

def deteksi_komponen(biner):
    n, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            biner,
            connectivity=8
        )
    )

    hasil = []

    for x, y, w, h, area in stats[1:]:
        if area >= 5 and h >= 3 and w >= 2:
            hasil.append((
                int(x), int(y),
                int(w), int(h)
            ))

    return sorted(
        hasil,
        key=lambda z: (z[0], z[1])
    )


# =========================================
# NORMALISASI KANDIDAT KARAKTER
# =========================================

def normalisasi_potongan(biner):
    titik = cv2.findNonZero(biner)

    if titik is None:
        return None

    x, y, w, h = cv2.boundingRect(titik)

    if w < 2 or h < 3:
        return None

    putih = cv2.bitwise_not(
        biner[y:y+h, x:x+w]
    )

    return normalisasi_karakter(putih)


# =========================================
# MENCARI BEBERAPA CALON BATAS
# =========================================

def calon_batas(blob, tinggi_acuan):
    h, w = blob.shape

    proyeksi = np.count_nonzero(
        blob, axis=0
    ).astype(np.float32)

    halus = np.convolve(
        proyeksi,
        np.ones(3) / 3,
        mode="same"
    )

    langkah = max(
        3,
        round(tinggi_acuan * 0.14)
    )

    batas = {0, w}

    # Kandidat batas di sekitar
    # bagian tinta yang lebih tipis.
    for p in range(langkah, w, langkah):
        a = max(1, p - langkah // 2)
        b = min(
            w - 1,
            p + langkah // 2 + 1
        )

        if b > a:
            posisi = a + int(
                np.argmin(halus[a:b])
            )
            batas.add(posisi)

    ambang = np.percentile(halus, 35)

    for p in range(2, w - 2):
        if (
            halus[p] <= halus[p-1]
            and halus[p] <= halus[p+1]
            and halus[p] <= ambang
        ):
            batas.add(p)

    return sorted(batas)


# =========================================
# PENILAIAN CNN DAN TEMPLATE MATCHING
# =========================================

def skor_potongan(
    blob,
    batas,
    tinggi_acuan,
    model,
    labels,
    templates,
    label_tm
):
    pasangan = []
    array = []

    min_w = max(
        3,
        int(0.17 * tinggi_acuan)
    )

    max_w = max(
        min_w + 1,
        int(1.30 * tinggi_acuan)
    )

    for i in range(len(batas) - 1):

        for j in range(i + 1, len(batas)):
            lebar = batas[j] - batas[i]

            if lebar > max_w:
                break

            if lebar < min_w:
                continue

            pot = normalisasi_potongan(
                blob[:, batas[i]:batas[j]]
            )

            if pot is not None:
                pasangan.append((i, j))
                array.append(pot)

    if not array:
        return {}

    # CNN
    X = np.asarray(
        array,
        dtype=np.float32
    )[..., None] / 255.0

    prob = model.predict(
        X,
        batch_size=128,
        verbose=0
    )

    # Template Matching
    V = X[..., 0].reshape(
        len(X), -1
    )

    V = V / np.maximum(
        np.linalg.norm(
            V,
            axis=1,
            keepdims=True
        ),
        1e-8
    )

    similarity = V @ templates.T

    skor = {}

    for row, (i, j) in enumerate(pasangan):
        c = int(np.argmax(prob[row]))
        t = int(np.argmax(similarity[row]))

        pc = float(prob[row, c])
        pt = float(similarity[row, t])

        panjang = batas[j] - batas[i]

        # Skor pemilihan kandidat.
        # Ini bukan nilai akurasi OCR.
        nilai = (
            0.65 * np.log(max(pc, 1e-6))
            + 0.35 * np.log(max(pt, 1e-6))
            - 0.22 * abs(
                np.log(
                    max(panjang, 1)
                    / max(
                        tinggi_acuan * 0.55,
                        1
                    )
                )
            )
            - 0.55
        )

        # Bonus kecil jika prediksi sama.
        if labels[c] == label_tm[t]:
            nilai += 0.12

        skor[(i, j)] = (
            nilai,
            labels[c],
            label_tm[t],
            pc,
            pt
        )

    return skor


# =========================================
# MEMILIH KOMBINASI PEMOTONGAN
# =========================================

def pilih_batas(batas, skor):
    n = len(batas)

    dp = [-np.inf] * n
    prev = [None] * n

    dp[0] = 0.0

    # Dynamic programming
    for j in range(1, n):
        for i in range(j):

            if (
                (i, j) in skor
                and np.isfinite(dp[i])
            ):
                val = (
                    dp[i]
                    + skor[(i, j)][0]
                )

                if val > dp[j]:
                    dp[j] = val
                    prev[j] = i

    if prev[-1] is None:
        return []

    hasil = []
    j = n - 1

    while j > 0:
        i = prev[j]

        if i is None:
            return []

        hasil.append((
            batas[i],
            batas[j],
            skor[(i, j)]
        ))

        j = i

    return list(reversed(hasil))


# =========================================
# SEGMENTASI UTAMA
# =========================================

def segmentasi_karakter(
    gambar,
    model=None,
    labels=None,
    templates=None,
    label_tm=None
):
    if model is None:
        model, labels = muat_cnn()

    if templates is None:
        templates, label_tm = muat_templates()

    biner = binarisasi(gambar)
    komponen = deteksi_komponen(biner)

    if not komponen:
        return biner, []

    tinggi = [
        h for _, _, _, h in komponen
    ]

    h_max = max(tinggi)

    tinggi_acuan = float(
        np.median([
            h for h in tinggi
            if h >= h_max * 0.45
        ])
    )

    hasil = []

    for x, y, w, h in komponen:
        blob = biner[
            y:y+h,
            x:x+w
        ]

        # Hanya komponen lebar yang
        # dicoba dipisahkan lebih lanjut.
        is_lebar = (
            w > 1.25 * tinggi_acuan
            and h > 0.40 * tinggi_acuan
        )

        if is_lebar:
            batas = calon_batas(
                blob,
                tinggi_acuan
            )
        else:
            batas = [0, w]

        nilai = skor_potongan(
            blob,
            batas,
            tinggi_acuan,
            model,
            labels,
            templates,
            label_tm
        )

        potong = pilih_batas(
            batas,
            nilai
        )

        if not potong:
            hasil.append({
                "x1": x,
                "y1": y,
                "x2": x+w,
                "y2": y+h,
                "cnn": "?",
                "tm": "?",
                "pc": 0.0,
                "pt": 0.0,
                "status": "BELUM_TERURAI"
            })
            continue

        for a, b, info in potong:
            _, c, t, pc, pt = info

            hasil.append({
                "x1": x+a,
                "y1": y,
                "x2": x+b,
                "y2": y+h,
                "cnn": c,
                "tm": t,
                "pc": pc,
                "pt": pt,
                "status": "KANDIDAT"
            })

    return biner, sorted(
        hasil,
        key=lambda z: (
            z["x1"],
            z["y1"]
        )
    )


# =========================================
# MENYUSUN HASIL MENJADI TEKS
# =========================================

def susun_teks(
    hasil,
    metode,
    tinggi_acuan
):
    teks = ""
    akhir = None

    for item in hasil:

        if (
            akhir is not None
            and item["x1"] - akhir
            > 0.58 * tinggi_acuan
        ):
            teks += " "

        teks += item[metode]

        akhir = max(
            item["x2"],
            akhir or 0
        )

    return teks


# =========================================
# PENGUJIAN DAN PENYIMPANAN
# =========================================

def proses(path_gambar):
    path = Path(path_gambar)

    gambar = baca_gambar(path)

    if gambar is None:
        raise ValueError(
            f"Gambar tidak dapat dibaca: {path}"
        )

    output = path.parent / (
        "segmentasi_" + path.stem
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Memuat CNN dan Template Matching..."
    )

    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    biner, hasil = segmentasi_karakter(
        gambar,
        model,
        labels,
        templates,
        label_tm
    )

    visual = cv2.resize(
        gambar,
        (
            biner.shape[1],
            biner.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )

    tinggi = [
        s["y2"] - s["y1"]
        for s in hasil
    ]

    tinggi_acuan = (
        float(np.median(tinggi))
        if tinggi else 12.0
    )

    daftar = []

    for no, item in enumerate(hasil, 1):

        x1, y1, x2, y2 = (
            item[k] for k in (
                "x1", "y1", "x2", "y2"
            )
        )

        if item["status"] == "KANDIDAT":
            warna = (0, 160, 0)
        else:
            warna = (0, 0, 255)

        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            warna,
            1
        )

        crop = cv2.bitwise_not(
            biner[y1:y2, x1:x2]
        )

        simpan_gambar(
            output / "potongan"
            / f"segmen_{no:03d}.png",
            crop
        )

        daftar.append({
            "nomor": no,
            **item
        })

    simpan_gambar(
        output / "biner.png",
        biner
    )

    simpan_gambar(
        output / "hasil_segmentasi.png",
        visual
    )

    with open(
        output / "hasil_prediksi.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        kolom = [
            "nomor",
            "x1", "y1",
            "x2", "y2",
            "cnn", "tm",
            "pc", "pt",
            "status"
        ]

        penulis = csv.DictWriter(
            f,
            fieldnames=kolom
        )

        penulis.writeheader()
        penulis.writerows(daftar)

    teks_cnn = susun_teks(
        hasil,
        "cnn",
        tinggi_acuan
    )

    teks_tm = susun_teks(
        hasil,
        "tm",
        tinggi_acuan
    )

    with open(
        output / "hasil_teks.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"CNN: {teks_cnn}\n"
            f"Template Matching: {teks_tm}\n"
        )

    print("\n=== HASIL OCR EKSPERIMEN ===")
    print("Jumlah kandidat:", len(hasil))
    print("CNN:", teks_cnn)
    print("Template Matching:", teks_tm)
    print("Output:", output)

    print(
        "\nPERINGATAN: Hasil masih perlu "
        "dibandingkan dengan ground truth."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            'Gunakan: python -m '
            'processing.segmentasi_karakter '
            '"path_crop.png"'
        )

    proses(sys.argv[1])
