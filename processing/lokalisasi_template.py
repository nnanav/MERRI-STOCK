"""
Lokalisasi karakter dengan Template Matching.

Peran modul ini sesuai metode penelitian:
- Template Matching  : menemukan/mencocokkan pola karakter pada citra nota
                       memakai template karakter dari dataset proyek
                       (dataset/huruf, dataset/angka, dataset/simbol).
- CNN                : mengenali karakter pada tiap kandidat yang ditemukan
                       (dilakukan di segmentasi_karakter, bukan di sini).

Di sini tidak ada klasifikasi: keluaran hanya posisi kandidat + label
template + skor kecocokan pola.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from processing.dataset_loader import (
    baca_gambar, normalisasi_karakter, daftar_folder_kelas,
)


BASE = Path(__file__).resolve().parent.parent
EKSTENSI = {".png", ".jpg", ".jpeg"}

NAMA_SIMBOL = {
    "koma": ",",
    "titik": ".",
    "strip": "-",
    "minus": "-",
    "slash": "/",
    "persen": "%",
    "petik": '"',
    "kurung_buka": "(",
    "kurung_tutup": ")",
    "samadengan": "=",
    "titik_dua": ":",
}


def label_dari_nama(stem):
    # Nama file template manual: "A_01", "a_02", "koma_01", "angka_1_01".
    for nama in sorted(NAMA_SIMBOL, key=len, reverse=True):
        if stem == nama or stem.startswith(nama + "_"):
            return NAMA_SIMBOL[nama]

    bagian = stem.split("_")
    for calon in bagian:
        if len(calon) == 1 and (calon.isalnum()):
            return calon
    return None


@lru_cache(maxsize=1)
def muat_template_manual(maks_per_kelas=6):
    """
    Memuat template karakter dari dataset yang dipakai proyek:
    dataset/huruf, dataset/angka, dataset/simbol (via daftar_folder_kelas).

    Setiap template disimpan sebagai citra biner yang sudah dipangkas
    rapat ke bentuk karakter, memakai normalisasi yang sama dengan input CNN.

    maks_per_kelas membatasi jumlah template per kelas (dipilih merata)
    agar pencocokan multi-skala tetap cepat.
    """
    templates = []
    labels = []

    for folder, label in daftar_folder_kelas():
        if not folder.exists():
            continue

        berkas = [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in EKSTENSI
        ]
        if not berkas:
            continue

        if maks_per_kelas and len(berkas) > maks_per_kelas:
            langkah = len(berkas) / maks_per_kelas
            berkas = [berkas[int(i * langkah)] for i in range(maks_per_kelas)]

        for path in berkas:
            gambar = baca_gambar(path)
            if gambar is None:
                continue

            normal = normalisasi_karakter(gambar)
            if normal is None:
                continue

            # Pangkas rapat ke bbox karakter (buang padding 32x32) agar
            # pencocokan pola tidak didominasi area kosong.
            nz = cv2.findNonZero(normal)
            if nz is None:
                continue
            x, y, w, h = cv2.boundingRect(nz)
            if w < 2 or h < 3:
                continue

            templates.append(normal[y:y + h, x:x + w].copy())
            labels.append(label)

    return templates, labels


def _tinggi_teks(biner):
    # Perkiraan tinggi karakter dari komponen tinggi pada baris teks.
    n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    tinggi = [int(s[3]) for s in stats[1:] if s[3] >= 5 and s[4] >= 8]
    if not tinggi:
        return max(8, int(biner.shape[0] * 0.6))
    maks = max(tinggi)
    besar = [h for h in tinggi if h >= 0.5 * maks]
    return int(np.median(besar))


def _nms(kandidat, ambang_iou=0.30):
    # Buang kandidat yang saling menumpuk, ambil skor tertinggi.
    kandidat = sorted(kandidat, key=lambda k: k["skor"], reverse=True)
    simpan = []
    for k in kandidat:
        tumpuk = False
        for s in simpan:
            ax1 = max(k["x1"], s["x1"])
            ay1 = max(k["y1"], s["y1"])
            ax2 = min(k["x2"], s["x2"])
            ay2 = min(k["y2"], s["y2"])
            if ax2 <= ax1 or ay2 <= ay1:
                continue
            inter = (ax2 - ax1) * (ay2 - ay1)
            luas_k = (k["x2"] - k["x1"]) * (k["y2"] - k["y1"])
            luas_s = (s["x2"] - s["x1"]) * (s["y2"] - s["y1"])
            iou = inter / max(1, luas_k + luas_s - inter)
            if iou > ambang_iou:
                tumpuk = True
                break
        if not tumpuk:
            simpan.append(k)
    return simpan


def _slot_tata_letak(biner, char_w):
    """
    Menentukan slot kira-kira satu karakter dari tinta pada baris teks.

    Slot dipakai sebagai batas wilayah pencarian Template Matching,
    sehingga jumlah kandidat mengikuti tata letak (bukan semua puncak).
    """
    proj = np.count_nonzero(biner, axis=0)
    W = len(proj)
    ada = proj > 0

    runs = []
    mulai = None
    for x in range(W):
        if ada[x]:
            if mulai is None:
                mulai = x
        else:
            if mulai is not None:
                runs.append((mulai, x - 1))
                mulai = None
    if mulai is not None:
        runs.append((mulai, W - 1))

    slot = []
    for a, b in runs:
        lebar = b - a + 1
        n = max(1, int(round(lebar / max(1.0, char_w))))
        batas = [a + round(k * lebar / n) for k in range(n + 1)]
        for k in range(n):
            slot.append((batas[k], batas[k + 1]))
    return slot


def cari_kandidat_karakter(crop, ambang_skor=0.30, skala=(0.85, 1.0, 1.15)):
    """
    Menentukan posisi karakter pada satu baris/field nota dengan Template
    Matching, lalu mengembalikan kandidat per posisi.

    Cara kerja:
    1. Tinta pada baris dibagi menjadi slot kira-kira satu karakter.
    2. Untuk setiap slot, semua template manual dicocokkan memakai
       cv2.matchTemplate (TM_CCOEFF_NORMED) multi-skala.
    3. Kandidat terbaik (skor tertinggi) di dalam slot dipilih sebagai
       posisi + pola karakter.

    Kembalian: daftar kandidat berisi bbox, label template, dan skor.
    """
    if crop is None or crop.size == 0:
        return []

    abu = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = abu.shape[:2]
    if h < 6 or w < 6:
        return []

    _, biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    biner = cv2.GaussianBlur(biner, (3, 3), 0)
    tinggi = _tinggi_teks(biner)

    templates, labels = muat_template_manual()
    if not templates:
        return []

    # Lebar karakter perkiraan dari template (median rasio lebar/tinggi).
    rasio = [t.shape[1] / max(1, t.shape[0]) for t in templates]
    char_w = max(4.0, float(np.median(rasio)) * tinggi)

    slot = _slot_tata_letak(biner, char_w)
    if not slot:
        return []

    kandidat = []

    for (sx1, sx2) in slot:
        lebar_slot = max(2, sx2 - sx1)
        terbaik = None

        # Lebar tinta nyata di dalam slot (untuk menilai kesesuaian bentuk).
        potong_ink = biner[:, sx1:sx2 + 1]
        kolom_ink = np.where(np.count_nonzero(potong_ink, axis=0) > 0)[0]
        if len(kolom_ink) == 0:
            continue
        lebar_ink = max(2, int(kolom_ink.max() - kolom_ink.min() + 1))

        for faktor in skala:
            tinggi_t = max(6, int(round(tinggi * faktor)))

            for tmpl, label in zip(templates, labels):
                th, tw = tmpl.shape[:2]
                lebar_t = max(3, int(round(tw * tinggi_t / max(1, th))))
                if lebar_t > lebar_slot * 1.8:
                    continue
                if tinggi_t > biner.shape[0] or lebar_t > biner.shape[1]:
                    continue

                t = cv2.resize(tmpl, (lebar_t, tinggi_t),
                               interpolation=cv2.INTER_AREA)
                t = cv2.GaussianBlur(t, (3, 3), 0)

                hasil = cv2.matchTemplate(biner, t, cv2.TM_CCOEFF_NORMED)
                hasil = np.nan_to_num(hasil, nan=0.0, posinf=0.0, neginf=0.0)

                # Cari puncak di dalam wilayah slot saja.
                x_lo = max(0, sx1 - 2)
                x_hi = min(hasil.shape[1] - 1, sx2)
                if x_hi < x_lo:
                    continue

                potong = hasil[:, x_lo:x_hi + 1]
                if potong.size == 0:
                    continue

                nilai = float(potong.max())
                if nilai < ambang_skor:
                    continue

                # Penalti bila lebar pola jauh berbeda dari lebar tinta
                # slot, agar template kurus (mis. 'l') tidak selalu menang.
                selisih = abs(lebar_t - lebar_ink) / max(1, lebar_ink)
                nilai_adj = nilai - 0.45 * min(1.0, selisih)

                if terbaik is not None and nilai_adj <= terbaik["skor"]:
                    continue

                idx = np.unravel_index(int(np.argmax(potong)), potong.shape)
                cx = int(x_lo + idx[1])
                cy = int(idx[0])
                terbaik = {
                    "x1": cx,
                    "y1": cy,
                    "x2": cx + lebar_t,
                    "y2": cy + tinggi_t,
                    "template": label,
                    "skor": nilai,
                    "skor_adj": nilai_adj,
                }

        if terbaik is not None:
            kandidat.append(terbaik)

    kandidat.sort(key=lambda k: k["x1"])
    return kandidat