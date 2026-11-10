"""
Menambahkan template karakter dari nota langsung ke dataset/ yang sudah ada.

Struktur tujuan (TIDAK membuat folder baru):
  dataset/huruf/uppercase/<A-Z>/
  dataset/huruf/lowercase/<a-z>/
  dataset/angka/<0-9>/
  dataset/simbol/<koma|titik|strip|slash|persen|petik|kurung_buka|kurung_tutup>/

Crop dirapikan (rapat ke tinta + margin seragam) sehingga tampak seperti
dipotong manual. Nama file berawalan "nota_" agar mudah dibedakan dari
data asli (asli_) dan augmentasi (aug_).

Label diambil dari prediksi CNN. Karakter yang tidak termasuk 70 kelas
dataset akan dilewati.
"""

import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter, SKALA

MARGIN = 2

# Folder simbol yang tersedia di dataset.
SIMBOL = {
    ",": "koma",
    ".": "titik",
    "-": "strip",
    "/": "slash",
    "%": "persen",
    '"': "petik",
    "(": "kurung_buka",
    ")": "kurung_tutup",
}

FIELD = ("nomor", "jumlah_1", "jumlah_2", "jumlah_3",
         "harga_1", "harga_2", "harga_3",
         "satuan_1", "satuan_2", "satuan_3",
         "produk_1", "produk_2", "produk_3",
         "penerima", "proyek")


def folder_tujuan(label):
    """Menentukan folder dataset yang sudah ada untuk sebuah label."""
    if len(label) != 1:
        return None

    if label.isdigit():
        return BASE / "dataset" / "angka" / label

    if label.isupper():
        return BASE / "dataset" / "huruf" / "uppercase" / label

    if label.islower():
        return BASE / "dataset" / "huruf" / "lowercase" / label

    if label in SIMBOL:
        return BASE / "dataset" / "simbol" / SIMBOL[label]

    return None


def rapikan_crop(abu):
    """Rapatkan ke tinta + margin seragam agar seperti crop manual."""
    if abu is None or abu.size == 0:
        return None

    _, biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    nz = cv2.findNonZero(biner)
    if nz is None:
        return None

    x, y, w, h = cv2.boundingRect(nz)
    if w < 2 or h < 3:
        return None

    inti = abu[y:y + h, x:x + w]

    kanvas = np.full(
        (h + 2 * MARGIN, w + 2 * MARGIN), 255, dtype=np.uint8
    )
    kanvas[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = inti
    return kanvas


def simpan(path, gambar):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", gambar)
    if ok:
        data.tofile(str(path))


def daftar_nota():
    """Semua nota yang tersedia, agar template lebih kaya."""
    hasil = []
    for folder in (BASE / "data_nota_asli", BASE / "data_uji"):
        if not folder.exists():
            continue
        for pola in ("*.jpeg", "*.jpg", "*.png"):
            for p in sorted(folder.glob(pola)):
                hasil.append(p)
    return hasil


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    semua_nota = daftar_nota()
    print(f"Nota yang diproses: {len(semua_nota)}")
    for p in semua_nota:
        print("  -", p.parent.name + "/" + p.name)

    jumlah = Counter()
    dilewati = Counter()
    total = 0

    for path in semua_nota:
        nama_nota = path.stem

        det = app.jalankan_deteksi(baca_gambar(path))

        for f in det["fields"]:
            if f["id"] not in FIELD:
                continue
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue

            img = app.ke_tigakanal(crop)
            _, segmen = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            if not segmen:
                continue

            besar = cv2.resize(
                img, None, fx=SKALA, fy=SKALA,
                interpolation=cv2.INTER_CUBIC
            )

            urut = 0
            for s in segmen:
                y1 = max(0, min(besar.shape[0] - 1, int(s["y1"])))
                y2 = max(y1 + 1, min(besar.shape[0], int(s["y2"])))
                x1 = max(0, min(besar.shape[1] - 1, int(s["x1"])))
                x2 = max(x1 + 1, min(besar.shape[1], int(s["x2"])))

                potong = besar[y1:y2, x1:x2]
                if potong.size == 0:
                    continue
                if potong.ndim == 3:
                    potong = cv2.cvtColor(potong, cv2.COLOR_BGR2GRAY)

                rapi = rapikan_crop(potong)
                if rapi is None:
                    continue

                label = s.get("cnn", "")
                tujuan = folder_tujuan(label)

                if tujuan is None:
                    dilewati[label or "?"] += 1
                    continue

                urut += 1
                simpan(
                    tujuan / f"nota_{nama_nota}_{urut:03d}.png",
                    rapi,
                )
                jumlah[label] += 1
                total += 1

    print("\n=== SELESAI ===")
    print(f"Total crop masuk dataset : {total}")
    print(f"Kelas terisi             : {len(jumlah)}")
    print(f"Karakter dilewati        : {sum(dilewati.values())}")

    print("\nRincian per kelas (label -> jumlah crop baru):")
    for k, v in sorted(jumlah.items()):
        print(f"  {k!r:5s} : {v}")

    if dilewati:
        print("\nDilewati (bukan 70 kelas dataset):")
        for k, v in dilewati.most_common():
            print(f"  {k!r:5s} : {v}")


if __name__ == "__main__":
    main()