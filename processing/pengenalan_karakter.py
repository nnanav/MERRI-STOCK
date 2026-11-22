
from pathlib import Path
from functools import lru_cache
import json
import os
import sys

import cv2
import numpy as np

# Kurangi noise log TensorFlow. Batas thread TIDAK dipaksa ke 1 karena
# membuat inferensi per karakter sangat lambat (pengenalan satu nota bisa
# memakan beberapa menit). Kebutuhan memori sudah ditekan dengan tidak
# lagi menyimpan citra biner/visual yang tidak terpakai.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from processing.dataset_loader import (
    daftar_folder_kelas,
    baca_gambar,
    normalisasi_karakter
)


BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"

MODEL_PATH = MODEL_DIR / "cnn_karakter.keras"
LABEL_PATH = MODEL_DIR / "labels.json"

EKSTENSI = {".png", ".jpg", ".jpeg"}


# ==========================================
# MEMUAT CNN
# ==========================================

@lru_cache(maxsize=1)
def muat_cnn():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model CNN tidak ditemukan: {MODEL_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label CNN tidak ditemukan: {LABEL_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    with open(LABEL_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            "Jumlah output CNN tidak sesuai labels.json"
        )

    return model, labels


# ==========================================
# MEMUAT TEMPLATE MATCHING
# ==========================================

@lru_cache(maxsize=1)
def muat_templates():

    templates = []
    labels = []

    for folder, label in daftar_folder_kelas():

        if not folder.exists():
            continue

        for path in sorted(folder.iterdir()):

            if (
                not path.is_file()
                or path.suffix.lower() not in EKSTENSI
            ):
                continue

            gambar = baca_gambar(path)

            if gambar is None:
                continue

            gambar = normalisasi_karakter(gambar)

            if gambar is None:
                continue

            vektor = gambar.astype(np.float32).flatten()
            norma = np.linalg.norm(vektor)

            if norma == 0:
                continue

            templates.append(vektor / norma)
            labels.append(label)

    if not templates:
        raise RuntimeError(
            "Tidak ada template karakter yang berhasil dimuat."
        )

    return np.stack(templates), labels


# ==========================================
# TEMPLATE MATCHING
# ==========================================

def prediksi_template_matching(gambar):

    normal = normalisasi_karakter(gambar)

    if normal is None:
        raise ValueError("Gambar karakter kosong.")

    templates, labels = muat_templates()

    vektor = normal.astype(np.float32).flatten()
    norma = np.linalg.norm(vektor)

    if norma == 0:
        raise ValueError("Karakter tidak terdeteksi.")

    vektor = vektor / norma

    # Kemiripan kosinus untuk citra yang sejajar.
    skor = templates @ vektor

    indeks = int(np.argmax(skor))

    return {
        "karakter": labels[indeks],
        "skor": float(skor[indeks])
    }


# ==========================================
# CNN
# ==========================================

def prediksi_cnn(gambar):

    normal = normalisasi_karakter(gambar)

    if normal is None:
        raise ValueError("Gambar karakter kosong.")

    model, labels = muat_cnn()

    X = normal.astype(np.float32)
    X = X.reshape(1, 32, 32, 1) / 255.0

    probabilitas = model.predict(
        X,
        verbose=0
    )[0]

    indeks = int(np.argmax(probabilitas))

    return {
        "karakter": labels[indeks],
        "skor": float(probabilitas[indeks])
    }


# ==========================================
# PENGUJIAN KEDUA METODE
# ==========================================

def bandingkan_metode(gambar):

    hasil_tm = prediksi_template_matching(gambar)
    hasil_cnn = prediksi_cnn(gambar)

    return {
        "template_matching": hasil_tm,
        "cnn": hasil_cnn
    }


# Aturan penggabungan CNN + Template Matching (satu pipeline OCR).
# CNN adalah SATU-SATUNYA penentu label karakter. Template Matching
# hanya mencocokkan pola kandidat dan hasilnya disimpan sebagai
# informasi pendukung (dukungan/ketidaksepakatan) untuk evaluasi.
# Ambang dipisah karena skala kedua metode berbeda:
# pc = probabilitas softmax [0..1], pt = cosine similarity.
AMBANG_CNN = 0.60
AMBANG_TM = 0.80


def gabung_keputusan(cnn_char, pc, tm_char, pt,
                     ambang_cnn=AMBANG_CNN, ambang_tm=AMBANG_TM):
    """
    Menentukan label final per karakter.

    Label final SELALU berasal dari CNN. Template Matching tidak pernah
    mengganti label; hasilnya hanya dipakai untuk menandai dukungan pola
    sehingga dapat dievaluasi.

    Mengembalikan (karakter_final, aturan, dukungan_tm).
    """
    if pc >= ambang_cnn:
        aturan = "cnn_kuat"
    elif tm_char == cnn_char:
        aturan = "tm_mendukung"
    else:
        aturan = "tm_berbeda"

    # Dukungan pola dari Template Matching (informasi pendukung saja).
    if tm_char == cnn_char:
        dukungan = "mendukung"
    elif pt >= ambang_tm:
        dukungan = "pola_kuat_beda"
    else:
        dukungan = "lemah"

    return cnn_char, aturan, dukungan


def gabung_segmen(hasil_segmen, ambang_cnn=AMBANG_CNN, ambang_tm=AMBANG_TM):
    # Gabungkan prediksi tiap segmen menjadi keputusan akhir + detail.
    hasil_final = []
    detail = []

    for s in hasil_segmen:
        cnn = s.get("cnn", "")
        tm = s.get("tm", "")
        pc = float(s.get("pc", 0.0) or 0.0)
        pt = float(s.get("pt", 0.0) or 0.0)

        final, aturan, dukungan = gabung_keputusan(
            cnn, pc, tm, pt, ambang_cnn, ambang_tm
        )

        baru = dict(s)
        baru["final"] = final
        baru["aturan"] = aturan
        baru["dukungan_tm"] = dukungan
        hasil_final.append(baru)

        detail.append({
            "x1": s.get("x1"),
            "x2": s.get("x2"),
            "cnn": cnn,
            "pc": round(pc, 4),
            "tm": tm,
            "pt": round(pt, 4),
            "final": final,
            "aturan": aturan,
            "dukungan_tm": dukungan,
        })

    return hasil_final, detail


# ==========================================
# PASCA-PROSES TEKS BERDASARKAN JENIS FIELD
# ==========================================

# Field yang isinya selalu angka pada nota CV Merriena: jumlah barang dan
# harga satuan. Nilainya hanya berisi digit serta pemisah "." dan ",".
JENIS_NUMERIK = ("harga", "jumlah")

# Huruf yang bentuknya sangat mirip angka. Dipakai HANYA pada field
# numerik, karena di sana huruf tidak mungkin muncul sehingga huruf
# tersebut pasti salah baca angka.
_MIRIP_ANGKA = {
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "l": "1", "i": "1", "j": "1", "L": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
    "T": "7",
    "A": "4",
    "g": "9", "q": "9",
}

# Huruf kecil yang mudah tertukar dengan huruf besar pada teks nota.
_HURUF_KONFUSABLE = {"l", "i"}


def bersihkan_numerik(teks):
    """
    Rapikan teks field numerik (jumlah, harga satuan).

    Hanya digit dan pemisah "." / "," yang dipertahankan. Huruf yang
    bentuknya mirip angka diubah ke angka (mis. "1Sx8" -> "15x8",
    "lO.5S0" -> "10.550"), sedangkan karakter lain dibuang karena pada
    field ini tidak mungkin muncul.
    """
    if not teks:
        return teks

    keluar = []
    for c in str(teks):
        if c.isdigit() or c in ".,":
            keluar.append(c)
        elif c in _MIRIP_ANGKA:
            keluar.append(_MIRIP_ANGKA[c])

    hasil = "".join(keluar).strip(".,")

    # Pemisah ganda berurutan (mis. ".." atau ",.") tidak wajar.
    while ".." in hasil or ",," in hasil or ".," in hasil or ",." in hasil:
        hasil = (hasil.replace("..", ".").replace(",,", ",")
                 .replace(".,", ".").replace(",.", ","))
    return hasil


def rapikan_huruf(teks):
    """
    Perbaiki huruf besar/kecil yang mudah tertukar pada teks nota.

    Dua kesalahan yang paling sering muncul karena setiap karakter
    dinormalkan ke kanvas 32x32 sehingga ukuran aslinya hilang:

    1. Huruf 'I' terbaca 'l' pada kata berhuruf besar semua
       (mis. "PSSI" menjadi "PSSl", "IKN" menjadi "lKN").
       Bila satu kata hanya memuat huruf kecil 'l'/'i' di antara huruf
       besar, huruf kecil itu dikembalikan menjadi 'I'.

    2. Huruf 'l' terbaca 'I' di tengah kata berhuruf campuran
       (mis. "Lantal" menjadi "LantaI"). Bila kata memuat huruf kecil
       lain, 'I' yang bukan huruf pertama kata dikembalikan menjadi 'l'.

    Kata yang seluruhnya huruf besar (mis. "KSO", "PSSI") dan kata yang
    wajar berhuruf besar di awal (mis. "Indra") tidak diubah.
    """
    if not teks:
        return teks

    import re

    def perbaiki(kata):
        huruf = [c for c in kata if c.isalpha()]
        if len(huruf) < 2:
            return kata

        kecil = [c for c in huruf if c.islower()]

        # (1) Kata huruf besar yang hanya memuat 'l'/'i' kecil.
        if kecil and all(c in _HURUF_KONFUSABLE for c in kecil):
            return kata.replace("l", "I").replace("i", "I")

        # (2) Kata campuran: 'I' di tengah kata hampir pasti 'l'.
        if kecil and len(kata) > 1:
            depan = kata[0]
            sisa = kata[1:].replace("I", "l")
            return depan + sisa

        return kata

    bagian = re.split(r"(\s+)", str(teks))
    return "".join(
        b if not b.strip() else perbaiki(b) for b in bagian
    )


# Huruf yang dipakai pada aturan "menempel pada digit". Hanya huruf yang
# bentuknya hampir sama dengan angka DAN hampir tidak mungkin menjadi
# bagian kode produk. Huruf seperti L, T, A, U sengaja TIDAK dipakai di
# sini karena sering muncul wajar berdampingan dengan angka
# (mis. "PAMA L4", "K-300", "PU 50x30").
_MIRIP_ANGKA_MENEMPEL = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
}


def rapikan_angka_menempel(teks):
    """
    Ubah huruf mirip angka yang MENEMPEL pada angka menjadi angka.

    Pada nama produk sering muncul ukuran seperti "40x30x15x8". Huruf
    'S' pada "1Sx8" hampir pasti angka 5, karena berdampingan langsung
    dengan digit. Sebaliknya huruf berdiri sendiri (mis. "Konstein S
    40x30...") atau bagian kode produk (mis. "PAMA L4", "K-300") TIDAK
    diubah.

    Hanya berlaku bila huruf tersebut bersebelahan langsung (tanpa spasi)
    dengan sebuah digit.
    """
    if not teks:
        return teks

    teks = str(teks)
    keluar = []
    for i, c in enumerate(teks):
        if c in _MIRIP_ANGKA_MENEMPEL:
            kiri = teks[i - 1] if i > 0 else ""
            kanan = teks[i + 1] if i + 1 < len(teks) else ""
            if kiri.isdigit() or kanan.isdigit():
                keluar.append(_MIRIP_ANGKA_MENEMPEL[c])
                continue
        keluar.append(c)
    return "".join(keluar)


def rapikan_kali(teks):
    """
    Perbaiki tanda kali pada ukuran produk yang terbaca sebagai huruf.

    Nama produk memuat ukuran seperti "50x30x18x13". Huruf 'x' sempit
    dan sering dikenali 'K' (mis. "18K13"), sehingga ukuran menjadi
    tidak terbaca. Pada field produk, satu huruf yang diapit dua digit
    selalu tanda kali, bukan huruf.

    Penggantian dibatasi pada huruf yang bentuknya mirip 'x' agar kode
    produk lain (mis. "K-300") tidak ikut berubah.
    """
    if not teks:
        return teks

    import re

    mirip_kali = "KkXxYyVvHhNnMmWw*×"
    pola = r"(?<=\d)([" + re.escape(mirip_kali) + r"])(?=\d)"
    return re.sub(pola, "x", str(teks))


def rapikan_teks(teks, jenis=None):
    """
    Terapkan pasca-proses yang sesuai jenis field.

    Field numerik (jumlah, harga satuan) dibersihkan menjadi angka;
    field teks (nomor, penerima, proyek, produk) diperbaiki huruf
    besar/kecilnya dan huruf mirip angka yang menempel pada digit.
    Nama produk juga diperbaiki tanda kalinya. Model CNN, Template
    Matching, dan segmentasi tidak diubah.
    """
    if jenis in JENIS_NUMERIK:
        return bersihkan_numerik(teks)
    if jenis == "produk":
        return rapikan_huruf(rapikan_angka_menempel(rapikan_kali(teks)))
    if jenis in ("nomor", "penerima", "proyek"):
        return rapikan_huruf(rapikan_angka_menempel(teks))
    return teks


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Penggunaan: python -m '
            'processing.pengenalan_karakter "path_gambar"'
        )
        sys.exit(1)

    path = Path(sys.argv[1])

    gambar = baca_gambar(path)

    if gambar is None:
        raise ValueError(
            f"Gambar tidak dapat dibaca: {path}"
        )

    hasil = bandingkan_metode(gambar)

    tm = hasil["template_matching"]
    cnn = hasil["cnn"]

    print("\n=== PERBANDINGAN PENGENALAN ===")

    print("\nTemplate Matching")
    print("Karakter:", repr(tm["karakter"]))
    print(f"Skor kemiripan: {tm['skor']:.4f}")

    print("\nCNN")
    print("Karakter:", repr(cnn["karakter"]))
    print(f"Skor softmax: {cnn['skor']:.4f}")

    print("\nKedua metode sepakat:",
          tm["karakter"] == cnn["karakter"])
