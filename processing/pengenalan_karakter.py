
from pathlib import Path
from functools import lru_cache
import json
import sys

import cv2
import numpy as np
import tensorflow as tf

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
