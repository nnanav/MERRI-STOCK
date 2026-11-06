
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import tensorflow as tf

from processing.dataset_loader import (
    baca_gambar,
    normalisasi_karakter
)

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "cnn_karakter.keras"
LABEL_PATH = BASE / "models" / "labels.json"

# Penggunaan:
# python uji_cnn.py "D:\...\contoh_karakter.png"

if len(sys.argv) < 2:
    print("Gunakan: python uji_cnn.py path_gambar")
    sys.exit(1)

gambar_path = Path(sys.argv[1])

if not gambar_path.exists():
    print("File gambar tidak ditemukan:", gambar_path)
    sys.exit(1)

model = tf.keras.models.load_model(
    str(MODEL_PATH)
)

with open(LABEL_PATH, encoding="utf-8") as f:
    labels = json.load(f)

gambar = baca_gambar(gambar_path)

if gambar is None:
    raise ValueError("Gambar tidak dapat dibaca.")

gambar = normalisasi_karakter(gambar)

if gambar is None:
    raise ValueError("Karakter kosong.")

X = gambar.astype(np.float32)
X = X.reshape(1, 32, 32, 1) / 255.0

prediksi = model.predict(X, verbose=0)[0]

urutan = np.argsort(prediksi)[::-1][:5]

print("\n=== HASIL PREDIKSI CNN ===")

for indeks in urutan:
    label = labels[int(indeks)]
    nilai = prediksi[indeks] * 100

    print(f"{repr(label):5} : {nilai:.2f}%")

print("\nPrediksi utama:", repr(labels[int(urutan[0])]))
