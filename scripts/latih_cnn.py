
from pathlib import Path
import csv
import json
import random
import sys

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from processing.dataset_loader import muat_dataset


# ==========================================
# KONFIGURASI
# ==========================================

BASE = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE / "models"
DATASET_DIR = BASE / "dataset"
CSV_VALIDASI = (
    BASE / "output" / "laporan_dataset" / "hasil_validasi.csv"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "cnn_karakter.keras"
LABEL_PATH = MODEL_DIR / "labels.json"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 32
VALIDASI = 0.20

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ==========================================
# BACA HASIL VALIDASI
# ==========================================

def baca_status_dataset():
    """
    Ambil file yang berstatus selain OK
    dari laporan validasi dataset.
    """

    bermasalah = set()

    if not CSV_VALIDASI.exists():
        raise FileNotFoundError(
            "Laporan validasi tidak ditemukan. "
            "Jalankan validasi_dataset.py dahulu."
        )

    with open(
        CSV_VALIDASI,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        pembaca = csv.DictReader(f)

        for baris in pembaca:
            status = baris["status"].strip()

            if status != "OK":
                path = baris["file"].replace("\\", "/")
                bermasalah.add(path)

    return bermasalah


# ==========================================
# PERSIAPAN DATA LATIH
# ==========================================

print("\n=== MEMUAT DATASET ===")

X, y, labels, info = muat_dataset()

if len(X) == 0:
    raise RuntimeError(
        "Dataset kosong atau gagal dibaca."
    )

bermasalah = baca_status_dataset()

indeks_valid = []

for i, item in enumerate(info):
    path = Path(item["file"])

    relatif = path.relative_to(
        DATASET_DIR
    ).as_posix()

    if relatif not in bermasalah:
        indeks_valid.append(i)

X = X[indeks_valid]
y = y[indeks_valid]

print("Total dataset awal:", len(info))
print("Total data latih:", len(X))
print("Data dikecualikan:", len(info) - len(X))
print("Jumlah kelas:", len(labels))

if len(labels) != 70:
    raise RuntimeError(
        f"Jumlah kelas {len(labels)}, "
        "seharusnya 70. Periksa dataset."
    )

# Pastikan tidak ada kelas yang hilang
kelas_tersedia = set(y.tolist())

if kelas_tersedia != set(range(len(labels))):
    raise RuntimeError(
        "Ada kelas yang tidak memiliki data latih."
    )


# ==========================================
# ARSITEKTUR CNN
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(32, 32, 1)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(labels),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# PELATIHAN
# ==========================================

print("\n=== MULAI PELATIHAN CNN ===")

# Data dibagi menjadi latih dan validasi (stratified per kelas) agar
# overfitting dapat dideteksi. Pengujian akhir tetap memakai nota baru.
# Split dilakukan manual per kelas agar tidak menambah dependensi baru.
rng = np.random.RandomState(SEED)

indeks_latih = []
indeks_val = []

for kelas in np.unique(y):
    idx = np.where(y == kelas)[0]
    rng.shuffle(idx)

    n_val = max(1, int(round(len(idx) * VALIDASI)))
    indeks_val.extend(idx[:n_val])
    indeks_latih.extend(idx[n_val:])

indeks_latih = np.array(sorted(indeks_latih))
indeks_val = np.array(sorted(indeks_val))

X_latih, y_latih = X[indeks_latih], y[indeks_latih]
X_val, y_val = X[indeks_val], y[indeks_val]

print("Data latih   :", len(X_latih))
print("Data validasi:", len(X_val))

history = model.fit(
    X_latih,
    y_latih,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1,

    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.001,
            restore_best_weights=True
        )
    ]
)


# ==========================================
# SIMPAN MODEL DAN LABEL
# ==========================================

model.save(str(MODEL_PATH))

with open(
    LABEL_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        labels,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================
# RINGKASAN HASIL PELATIHAN
# ==========================================

loss, akurasi = model.evaluate(
    X_latih,
    y_latih,
    verbose=0
)

loss_val, akurasi_val = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

best = history.history

print("\n================================")
print("PELATIHAN CNN SELESAI")
print("================================")

print("Jumlah kelas:", len(labels))
print("Jumlah data latih:", len(X_latih))
print("Jumlah data validasi:", len(X_val))
print("Epoch terlaksana:", len(best["loss"]))

print(f"\nTraining loss       : {loss:.4f}")
print(f"Training accuracy   : {akurasi * 100:.2f}%")
print(f"Validation loss     : {loss_val:.4f}")
print(f"Validation accuracy : {akurasi_val * 100:.2f}%")

if "val_accuracy" in best:
    print(f"Val accuracy terbaik: {max(best['val_accuracy']) * 100:.2f}% "
          f"(epoch {int(np.argmax(best['val_accuracy'])) + 1})")

print("\nModel tersimpan:")
print(MODEL_PATH)

print("\nLabel tersimpan:")
print(LABEL_PATH)

print(
    "\nPERHATIAN: Training accuracy bukan "
    "akurasi pada nota baru."
)