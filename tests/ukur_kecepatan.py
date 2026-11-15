"""Ukur kecepatan memuat halaman Segmentasi / Pengenalan / Evaluasi.

Membandingkan muat pertama (cache kosong) dengan muat kedua (cache terisi),
yang mewakili pengalaman memakai aplikasi sehari-hari.
"""

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
NOTA = BASE / "data_uji" / "nota_52.jpeg"
gambar = baca_gambar(NOTA)

t0 = time.time()
det = appmod.jalankan_deteksi(gambar)
print(f"deteksi field          : {time.time() - t0:6.1f} s")

ocr = appmod.jalankan_ocr_semua(det, False)
print("ocr semua field siap", flush=True)


def ukur(halaman, ulang=2):
    hasil = []
    for i in range(ulang):
        at = AppTest.from_file(str(APP), default_timeout=600)
        at.run()
        at.session_state["gambar_bgr"] = gambar
        at.session_state["nama_file"] = "nota_52.jpeg"
        at.session_state["deteksi"] = det
        at.session_state["ocr_semua"] = ocr
        at.session_state["ocr_kunci"] = ("nota_52.jpeg",)
        t = time.time()
        at.sidebar.radio[0].set_value(halaman).run()
        hasil.append(time.time() - t)
        if at.exception:
            print(f"   exception: {at.exception[0].value[:120]}")
    return hasil


for hal in ("Segmentasi Karakter", "Pengenalan Karakter", "Evaluasi"):
    d = ukur(hal)
    print(f"{hal:22s}: muat-1 {d[0]:6.1f} s | muat-2 {d[1]:6.1f} s")
