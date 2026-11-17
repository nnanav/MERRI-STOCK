"""Render halaman dengan nota nyata (session state terisi) -> pastikan tanpa error."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
NOTA = BASE / "data_uji" / "nota_26.jpeg"
gambar = baca_gambar(NOTA)
det = appmod.jalankan_deteksi(gambar)
ocr = appmod.jalankan_ocr_semua(det, False)

print(f"deteksi: {len(det['fields'])} field")
print(f"kunci det: {sorted(det.keys())}")

gagal = 0
for halaman in ("Preprocessing", "Segmentasi Karakter", "Pengenalan Karakter",
                "Hasil Ekstraksi", "Evaluasi"):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.session_state["gambar_bgr"] = gambar
    at.session_state["nama_file"] = "nota_26.jpeg"
    at.session_state["deteksi"] = det
    at.session_state["ocr_semua"] = ocr
    at.session_state["ocr_kunci"] = ("nota_26.jpeg",)
    at.sidebar.radio[0].set_value(halaman).run()
    galat = [e.value for e in at.exception]
    gagal += len(galat)
    print(f"[{halaman}] exception={galat}")
    if galat:
        print("   detail:", str(galat[0])[:300])

print(f"\nTotal halaman gagal: {gagal}")
