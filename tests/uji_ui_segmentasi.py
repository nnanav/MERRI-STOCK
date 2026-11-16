"""Uji halaman Segmentasi: harus tanpa dropdown/expander, tanpa exception."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
NOTA = BASE / "data_uji" / "nota_52.jpeg"
gambar = baca_gambar(NOTA)
det = appmod.jalankan_deteksi(gambar)
ocr = appmod.jalankan_ocr_semua(det, False)

at = AppTest.from_file(str(APP), default_timeout=600)
at.run()
at.session_state["gambar_bgr"] = gambar
at.session_state["nama_file"] = "nota_52.jpeg"
at.session_state["deteksi"] = det
at.session_state["ocr_semua"] = ocr
at.session_state["ocr_kunci"] = ("nota_52.jpeg",)
at.sidebar.radio[0].set_value("Segmentasi Karakter").run()

print("exception   :", [e.value for e in at.exception])
print("expander    :", len(at.expander), "(harus 0)")
print("subheader   :", [h.value for h in at.subheader])
print("markdown    :", len(at.markdown))
print("dataframe   :", len(at.dataframe))
print("metric      :", [(m.label, m.value) for m in at.metric])
print("code        :", len(at.code))
print("caption     :", len(at.caption))
