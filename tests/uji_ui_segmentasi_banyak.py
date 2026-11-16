"""Uji UI Segmentasi pada nota dengan banyak field (23 field)."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"

for nama in ("nota_01.jpeg", "nota_22.jpeg"):
    path = None
    for d in (BASE / "data_nota_asli", BASE / "data_uji"):
        if (d / nama).exists():
            path = d / nama
            break
    if path is None:
        print(f"{nama}: tidak ada")
        continue

    gambar = baca_gambar(path)
    det = appmod.jalankan_deteksi(gambar)
    ocr = appmod.jalankan_ocr_semua(det, False)

    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    at.session_state["gambar_bgr"] = gambar
    at.session_state["nama_file"] = nama
    at.session_state["deteksi"] = det
    at.session_state["ocr_semua"] = ocr
    at.session_state["ocr_kunci"] = (nama,)
    at.sidebar.radio[0].set_value("Segmentasi Karakter").run()

    print(f"\n=== {nama} ({len(det['fields'])} field) ===")
    print("exception :", [e.value for e in at.exception])
    print("expander  :", len(at.expander))
    print("metric    :", [(m.label, m.value) for m in at.metric])
    print("kode blok :", len(at.code))
