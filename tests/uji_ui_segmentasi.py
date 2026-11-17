"""Uji UI Segmentasi: tanpa metrik & tabel ringkasan, hanya kartu per field."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
LOG = BASE / "output" / "hasil_uji_ui.txt"
baris = []


def catat(t):
    baris.append(str(t))
    print(t, flush=True)


for nama in ("nota_52.jpeg", "nota_22.jpeg"):
    path = None
    for d in (BASE / "data_nota_asli", BASE / "data_uji"):
        if (d / nama).exists():
            path = d / nama
            break
    if path is None:
        catat(f"{nama}: tidak ada")
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

    teks_md = " ".join(str(getattr(el, "value", "")) for el in at.markdown)

    catat(f"=== {nama} ({len(det['fields'])} field) ===")
    catat(f"  exception : {[e.value for e in at.exception]}")
    catat(f"  expander  : {len(at.expander)} (harus 0)")
    catat(f"  metric    : {len(at.metric)} (harus 0)")
    catat(f"  dataframe : {len(at.dataframe)} (harus 0)")
    catat(f"  subheader : {[h.value for h in at.subheader]}")
    catat(f"  kartu     : {teks_md.count('kotak karakter')}")
    catat(f"  kode blok : {len(at.code)}")
    catat(f"  ada 'Ringkasan seluruh field': "
          f"{'Ringkasan seluruh field' in teks_md}")

LOG.write_text("\n".join(baris), encoding="utf-8")


