"""Uji UI halaman Pengenalan Karakter setelah dirapikan.

Memeriksa: judul sesuai menu, 6 tabs (satu per jenis field, tanpa Satuan),
tanpa expander, tanpa HTML mentah/mojibake, satu subjudul per field, dan
teks OCR setiap field muncul pada halaman.
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
LOG = BASE / "output" / "uji_ui_pengenalan.txt"
baris = []


def catat(t):
    baris.append(str(t))
    print(t, flush=True)


for nama in ("nota_52.jpeg", "nota_01.jpeg"):
    path = None
    for d in (BASE / "data_nota_asli", BASE / "data_uji"):
        if (d / nama).exists():
            path = d / nama
            break
    if path is None:
        catat(f"{nama}: tidak ditemukan")
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
    at.sidebar.radio[0].set_value("Pengenalan Karakter").run()

    judul = [h.value for h in at.header]
    subjudul = [h.value for h in at.subheader]
    kode = [str(c.value) for c in at.code]
    caption = [str(c.value) for c in at.caption]

    teks_semua = " ".join(
        str(getattr(el, "value", ""))
        for jenis_el in ("markdown", "caption", "subheader", "header", "code")
        for el in getattr(at, jenis_el, [])
    )
    rusak = "\u00e2\u20ac\u201d" in teks_semua or "\u00c3\u2014" in teks_semua

    # Setiap field yang punya tab harus muncul sebagai satu subjudul.
    # Field "satuan" tidak ditampilkan pada halaman ini.
    field_tampil = [
        f for f in det["fields"] if f.get("jenis") in appmod.JENIS_TAMPIL
    ]
    harap_subjudul = len(field_tampil)
    # Setiap field dengan teks OCR harus menampilkan teks itu lewat st.code.
    teks_ocr = [
        (h.get("tanpa", {}).get("teks", "") or "").strip()
        for h in (ocr.get(f["id"], {}) for f in field_tampil)
    ]
    harap_kode = sum(1 for t in teks_ocr if t)
    jumlah_caption_karakter = sum(
        1 for c in caption if c.startswith("Jumlah karakter:")
    )

    catat(f"=== {nama} ({len(det['fields'])} field) ===")
    catat(f"  exception          : {[e.value for e in at.exception]}")
    catat(f"  judul              : {judul}")
    catat(f"  tabs               : {len(at.tabs)} (harus 6)")
    catat(f"  expander           : {len(at.expander)} (harus 0)")
    catat(f"  subjudul           : {len(subjudul)} (harap {harap_subjudul})")
    catat(f"  teks OCR (st.code) : {len(kode)} (harap {harap_kode})")
    catat(f"  caption jumlah     : {jumlah_caption_karakter} "
          f"(harap {harap_subjudul})")
    catat(f"  ada HTML mentah    : {'<span' in teks_semua or '<div' in teks_semua}")
    catat(f"  ada mojibake       : {rusak}")

    assert not at.exception, [e.value for e in at.exception]
    assert judul and judul[0] == "Pengenalan Karakter", judul
    assert len(at.tabs) == 6, f"tabs = {len(at.tabs)}"
    assert not at.expander, "halaman tidak boleh memakai expander"
    assert len(subjudul) == harap_subjudul, (len(subjudul), harap_subjudul)
    assert len(kode) == harap_kode, (len(kode), harap_kode)
    assert jumlah_caption_karakter == harap_subjudul, jumlah_caption_karakter
    assert not rusak, "teks halaman masih mengandung mojibake"

    # Teks OCR harus sama persis dengan hasil pipeline.
    for t in teks_ocr:
        if t:
            assert t in kode, f"teks OCR tidak tampil: {t!r}"

LOG.write_text("\n".join(baris), encoding="utf-8")
