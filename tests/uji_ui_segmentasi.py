"""Uji UI halaman Segmentasi Karakter setelah revisi.

Memeriksa: 7 tabs (satu per jenis field), tanpa expander, tanpa HTML
mentah, caption karakter sesuai format "Karakter N", dan jumlah gambar
sama dengan jumlah karakter hasil segmentasi sebenarnya.
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
LOG = BASE / "output" / "uji_ui_segmentasi_revisi.txt"
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
    at.sidebar.radio[0].set_value("Segmentasi Karakter").run()

    teks_md = " ".join(str(getattr(el, "value", "")) for el in at.markdown)
    teks_cap = [str(getattr(el, "value", "")) for el in at.caption]

    # Caption pada st.image tersimpan di elemen gambar, bukan st.caption.
    # Hitung elemen gambar dengan menelusuri pohon elemen.
    def hitung_gambar(node, hasil):
        for anak in list(getattr(node, "children", {}).values()):
            nama_kelas = type(anak).__name__
            tipe = str(getattr(anak, "type", "")).lower()
            if "Image" in nama_kelas or "img" in tipe:
                hasil.append(anak)
            hitung_gambar(anak, hasil)

    daftar_gambar = []
    hitung_gambar(at.main, daftar_gambar)
    n_gambar = len(daftar_gambar)

    # Hitung total karakter dari pipeline segmentasi sebenarnya.
    total = 0
    for f in det["fields"]:
        crop = f.get("crop_show")
        if crop is None or not getattr(crop, "size", 0):
            continue
        _, kar, _b, _t, _d = appmod.segmen_field(crop, f.get("jenis"))
        total += len(kar)

    # Semua jenis field harus punya tab, termasuk satuan.
    jenis_tab = ("nomor", "penerima", "proyek", "produk", "jumlah",
                 "satuan", "harga")
    field_tab = [f for f in det["fields"] if f.get("jenis") in jenis_tab]
    karakter_tab = 0
    for f in field_tab:
        crop = f.get("crop_show")
        if crop is None or not getattr(crop, "size", 0):
            continue
        _, kar, _b, _t, _d = appmod.segmen_field(crop, f.get("jenis"))
        karakter_tab += len(kar)

    # Perkiraan: 2 gambar per field (asli + bounding box) + potongan karakter.
    harap_gambar = 2 * len(field_tab) + karakter_tab

    catat(f"=== {nama} ({len(det['fields'])} field) ===")
    catat(f"  exception         : {[e.value for e in at.exception]}")
    catat(f"  tabs              : {len(at.tabs)} (harus 7)")
    catat(f"  expander          : {len(at.expander)} (harus 0)")
    catat(f"  gambar            : {n_gambar} "
          f"(harap 2x{len(field_tab)} + {karakter_tab} = {harap_gambar})")
    catat(f"  gambar cocok      : {n_gambar == harap_gambar}")
    catat(f"  ada HTML mentah   : {'<span' in teks_md or '<div' in teks_md}")
    catat(f"  jumlah 'terdeteksi': "
          f"{sum(1 for c in teks_cap if 'terdeteksi' in c)}")
    catat(f"  subheader         : {[h.value for h in at.subheader][:5]}")
    catat(f"  semua field tampil: {len(field_tab) == len(det['fields'])}")

    teks_semua = " ".join(
        str(getattr(el, "value", ""))
        for jenis_el in ("markdown", "caption", "subheader", "header", "code")
        for el in getattr(at, jenis_el, [])
    )
    rusak = "\u00e2\u20ac\u201d" in teks_semua or "\u00c3\u2014" in teks_semua
    catat(f"  ada mojibake      : {rusak}")

    assert len(at.tabs) == 7, f"tabs = {len(at.tabs)}"
    assert not at.expander, "halaman tidak boleh memakai expander"
    assert n_gambar == harap_gambar, (n_gambar, harap_gambar)
    assert len(field_tab) == len(det["fields"]), (
        len(field_tab), len(det["fields"])
    )
    assert not rusak, "teks halaman masih mengandung mojibake"

LOG.write_text("\n".join(baris), encoding="utf-8")
