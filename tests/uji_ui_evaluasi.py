"""Uji halaman Evaluasi: tanpa CER, kolom sesuai, status benar/salah akurat."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"

for nama in ("nota_52.jpeg", "nota_01.jpeg"):
    path = None
    for d in (BASE / "data_nota_asli", BASE / "data_uji"):
        if (d / nama).exists():
            path = d / nama
            break
    if path is None:
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
    at.sidebar.radio[0].set_value("Evaluasi").run()

    print(f"\n=== {nama} ===")
    print("exception :", [e.value for e in at.exception])

    tabel = at.dataframe[0].value
    kolom = list(tabel.columns)
    print("kolom     :", kolom)
    assert kolom == ["Field", "Ground Truth", "Hasil OCR", "Status"], kolom

    # Tidak boleh ada kolom/teks CER di mana pun pada halaman ini.
    teks_semua = []
    for jenis in ("markdown", "caption", "subheader", "header", "text",
                  "info", "latex"):
        for el in getattr(at, jenis, []):
            teks_semua.append(str(getattr(el, "value", "")))
    gabung = " ".join(teks_semua).upper()
    print("ada kata CER :", "CER" in gabung)
    assert "CER" not in gabung, "masih ada keterangan CER"
    assert not any("cer" in str(k).lower() for k in kolom), kolom

    print("metric    :", [(m.label, m.value) for m in at.metric])
    assert not any("cer" in m.label.lower() for m in at.metric)

    print("expander  :", len(at.expander))
    assert len(at.expander) == 0

    # Status harus konsisten dengan perbandingan teks.
    for b in tabel.to_dict("records"):
        gt = b["Ground Truth"]
        ocr_t = b["Hasil OCR"]
        if gt == "(tidak tersedia)":
            assert b["Status"] == "Tanpa GT", b
            continue
        if ocr_t == "(kosong)":
            ocr_t = ""
        sama = " ".join(gt.split()) == " ".join(ocr_t.split())
        harap = "Benar" if sama else "Salah"
        assert b["Status"] == harap, (b, harap)
    print(f"status    : {len(tabel)} baris konsisten")
