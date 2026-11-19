"""Uji halaman Evaluasi: Ground Truth diisi manual, tanpa CER, status akurat.

Yang diperiksa:
- tabel punya 4 kolom dan Ground Truth dapat diketik (data_editor),
- Ground Truth kosong saat halaman pertama dibuka (tidak diisi otomatis),
- status "Belum diisi" dan ringkasan 0 selama GT kosong,
- setelah GT diketik, Status dan ringkasan ikut berubah,
- tombol Kosongkan GT mengembalikan semuanya ke kosong,
- tidak ada kata CER di mana pun pada halaman ini,
- Hasil OCR tidak pernah berubah karena pengisian GT.
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"


def buka(nama):
    path = None
    for d in (BASE / "data_nota_asli", BASE / "data_uji"):
        if (d / nama).exists():
            path = d / nama
            break
    if path is None:
        return None

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
    return at, det, ocr


for nama in ("nota_52.jpeg", "nota_01.jpeg"):
    hasil_buka = buka(nama)
    if hasil_buka is None:
        continue
    at, det, ocr = hasil_buka

    print(f"\n=== {nama} ===")
    print("exception :", [e.value for e in at.exception])
    assert not at.exception, [e.value for e in at.exception]

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
    assert not any("cer" in m.label.lower() for m in at.metric)

    # Ground Truth harus kosong di awal: diisi sendiri oleh pengguna.
    gt_awal = [str(v) for v in tabel["Ground Truth"]]
    print("GT awal   :", gt_awal[:3], "...")
    assert all(not v.strip() for v in gt_awal), gt_awal

    status_awal = set(tabel["Status"])
    print("status    :", status_awal)
    assert status_awal == {"Belum diisi"}, status_awal

    metrik = {m.label: m.value for m in at.metric}
    print("metric    :", metrik)
    assert metrik["Field benar"] == "0", metrik
    assert metrik["Field salah"] == "0", metrik
    assert metrik["Persentase benar"] == "0.00%", metrik

    # Hasil OCR harus sama dengan keluaran pipeline.
    ocr_teks = {
        f["label"]: ((ocr.get(f["id"], {}) or {}).get("tanpa", {})
                     or {}).get("teks", "") or ""
        for f in det["fields"]
    }
    for b in tabel.to_dict("records"):
        if b["Field"] in ocr_teks:
            harap = ocr_teks[b["Field"]] or "(kosong)"
            assert b["Hasil OCR"] == harap, (b["Field"], b["Hasil OCR"])

    kunci = f"evaluasi_{nama}_tabel_v0"

    # Isi dua baris: satu disalin dari OCR (harus Benar), satu dibedakan.
    baris = tabel.to_dict("records")
    idx_benar = next(
        i for i, b in enumerate(baris) if b["Hasil OCR"] != "(kosong)"
    )
    idx_salah = next(
        (i for i, b in enumerate(baris)
         if i != idx_benar and b["Hasil OCR"] != "(kosong)"), None
    )
    editan = {idx_benar: {"Ground Truth": baris[idx_benar]["Hasil OCR"]}}
    if idx_salah is not None:
        editan[idx_salah] = {"Ground Truth": "acuan-salah"}

    at.session_state[kunci] = {
        "edited_rows": editan, "added_rows": [], "deleted_rows": [],
    }
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    baris2 = at.dataframe[0].value.to_dict("records")
    print("status 2  :", baris2[idx_benar]["Status"],
          baris2[idx_salah]["Status"] if idx_salah is not None else "-")
    assert baris2[idx_benar]["Status"] == "Benar", baris2[idx_benar]
    if idx_salah is not None:
        assert baris2[idx_salah]["Status"] == "Salah", baris2[idx_salah]

    metrik2 = {m.label: m.value for m in at.metric}
    print("metric 2  :", metrik2)
    harap_salah = 1 if idx_salah is not None else 0
    assert metrik2["Field benar"] == "1", metrik2
    assert metrik2["Field salah"] == str(harap_salah), metrik2
    harap_persen = 100.0 / (1 + harap_salah)
    assert metrik2["Persentase benar"] == f"{harap_persen:.2f}%", metrik2

    # Hasil OCR tidak boleh ikut berubah karena pengisian GT.
    for b, lama in zip(baris2, baris):
        assert b["Hasil OCR"] == lama["Hasil OCR"], (b, lama)

    # Tombol Kosongkan GT mengembalikan semua ke kosong.
    tombol = [b for b in at.button if b.label == "Kosongkan GT"]
    assert tombol, [b.label for b in at.button]
    assert not tombol[0].disabled, "tombol harus aktif saat ada isian"
    tombol[0].click().run()

    assert not at.exception, [e.value for e in at.exception]
    tabel3 = at.dataframe[0].value
    print("status 3  :", set(tabel3["Status"]))
    assert all(not str(v).strip() for v in tabel3["Ground Truth"]), \
        list(tabel3["Ground Truth"])
    assert set(tabel3["Status"]) == {"Belum diisi"}, set(tabel3["Status"])
    metrik3 = {m.label: m.value for m in at.metric}
    assert metrik3["Field benar"] == "0", metrik3
    print("kosongkan : kembali kosong, ringkasan 0")

