"""Uji halaman Hasil Ekstraksi: tampilan, koreksi, validasi, simpan, unduh.

Yang diperiksa:
- halaman memakai hasil OCR yang sudah ada (tidak menjalankan OCR ulang),
- hasil OCR asli tidak berubah setelah halaman dibuka atau disimpan,
- isian awal sama dengan hasil OCR dan dapat dikoreksi,
- angka format Indonesia terbaca, angka rusak ditolak,
- penyimpanan SQLite tidak membuat data ganda saat tombol diklik ulang,
- CSV berisi satu baris per produk dengan angka tanpa pemisah ribuan,
- halaman Evaluasi tetap memakai hasil OCR, bukan hasil koreksi.

Basis data uji memakai berkas sementara agar data asli tidak tercemar.
"""

import csv
import io
import sqlite3
import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
import processing.data_transaksi as dt  # noqa: E402
from processing.ground_truth import gt_field  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402

APP = BASE / "app.py"
LOG = BASE / "output" / "uji_hasil_ekstraksi.txt"
baris_log = []


def catat(t):
    baris_log.append(str(t))
    print(t, flush=True)


def state(at, kunci, bawaan=None):
    """Ambil session_state AppTest tanpa melempar KeyError bila belum ada."""
    try:
        return at.session_state[kunci]
    except Exception:
        return bawaan


def cek_db(path):
    """Baca isi basis data uji: (jumlah transaksi, jumlah produk, produk)."""
    conn = sqlite3.connect(str(path))
    try:
        n_transaksi = conn.execute("SELECT COUNT(*) FROM transaksi").fetchone()[0]
        n_produk = conn.execute(
            "SELECT COUNT(*) FROM transaksi_produk"
        ).fetchone()[0]
        produk = conn.execute(
            "SELECT baris, nama_produk, jumlah, harga_satuan "
            "FROM transaksi_produk ORDER BY baris"
        ).fetchall()
    finally:
        conn.close()
    return n_transaksi, n_produk, produk


def caption_gambar(at):
    """Kumpulkan caption st.image (tersimpan pada proto, bukan st.caption)."""
    hasil = []

    def jalan(node):
        for anak in list(getattr(node, "children", {}).values()):
            proto = getattr(anak, "proto", None)
            if type(proto).__name__ == "ImageList":
                for img in proto.imgs:
                    hasil.append(img.caption)
            jalan(anak)

    jalan(at.main)
    return hasil


def uji_angka():
    kasus = {
        "9.000": 9000.0,
        "10.550": 10550.0,
        "1.250.000": 1250000.0,
        "9.000,00": 9000.0,
        "400": 400.0,
        "3,5": 3.5,
        "65,00": 65.0,
        "2.701,00": 2701.0,
        "9.000,-": 9000.0,
        "Rp 10.550": 10550.0,
        "9.000P00": None,
        "200a000": None,
        "abc": None,
        "": None,
        "1.2345": None,
    }
    for teks, harap in kasus.items():
        hasil = dt.parse_angka_indonesia(teks)
        assert hasil == harap, (teks, hasil, harap)
    catat(f"angka      : {len(kasus)} kasus sesuai")
    assert dt.format_angka_indonesia(9000) == "9.000"
    assert dt.format_angka_indonesia(1250000.5) == "1.250.000,50"


def uji_validasi():
    # Field wajib kosong.
    bersih, galat = dt.validasi_transaksi(
        {"nomor": " ", "penerima": "", "proyek": "", "produk": []}
    )
    assert len(galat) >= 4, galat
    catat(f"validasi   : {len(galat)} kesalahan untuk data kosong")

    # Jumlah nol dan harga negatif.
    bersih, galat = dt.validasi_transaksi({
        "nomor": "0299", "penerima": "Bpk", "proyek": "Proyek",
        "produk": [{"nama": "Produk", "jumlah": "0", "harga": "-1.000"}],
    })
    assert any("positif" in g for g in galat), galat
    assert any("negatif" in g for g in galat), galat
    catat(f"validasi   : nilai tidak wajar ditolak ({len(galat)} pesan)")

    # Angka rusak seperti hasil OCR.
    bersih, galat = dt.validasi_transaksi({
        "nomor": "0299", "penerima": "Bpk", "proyek": "Proyek",
        "produk": [{"nama": "Produk", "jumlah": "9.000P00", "harga": "10.550"}],
    })
    assert any("9.000P00" in g for g in galat), galat
    catat("validasi   : angka '9.000P00' ditolak")

    # Data benar: baris kosong tambahan diabaikan.
    bersih, galat = dt.validasi_transaksi({
        "nomor": "0299", "penerima": "Bpk Indra", "proyek": "KSO Adhi",
        "produk": [
            {"nama": "Konstein", "jumlah": "400", "harga": "35,696"},
            {"nama": "", "jumlah": "", "harga": ""},
        ],
    })
    assert not galat, galat
    assert len(bersih["produk"]) == 1, bersih
    assert bersih["produk"][0]["jumlah"] == 400.0
    assert bersih["produk"][0]["harga"] == 35696.0
    catat("validasi   : data benar diterima, baris kosong dilewati")
    return bersih


def uji_simpan_csv(bersih):
    tmp = Path(tempfile.mkdtemp()) / "uji.db"

    id1, baru1 = dt.simpan_transaksi(bersih, "nota_52.jpeg", tmp)
    id2, baru2 = dt.simpan_transaksi(bersih, "nota_52.jpeg", tmp)
    assert baru1 and not baru2, (baru1, baru2)
    assert id1 == id2, (id1, id2)

    conn = sqlite3.connect(str(tmp))
    try:
        n_transaksi = conn.execute(
            "SELECT COUNT(*) FROM transaksi"
        ).fetchone()[0]
        n_produk = conn.execute(
            "SELECT COUNT(*) FROM transaksi_produk"
        ).fetchone()[0]
        isi = conn.execute(
            "SELECT nomor_nota, nama_penerima, proyek FROM transaksi"
        ).fetchone()
    finally:
        conn.close()

    assert n_transaksi == 1, n_transaksi
    assert n_produk == 1, n_produk
    assert isi == ("0299", "Bpk Indra", "KSO Adhi"), isi
    catat(f"simpan     : 2x simpan -> {n_transaksi} transaksi, {n_produk} produk")

    dua_produk = dict(bersih)
    dua_produk["produk"] = bersih["produk"] + [
        {"nama": "Konstein Jepit", "jumlah": 3800.0, "harga": 20240.0}
    ]
    teks = dt.buat_csv(dua_produk).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(teks)))
    assert rows[0] == dt.KOLOM_CSV, rows[0]
    assert len(rows) == 3, rows
    assert rows[1][3:] == ["Konstein", "400", "35696"], rows[1]
    assert rows[2][3:] == ["Konstein Jepit", "3800", "20240"], rows[2]
    catat(f"csv        : {len(rows) - 1} baris produk, angka tanpa pemisah ribuan")



def uji_halaman(nama):
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

    # Salinan hasil OCR untuk memastikan tidak berubah.
    ocr_sebelum = {
        fid: {"tanpa": dict((h.get("tanpa") or {})),
              "clahe": dict((h.get("clahe") or {}))}
        for fid, h in ocr.items()
    }

    tmp_db = Path(tempfile.mkdtemp()) / "uji.db"
    db_lama = dt.DB_BAKU
    dt.DB_BAKU = tmp_db
    try:
        at = AppTest.from_file(str(APP), default_timeout=900)
        at.run()
        at.session_state["gambar_bgr"] = gambar
        at.session_state["nama_file"] = nama
        at.session_state["deteksi"] = det
        at.session_state["ocr_semua"] = ocr
        at.session_state["ocr_kunci"] = (nama,)
        at.sidebar.radio[0].set_value("Hasil Ekstraksi").run()

        judul = [h.value for h in at.header]
        subjudul = [h.value for h in at.subheader]
        caption = [c.value for c in at.caption]
        isian = {t.label: t.value for t in at.text_input}
        tombol = {b.label: b for b in at.button}
        sukses = [s.value for s in at.success]

        catat(f"=== {nama} ({len(det['fields'])} field) ===")
        catat(f"  exception   : {[e.value for e in at.exception]}")
        catat(f"  judul       : {judul}")
        catat(f"  subjudul    : {subjudul}")
        catat(f"  isian       : {isian}")
        catat(f"  tombol      : {list(tombol)}")

        assert not at.exception, [e.value for e in at.exception]
        assert judul and judul[0] == "Hasil Pembacaan Nota", judul
        assert "Periksa dan koreksi informasi transaksi sebelum disimpan." \
            in caption, caption
        # Status menyatakan OCR selesai, bukan hasilnya benar.
        assert sukses, sukses
        assert "belum tentu benar" in sukses[0], sukses[0]
        assert "Pratinjau Nota" in subjudul
        assert "Informasi Transaksi" in subjudul
        assert "Produk" in subjudul

        # Pratinjau harus memakai hasil crop nota, bukan foto unggahan asli.
        cap_gambar = caption_gambar(at)
        assert len(cap_gambar) == 1, cap_gambar
        assert cap_gambar[0].startswith("Hasil crop kertas nota"), cap_gambar
        assert "Gambar nota asli" not in cap_gambar[0], cap_gambar
        catat(f"  pratinjau   : {cap_gambar[0]}")

        # Tiga isian teks terisi nilai OCR apa adanya.
        for label, fid in (("Nomor Nota", "nomor"),
                           ("Nama Penerima", "penerima"),
                           ("Proyek", "proyek")):
            harap = (ocr.get(fid, {}).get("tanpa", {}).get("teks") or "").strip()
            assert isian.get(label) == harap, (label, isian.get(label), harap)
        catat("  isian awal  : sama dengan hasil OCR")

        # Belum ada data tersimpan: unduh belum aktif.
        assert "Simpan Data" in tombol, list(tombol)
        assert not state(at, "ekstraksi_tersimpan")

        # Hasil OCR asli tidak boleh berubah karena halaman ini.
        assert ocr == ocr_sebelum, "hasil OCR berubah"
        catat("  OCR asli    : tidak berubah setelah halaman dibuka")

        # Klik Simpan Data memakai nilai isian apa adanya.
        tombol["Simpan Data"].click().run()
        galat = [e.value for e in at.error]
        sukses2 = [s.value for s in at.success]
        info2 = [i.value for i in at.info]
        tersimpan = state(at, "ekstraksi_tersimpan")

        catat(f"  setelah simpan: error={len(galat)} sukses={sukses2[:1]}")
        assert not at.exception, [e.value for e in at.exception]

        if galat:
            # Nota dengan angka OCR rusak: data ditolak, tidak tersimpan.
            assert tersimpan is None, tersimpan
            n = cek_db(tmp_db)[0] if tmp_db.exists() else 0
            assert n == 0, n
            catat(f"  validasi    : ditolak -> {galat[:2]}")

            # Koreksi manual pada sel tabel harus bisa menyelamatkan data.
            editan = {}
            produk_field = [f for f in det["fields"] if f.get("jenis") == "produk"]
            for idx, f in enumerate(produk_field):
                urutan = str(f["id"]).split("_")[-1]
                for kolom, fid, cadangan in (
                    ("Jumlah", f"jumlah_{urutan}", "1"),
                    ("Harga Satuan", f"harga_{urutan}", "1.000"),
                ):
                    teks = ((ocr.get(fid, {}) or {}).get("tanpa", {})
                            or {}).get("teks", "") or ""
                    if dt.parse_angka_indonesia(teks) is None:
                        editan.setdefault(idx, {})[kolom] = (
                            gt_field(nama, fid) or cadangan
                        )

            at.session_state[f"ekstraksi_{nama}_produk"] = {
                "edited_rows": editan, "added_rows": [], "deleted_rows": [],
            }
            at.button(key=f"ekstraksi_{nama}_simpan").click().run()

            galat2 = [e.value for e in at.error]
            sukses2 = [s.value for s in at.success]
            tersimpan = state(at, "ekstraksi_tersimpan")
            catat(f"  koreksi     : {editan} -> error={len(galat2)}")
            assert not at.exception, [e.value for e in at.exception]
            assert not galat2, galat2
            assert tersimpan, "data hasil koreksi tidak tersimpan"
            assert any("berhasil disimpan" in s for s in sukses2), sukses2

            n_transaksi, n_produk, isi_produk = cek_db(tmp_db)
            assert n_transaksi == 1, n_transaksi
            assert n_produk == len(tersimpan["data"]["produk"]), n_produk
            for baris, _nama_p, _jumlah, _harga in isi_produk:
                nilai = tersimpan["data"]["produk"][baris - 1]
                assert _jumlah == nilai["jumlah"], (_jumlah, nilai)
                assert _harga == nilai["harga"], (_harga, nilai)
            catat(f"  simpan      : 1 transaksi, {n_produk} produk hasil koreksi")

            # Klik ulang tidak boleh menambah baris (cegah simpan ganda).
            at.button(key=f"ekstraksi_{nama}_simpan").click().run()
            assert cek_db(tmp_db)[0] == 1, cek_db(tmp_db)[0]
            assert any(
                "sudah tersimpan" in w.value for w in at.warning
            ), [w.value for w in at.warning]
            catat("  simpan ulang: tetap 1 transaksi")
        else:
            assert tersimpan, "data tidak tersimpan"
            assert any("berhasil disimpan" in s for s in sukses2), sukses2
            n_transaksi, n_produk, _ = cek_db(tmp_db)
            assert n_transaksi == 1, n_transaksi
            assert n_produk == len(tersimpan["data"]["produk"]), n_produk
            catat(f"  simpan      : 1 transaksi, {n_produk} produk")

            # Status konfirmasi tampil dan tombol unduh sudah aktif.
            assert any("Data terkonfirmasi" in i for i in info2), info2
            catat("  konfirmasi  : data terkonfirmasi, unduh aktif")

            # Klik ulang tidak boleh menambah baris (cegah simpan ganda).
            at.button(key=f"ekstraksi_{nama}_simpan").click().run()
            assert cek_db(tmp_db)[0] == 1, cek_db(tmp_db)[0]
            assert any(
                "sudah tersimpan" in w.value for w in at.warning
            ), [w.value for w in at.warning]
            catat("  simpan ulang: tetap 1 transaksi")

            # Evaluasi tetap memakai hasil OCR, bukan hasil koreksi.
            at.sidebar.radio[0].set_value("Evaluasi").run()
            tabel = at.dataframe[0].value
            kolom = list(tabel.columns)
            assert kolom == [
                "Field", "Ground Truth", "Hasil OCR", "Status"
            ], kolom
            ocr_teks = {
                f["label"]: ((ocr.get(f["id"], {}) or {}).get("tanpa", {})
                             or {}).get("teks", "") or ""
                for f in det["fields"]
            }
            for b in tabel.to_dict("records"):
                if b["Field"] in ocr_teks:
                    harap = ocr_teks[b["Field"]] or "(kosong)"
                    assert b["Hasil OCR"] == harap, (
                        b["Field"], b["Hasil OCR"]
                    )
            catat("  evaluasi    : tetap memakai hasil OCR")

        assert ocr == ocr_sebelum, "hasil OCR berubah setelah simpan"
        return nama
    finally:
        dt.DB_BAKU = db_lama


def uji_tanpa_ocr():
    """Halaman tidak boleh menjalankan OCR sendiri saat hasil OCR belum ada."""
    path = BASE / "data_uji" / "nota_52.jpeg"
    gambar = baca_gambar(path)

    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    at.session_state["gambar_bgr"] = gambar
    at.session_state["nama_file"] = "nota_52.jpeg"
    at.session_state["deteksi"] = appmod.jalankan_deteksi(gambar)
    at.session_state["ocr_semua"] = None
    at.session_state["ocr_kunci"] = None
    at.sidebar.radio[0].set_value("Hasil Ekstraksi").run()

    info = [i.value for i in at.info]
    assert not at.exception, [e.value for e in at.exception]
    assert any("Hasil OCR belum tersedia" in i for i in info), info
    assert not at.text_input, "form tidak boleh tampil tanpa hasil OCR"
    assert state(at, "ocr_semua") is None, "halaman menjalankan OCR sendiri"
    catat("tanpa OCR  : halaman meminta buka Pengenalan Karakter, OCR tidak dijalankan")


def main():
    uji_angka()
    bersih = uji_validasi()
    uji_simpan_csv(bersih)
    uji_tanpa_ocr()

    for nama in ("nota_52.jpeg", "nota_01.jpeg"):
        uji_halaman(nama)

    LOG.write_text("\n".join(baris_log), encoding="utf-8")


if __name__ == "__main__":
    main()

