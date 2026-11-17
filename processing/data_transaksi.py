"""
Data transaksi hasil halaman Hasil Ekstraksi.

Modul ini menangani data teks hasil koreksi pengguna saja:
- pembacaan angka format Indonesia ("9.000", "10.550", "9.000,00"),
- validasi field wajib dan nilai angka,
- penyimpanan lokal memakai SQLite,
- pembuatan berkas CSV untuk diunduh.

Hasil OCR mentah TIDAK diubah di sini. Pemanggil menyimpan hasil OCR asli
secara terpisah dan hanya mengirimkan nilai yang sudah dikoreksi serta
dikonfirmasi pengguna.
"""

import csv
import hashlib
import io
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent

# Berkas basis data lokal. Dibuat otomatis saat transaksi pertama disimpan.
DB_BAKU = BASE / "database" / "transaksi.db"

KOLOM_CSV = [
    "Nomor Nota", "Nama Penerima", "Proyek",
    "Nama Produk", "Jumlah", "Harga Satuan",
]

# Field teks yang wajib diisi pada halaman Hasil Ekstraksi.
FIELD_WAJIB = (
    ("nomor", "Nomor Nota"),
    ("penerima", "Nama Penerima"),
    ("proyek", "Proyek"),
)

SKEMA = """
CREATE TABLE IF NOT EXISTS transaksi (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sidik         TEXT    NOT NULL UNIQUE,
    nama_file     TEXT    NOT NULL,
    nomor_nota    TEXT    NOT NULL,
    nama_penerima TEXT    NOT NULL,
    proyek        TEXT    NOT NULL,
    dibuat_pada   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS transaksi_produk (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    transaksi_id INTEGER NOT NULL
                 REFERENCES transaksi(id) ON DELETE CASCADE,
    baris        INTEGER NOT NULL,
    nama_produk  TEXT    NOT NULL,
    jumlah       REAL    NOT NULL,
    harga_satuan REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_produk_transaksi
    ON transaksi_produk(transaksi_id);
"""


def parse_angka_indonesia(teks):
    """
    Ubah teks angka nota menjadi float, atau None bila tidak valid.

    Aturan yang dipakai (format angka pada nota):
    - Pemisah yang membentuk kelompok tiga digit adalah pemisah ribuan,
      sehingga "9.000" -> 9000, "10.550" -> 10550, "1.250.000" -> 1250000,
      dan "3,800" -> 3800.
    - Pemisah terakhir yang diikuti 1-2 digit adalah tanda desimal,
      sehingga "3,5" -> 3.5, "65,00" -> 65.0, "1.234,50" -> 1234.5.
    - Awalan "Rp" dan akhiran ",-" diabaikan.
    - Karakter lain membuat nilai dianggap tidak valid, misalnya
      "9.000P00", "200a000", atau "abc" -> None.

    Nilai None dikembalikan agar pemanggil dapat meminta pengguna
    memperbaiki isian sebelum data disimpan.
    """
    if teks is None:
        return None

    s = str(teks).strip().lower()
    s = re.sub(r"^rp\.?\s*", "", s)
    s = re.sub(r"[.,\-]+$", "", s)
    s = s.replace(" ", "")

    # Tanda minus dipertahankan agar validasi dapat membedakan angka
    # negatif (mis. "-1.000") dari isian yang benar-benar tidak valid.
    tanda = 1.0
    if s[:1] in ("-", "\u2212"):
        tanda = -1.0
        s = s[1:]
    elif s[:1] == "+":
        s = s[1:]

    # Hanya angka dan pemisah yang diterima; huruf membuat nilai gugur.
    if not s or not re.fullmatch(r"[0-9][0-9.,]*", s):
        return None

    if not re.search(r"[.,]", s):
        return tanda * float(s)

    pos = max(s.rfind("."), s.rfind(","))
    kepala, ekor = s[:pos], s[pos + 1:]
    if not kepala or not ekor.isdigit():
        return None

    if len(ekor) == 3:
        bagian = re.split(r"[.,]", s)
        if bagian[0].isdigit() and all(len(b) == 3 for b in bagian[1:]):
            return tanda * float("".join(bagian))
        return None

    if len(ekor) in (1, 2):
        depan = re.sub(r"[.,]", "", kepala)
        if depan.isdigit():
            return tanda * float(depan + "." + ekor)
        return None

    return None


def format_angka_indonesia(nilai):
    """Tampilkan angka dengan pemisah ribuan titik dan desimal koma."""
    n = float(nilai)
    if n.is_integer():
        return f"{int(round(n)):,}".replace(",", ".")
    teks = f"{n:,.2f}"
    return teks.replace(",", "X").replace(".", ",").replace("X", ".")


def validasi_transaksi(data):
    """
    Periksa data hasil koreksi pengguna sebelum disimpan.

    Parameter
    ---------
    data : dict
        Berisi "nomor", "penerima", "proyek", dan "produk" (daftar dict
        dengan kunci "nama", "jumlah", "harga" berupa teks apa adanya).

    Mengembalikan (bersih, galat):
      bersih : dict siap simpan; jumlah dan harga sudah berupa float
      galat  : daftar pesan kesalahan; bila tidak kosong data tidak
               boleh disimpan.
    """
    data = data or {}
    galat = []

    bersih = {}
    for kunci, label in FIELD_WAJIB:
        nilai = str(data.get(kunci) or "").strip()
        if not nilai:
            galat.append(f"{label} wajib diisi.")
        bersih[kunci] = nilai

    produk = []
    for nomor, baris in enumerate(data.get("produk") or [], 1):
        baris = baris or {}
        nama = str(baris.get("nama") or "").strip()
        teks_jumlah = str(baris.get("jumlah") or "").strip()
        teks_harga = str(baris.get("harga") or "").strip()

        # Baris tambahan yang masih kosong (dari data editor) dilewati.
        if not (nama or teks_jumlah or teks_harga):
            continue

        if not nama:
            galat.append(f"Nama Produk baris {nomor} wajib diisi.")

        jumlah = parse_angka_indonesia(teks_jumlah)
        if jumlah is None:
            galat.append(
                f'Jumlah baris {nomor} tidak valid: "{teks_jumlah}". '
                "Gunakan angka, contoh 400 atau 9.000."
            )
        elif jumlah <= 0:
            galat.append(
                f"Jumlah baris {nomor} harus berupa angka positif "
                f"(terbaca {format_angka_indonesia(jumlah)})."
            )

        harga = parse_angka_indonesia(teks_harga)
        if harga is None:
            galat.append(
                f'Harga Satuan baris {nomor} tidak valid: "{teks_harga}". '
                "Gunakan angka, contoh 10.550."
            )
        elif harga < 0:
            galat.append(
                f"Harga Satuan baris {nomor} tidak boleh negatif "
                f"(terbaca {format_angka_indonesia(harga)})."
            )

        produk.append({"nama": nama, "jumlah": jumlah, "harga": harga})

    if not produk:
        galat.append("Minimal satu produk harus diisi.")

    bersih["produk"] = produk
    return bersih, galat


def _angka_canon(nilai):
    # Bentuk baku angka untuk sidik jari, agar 9000 dan 9000.0 sama.
    if nilai is None:
        return None
    return f"{float(nilai):.4f}"


def sidik_transaksi(data, nama_file=""):
    """
    Hitung sidik jari isi data untuk mencegah penyimpanan ganda.

    Sidik dihitung dari nota asal dan nilai yang sudah dikoreksi, sehingga
    klik berulang atau rerun Streamlit dengan isi yang sama tidak membuat
    baris baru di basis data.
    """
    data = data or {}
    inti = {
        "nama_file": str(nama_file or ""),
        "nomor": str(data.get("nomor") or ""),
        "penerima": str(data.get("penerima") or ""),
        "proyek": str(data.get("proyek") or ""),
        "produk": [
            {
                "nama": str(p.get("nama") or ""),
                "jumlah": _angka_canon(p.get("jumlah")),
                "harga": _angka_canon(p.get("harga")),
            }
            for p in (data.get("produk") or [])
        ],
    }
    teks = json.dumps(inti, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(teks.encode("utf-8")).hexdigest()


def _koneksi(db_path=None):
    path = Path(db_path) if db_path else DB_BAKU
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def siapkan_database(db_path=None):
    """Buat tabel transaksi bila belum ada. Mengembalikan path basis data."""
    conn = _koneksi(db_path)
    try:
        with conn:
            conn.executescript(SKEMA)
    finally:
        conn.close()
    return Path(db_path) if db_path else DB_BAKU


def simpan_transaksi(data, nama_file="", db_path=None):
    """
    Simpan satu transaksi beserta detail produknya.

    `data` harus sudah lolos validasi_transaksi (jumlah dan harga berupa
    float). Bila isi yang sama pernah disimpan untuk nota yang sama, tidak
    ada baris baru yang dibuat.

    Mengembalikan (id_transaksi, baru); baru=False berarti data sudah ada
    sebelumnya sehingga tidak disimpan ulang.
    """
    siapkan_database(db_path)
    sidik = sidik_transaksi(data, nama_file)
    dibuat = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = _koneksi(db_path)
    try:
        with conn:
            # Periksa lebih dulu agar klik berulang tidak memakai nomor id
            # baru. INSERT OR IGNORE tetap dipakai sebagai pengaman bila
            # ada dua penyimpanan hampir bersamaan.
            lama = conn.execute(
                "SELECT id FROM transaksi WHERE sidik = ?", (sidik,)
            ).fetchone()
            if lama is not None:
                return int(lama["id"]), False

            cur = conn.execute(
                "INSERT OR IGNORE INTO transaksi "
                "(sidik, nama_file, nomor_nota, nama_penerima, proyek, "
                "dibuat_pada) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sidik, str(nama_file or ""), data["nomor"],
                    data["penerima"], data["proyek"], dibuat,
                ),
            )

            if cur.rowcount == 0:
                lama = conn.execute(
                    "SELECT id FROM transaksi WHERE sidik = ?", (sidik,)
                ).fetchone()
                return int(lama["id"]), False

            id_transaksi = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO transaksi_produk "
                "(transaksi_id, baris, nama_produk, jumlah, harga_satuan) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        id_transaksi, urutan, p["nama"],
                        float(p["jumlah"]), float(p["harga"]),
                    )
                    for urutan, p in enumerate(data["produk"], 1)
                ],
            )
            return id_transaksi, True
    finally:
        conn.close()


def _angka_csv(nilai):
    n = float(nilai)
    return str(int(round(n))) if n.is_integer() else f"{n:.2f}"


def buat_csv(data):
    """
    Susun CSV dari data yang sudah dikoreksi dan dikonfirmasi.

    Satu produk menjadi satu baris. Angka ditulis tanpa pemisah ribuan
    agar langsung dapat diolah program lain. Hasil OCR mentah tidak ikut
    diunduh karena belum tentu benar.
    """
    data = data or {}
    buf = io.StringIO()
    tulis = csv.writer(buf)
    tulis.writerow(KOLOM_CSV)

    nomor = str(data.get("nomor") or "")
    penerima = str(data.get("penerima") or "")
    proyek = str(data.get("proyek") or "")

    for p in (data.get("produk") or []):
        tulis.writerow([
            nomor, penerima, proyek, str(p.get("nama") or ""),
            _angka_csv(p.get("jumlah")), _angka_csv(p.get("harga")),
        ])

    # utf-8-sig agar huruf/karakter khusus terbaca benar di Excel.
    return buf.getvalue().encode("utf-8-sig")


