"""Pastikan setiap halaman benar-benar menampilkan konten (tidak blank)."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

APP = BASE / "app.py"

HALAMAN = [
    "Business Understanding",
    "Upload Nota",
    "Preprocessing",
    "Segmentasi Karakter",
    "Pengenalan Karakter",
    "Evaluasi",
]

gagal = 0
for halaman in HALAMAN:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(halaman).run()

    galat = [e.value for e in at.exception]
    n_header = len(at.header) + len(at.subheader)
    n_teks = len(at.markdown) + len(at.caption) + len(at.info) + len(at.text)

    # Halaman dianggap TIDAK blank bila ada header/subheader atau teks.
    blank = (n_header + n_teks) == 0
    status = "BLANK!" if blank else "ada isi"
    gagal += int(blank or bool(galat))

    print(f"[{halaman:22s}] header={n_header} teks={n_teks} "
          f"-> {status} exception={galat}")

print(f"\nHalaman blank/bermasalah: {gagal}")
