import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

APP = BASE / "app.py"


def jalankan(halaman):
    at = AppTest.from_file(str(APP), default_timeout=180)
    # Muat pertama lalu pindah ke halaman target.
    at.run()
    if len(at.exception):
        return at, "GAGAL-MUAT"
    at.sidebar.radio[0].set_value(halaman).run()
    return at, "ok"


def main():
    for halaman in (
        "Business Understanding", "Upload Nota", "Preprocessing",
        "Segmentasi Karakter", "Pengenalan Karakter", "Evaluasi",
    ):
        at, status = jalankan(halaman)
        galat = [e.value for e in at.exception]
        print(f"[{halaman}] status={status} exception={galat}")
        if galat:
            print("  detail:", galat[0][:400])


if __name__ == "__main__":
    main()