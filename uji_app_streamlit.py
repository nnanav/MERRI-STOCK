from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "app.py"


def jalankan(halaman):
    at = AppTest.from_file(str(APP), default_timeout=180)
    # Muat pertama (Preparation) lalu pindah ke halaman target.
    at.run()
    if len(at.exception):
        return at, "GAGAL-MUAT"
    at.sidebar.radio[0].set_value(halaman).run()
    return at, "ok"


def main():
    for halaman in (
        "Preparation", "Upload Nota", "Preprocessing",
        "Segmentasi", "Pengenalan", "Evaluasi",
    ):
        at, status = jalankan(halaman)
        galat = [e.value for e in at.exception]
        print(f"[{halaman}] status={status} exception={galat}")
        if galat:
            print("  detail:", galat[0][:400])


if __name__ == "__main__":
    main()