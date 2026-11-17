import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from processing.pemrosesan_field import baca_gambar

APP = BASE / "app.py"


def ambil_state(at, kunci):
    try:
        return at.session_state[kunci]
    except Exception:
        return None


def main():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.session_state["gambar_bgr"] = baca_gambar(BASE / "data_uji" / "nota_26.jpeg")
    at.session_state["nama_file"] = "nota_26.jpeg"
    at.session_state["deteksi"] = None
    at.session_state["ocr_semua"] = None
    at.session_state["ocr_kunci"] = None
    at.run()

    at.sidebar.radio[0].set_value("Upload Nota").run()
    tombol = [b for b in at.button if b.key == "proses_nota"]
    if tombol:
        tombol[0].click().run()
    det = ambil_state(at, "deteksi")
    print("Setelah proses exception:", [e.value for e in at.exception])
    print("jumlah field:", len(det["fields"]) if det else None)

    at.sidebar.radio[0].set_value("Segmentasi Karakter").run()
    print("Segmentasi exception:", [e.value for e in at.exception])

    at.sidebar.radio[0].set_value("Pengenalan Karakter").run()
    print("Pengenalan exception:", [e.value for e in at.exception])

    at.sidebar.radio[0].set_value("Hasil Ekstraksi").run()
    print("Hasil Ekstraksi exception:", [e.value for e in at.exception])

    at.sidebar.radio[0].set_value("Evaluasi").run()
    print("Evaluasi exception:", [e.value for e in at.exception])


if __name__ == "__main__":
    main()