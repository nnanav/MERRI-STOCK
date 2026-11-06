from pathlib import Path

from streamlit.testing.v1 import AppTest

from processing.pemrosesan_field import baca_gambar

APP = Path(__file__).resolve().parent / "app.py"


def ambil_state(at, kunci):
    try:
        return at.session_state[kunci]
    except Exception:
        return None


def main():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()

    # Set nota uji langsung ke session (Upload kini khusus unggah berkas).
    at.session_state["gambar_bgr"] = baca_gambar(
        Path(__file__).resolve().parent / "data_uji" / "nota_52.jpeg"
    )
    at.session_state["nama_file"] = "nota_52.jpeg"
    at.session_state["deteksi"] = None
    at.session_state["ocr_semua"] = None
    at.session_state["ocr_kunci"] = None
    at.run()

    # Buka Upload lalu klik tombol "Proses Nota".
    at.sidebar.radio[0].set_value("Upload Nota").run()
    print("Upload exception:", [e.value for e in at.exception])
    tombol = [b for b in at.button if b.key == "proses_nota"]
    print("Tombol 'Proses Nota' ada:", bool(tombol))
    if tombol:
        tombol[0].click().run()
    print("Setelah Proses exception:", [e.value for e in at.exception])

    det = ambil_state(at, "deteksi")
    print("Jumlah field terdeteksi:", len(det["fields"]) if det else None)

    at.sidebar.radio[0].set_value("Segmentasi").run()
    print("Segmentasi exception:", [e.value for e in at.exception])

    at.sidebar.radio[0].set_value("Pengenalan").run()
    print("Pengenalan exception:", [e.value for e in at.exception])
    ocr = ambil_state(at, "ocr_semua")
    if ocr:
        for fid, h in ocr.items():
            t = h.get("tanpa", {})
            print(f"   {fid:12s} status={t.get('status')!s:8s} CNN={t.get('cnn')!r} TM={t.get('tm')!r}")
    else:
        print("ocr_semua kosong")

    at.sidebar.radio[0].set_value("Evaluasi").run()
    print("Evaluasi exception:", [e.value for e in at.exception])


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()