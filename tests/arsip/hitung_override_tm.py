import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter

_spes = importlib.util.spec_from_file_location(
    "pengenalan_lama",
    str(BASE / "archive" / "pengenalan_karakter_sebelum_opsiB.py"),
)
mod_lama = importlib.util.module_from_spec(_spes)
_spes.loader.exec_module(mod_lama)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    total = 0
    diganti_tm = 0

    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)
            _, hasil = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            for s in hasil:
                total += 1
                final_lama, aturan = mod_lama.gabung_keputusan(
                    s.get("cnn", ""), float(s.get("pc", 0.0)),
                    s.get("tm", ""), float(s.get("pt", 0.0)),
                )
                if final_lama != s.get("cnn", ""):
                    diganti_tm += 1

    print(f"Total karakter = {total}")
    print(f"Karakter yang labelnya DIGANTI TM pada aturan lama = {diganti_tm}")
    print(f"Proporsi = {100.0 * diganti_tm / max(1, total):.2f}%")


if __name__ == "__main__":
    main()