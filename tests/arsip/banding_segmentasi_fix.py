import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import segmentasi_karakter, susun_teks

_spes = importlib.util.spec_from_file_location(
    "seg_lama",
    str(BASE / "archive" / "segmentasi_sebelum_perbaikan.py"),
)
mod_lama = importlib.util.module_from_spec(_spes)
_spes.loader.exec_module(mod_lama)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg", "0086"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg", "0099"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg", "0299"),
]


def ukur(hasil):
    if not hasil:
        return 0, 0.0, 0
    ta = float(np.median([s["y2"] - s["y1"] for s in hasil]))
    lebar = [(s["x2"] - s["x1"]) / ta for s in hasil]
    return len(hasil), max(lebar), sum(1 for z in lebar if z > 1.05)


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    ringkas = {"lama": [0, 0], "baru": [0, 0]}
    for nama, path, gt in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)

            _, h_lama = mod_lama.segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            _, h_baru = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )

            nl, ml, kl = ukur(h_lama)
            nb, mb, kb = ukur(h_baru)
            ringkas["lama"][0] += kl
            ringkas["lama"][1] += nl
            ringkas["baru"][0] += kb
            ringkas["baru"][1] += nb

            tanda = ""
            if kb < kl:
                tanda = "  << lebih baik"
            elif kb > kl:
                tanda = "  << lebih buruk"

            print(f"  {f['label']:16s} lama n={nl:3d} maks={ml:.2f} nempel={kl}"
                  f" | baru n={nb:3d} maks={mb:.2f} nempel={kb}{tanda}")

    print("\nTOTAL nempel: lama={} dari {} segmen | baru={} dari {} segmen".format(
        ringkas["lama"][0], ringkas["lama"][1],
        ringkas["baru"][0], ringkas["baru"][1]))


if __name__ == "__main__":
    main()