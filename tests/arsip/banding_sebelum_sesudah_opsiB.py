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

# Modul lama (sebelum Opsi B') untuk pembanding.
_spes = importlib.util.spec_from_file_location(
    "pengenalan_lama",
    str(BASE / "archive" / "pengenalan_karakter_sebelum_opsiB.py"),
)
mod_lama = importlib.util.module_from_spec(_spes)
_spes.loader.exec_module(mod_lama)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg", "0086"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg", "0099"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg", "0299"),
]


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    total = 0
    berubah = 0
    perbaikan = 0
    kerusakan = 0

    for nama, path, gt in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            img = app.ke_tigakanal(crop)
            biner, hasil = segmentasi_karakter(
                img, model, labels, templates, label_tm
            )
            tinggi = [s["y2"] - s["y1"] for s in hasil]
            ta = float(np.median(tinggi)) if tinggi else 12.0

            final_lama, _ = mod_lama.gabung_segmen(hasil)
            final_baru, _ = app_gabung(hasil)

            t_lama = susun_teks(final_lama, "final", ta)
            t_baru = susun_teks(final_baru, "final", ta)

            total += 1
            tanda = "sama"
            if t_lama != t_baru:
                berubah += 1
                tanda = "BERUBAH"
                if f["jenis"] == "nomor":
                    cl = app.hitung_cer(gt, t_lama)
                    cb = app.hitung_cer(gt, t_baru)
                    if cb < cl:
                        perbaikan += 1
                        tanda = "MEMBAIK"
                    elif cb > cl:
                        kerusakan += 1
                        tanda = "MEMBURUK"

            print(f"  {f['label']:16s} [{tanda}]")
            if t_lama != t_baru:
                print(f"     lama  : {t_lama!r}")
                print(f"     baru  : {t_baru!r}")

    print(f"\nField diuji={total} | berubah={berubah} "
          f"| membaik={perbaikan} | memburuk={kerusakan}")


def app_gabung(hasil):
    from processing.pengenalan_karakter import gabung_segmen
    return gabung_segmen(hasil)


if __name__ == "__main__":
    main()