import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_cnn, muat_templates
from processing.segmentasi_karakter import (
    segmentasi_karakter, segmentasi_karakter_tm, susun_teks,
)

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg", "0086"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg", "0099"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg", "0299"),
]
FIELD = ["nomor", "jumlah_1", "satuan_1", "harga_1", "produk_1"]


def teks_dari(hasil):
    if not hasil:
        return "", 0
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    return susun_teks(hasil, "cnn", ta), len(hasil)


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    for nama, path, gt in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for fid in FIELD:
            f = next((x for x in det["fields"] if x["id"] == fid), None)
            if f is None or f["crop_ocr"] is None:
                continue
            crop = app.ke_tigakanal(f["crop_ocr"])

            _, h_proj = segmentasi_karakter(
                crop, model, labels, templates, label_tm
            )
            t_proj, n_proj = teks_dari(h_proj)

            _, h_tm = segmentasi_karakter_tm(
                crop, model, labels, templates, label_tm
            )
            t_tm, n_tm = teks_dari(h_tm)

            cer_proj = app.hitung_cer(gt, t_proj) if fid == "nomor" else None
            cer_tm = app.hitung_cer(gt, t_tm) if fid == "nomor" else None

            print(f"  {fid:10s} proyeksi n={n_proj:3d} {t_proj!r}")
            print(f"  {'':10s} TM       n={n_tm:3d} {t_tm!r}")
            if cer_proj is not None:
                print(f"  {'':10s} CER: proyeksi={cer_proj:.3f} TM={cer_tm:.3f}")


if __name__ == "__main__":
    main()