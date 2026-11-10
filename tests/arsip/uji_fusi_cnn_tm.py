import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import app
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import (
    muat_cnn, muat_templates, gabung_segmen,
)
from processing.segmentasi_karakter import segmentasi_karakter, susun_teks

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg", "0086"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg", "0099"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg", "0299"),
    ("nota_54", Path(r"D:\PROJECT MARRIENA\data_uji\nota_54.jpeg"), "0292"),
]


def kenali(img, model, labels, templates, label_tm):
    img = app.ke_tigakanal(img)
    biner, hasil = segmentasi_karakter(img, model, labels, templates, label_tm)
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    final, detail = gabung_segmen(hasil)
    return {
        "teks": susun_teks(final, "final", ta),
        "cnn": susun_teks(hasil, "cnn", ta),
        "tm": susun_teks(hasil, "tm", ta),
        "n": len(hasil),
        "detail": detail,
    }


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    for nama, path, gt in NOTA:
        if not path.exists():
            print(nama, "-> file tidak ada, dilewati")
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        field = next((f for f in det["fields"] if f["id"] == "nomor"), None)
        if field is None or field["crop_ocr"] is None:
            print(nama, "-> crop nomor tidak tersedia")
            continue

        r = kenali(field["crop_ocr"], model, labels, templates, label_tm)
        aturan = Counter(d["aturan"] for d in r["detail"])
        cer_f = app.hitung_cer(gt, r["teks"])
        cer_c = app.hitung_cer(gt, r["cnn"])
        cer_t = app.hitung_cer(gt, r["tm"])

        print(f"\n=== {nama} (GT {gt}) ===")
        print(f"  n_karakter = {r['n']}")
        print(f"  gabungan = {r['teks']!r}")
        print(f"  CNN saja = {r['cnn']!r}")
        print(f"  TM saja  = {r['tm']!r}")
        print(f"  aturan   = {dict(aturan)}")
        print(f"  CER gabungan={cer_f:.3f} | CNN={cer_c:.3f} | TM={cer_t:.3f}")
        # rincian tiap karakter
        for i, d in enumerate(r["detail"], 1):
            print(f"    chr{i}: cnn={d['cnn']!r}({d['pc']}) tm={d['tm']!r}({d['pt']}) "
                  f"-> {d['final']!r} [{d['aturan']}]")

        # field lain (bukan nomor) untuk melihat keragaman aturan keputusan
        for fid in ("produk_1", "jumlah_1", "satuan_1"):
            f2 = next((x for x in det["fields"] if x["id"] == fid), None)
            if f2 is None or f2["crop_ocr"] is None:
                continue
            r2 = kenali(f2["crop_ocr"], model, labels, templates, label_tm)
            at2 = Counter(d["aturan"] for d in r2["detail"])
            print(f"  [{fid}] n={r2['n']} teks={r2['teks']!r} aturan={dict(at2)}")


if __name__ == "__main__":
    main()