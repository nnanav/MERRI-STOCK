import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg", "0086"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg", "0099"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg", "0299"),
]


def main():
    total = 0
    sama = 0
    beda = 0
    lebih_baik = 0
    lebih_buruk = 0

    for nama, path, gt in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue

            tanpa = app.kenali_crop(crop, pakai_clahe=False)
            clahe = app.kenali_crop(crop, pakai_clahe=True)
            total += 1

            if tanpa["teks"] == clahe["teks"]:
                sama += 1
                tanda = "sama"
            else:
                beda += 1
                tanda = "BEDA"

            print(f"  {f['label']:16s} [{tanda}]")
            print(f"     tanpa CLAHE : {tanpa['teks']!r} (n={tanpa['n']})")
            if tanda == "BEDA":
                print(f"     pakai CLAHE : {clahe['teks']!r} (n={clahe['n']})")

            if f["jenis"] == "nomor":
                c_tanpa = app.hitung_cer(gt, tanpa["teks"])
                c_clahe = app.hitung_cer(gt, clahe["teks"])
                print(f"     CER: tanpa={c_tanpa:.3f} clahe={c_clahe:.3f}")
                if c_clahe < c_tanpa:
                    lebih_baik += 1
                elif c_clahe > c_tanpa:
                    lebih_buruk += 1

    print(f"\nField diuji={total} | sama={sama} | beda={beda}")
    print(f"Nomor: CLAHE lebih baik={lebih_baik}, lebih buruk={lebih_buruk}")


if __name__ == "__main__":
    main()