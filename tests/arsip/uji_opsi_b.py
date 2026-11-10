import sys
from collections import Counter
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
    beda = 0
    for nama, path, gt in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            hasil = app.kenali_crop(crop, pakai_clahe=False)
            detail = hasil.get("detail", [])
            aturan = Counter(d.get("aturan") for d in detail)
            dukung = Counter(d.get("dukungan_tm") for d in detail)

            salah = [
                d for d in detail
                if d.get("final") != d.get("cnn")
            ]
            total += len(detail)
            beda += len(salah)

            print(f"  {f['label']:16s} n={hasil['n']:3d} teks={hasil['teks']!r}")
            print(f"  {'':16s} aturan={dict(aturan)} dukungan_tm={dict(dukung)}")
            if salah:
                print(f"  {'':16s} !!! final != cnn sebanyak {len(salah)}")

            if f["jenis"] == "nomor":
                cer = app.hitung_cer(gt, hasil["teks"])
                print(f"  {'':16s} CER nomor = {cer:.3f} (GT {gt})")

    print(f"\nTOTAL karakter={total}, final!=cnn = {beda}")


if __name__ == "__main__":
    main()