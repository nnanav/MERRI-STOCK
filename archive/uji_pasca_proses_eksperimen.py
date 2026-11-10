import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar
from processing.pasca_proses import rapikan_field

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]

# Ground truth sebagian field untuk mengukur dampak (hanya untuk evaluasi).
GT = {
    ("nota_01", "jumlah_1"): "65,00",
    ("nota_01", "jumlah_2"): "2.701,00",
    ("nota_01", "harga_1"): "39.000",
    ("nota_01", "harga_2"): "139.500",
    ("nota_01", "harga_3"): "200.000",
    ("nota_01", "satuan_1"): "Pcs",
    ("nota_26", "jumlah_1"): "9.000,00",
    ("nota_26", "harga_1"): "10.550",
    ("nota_26", "satuan_1"): "Pcs",
    ("nota_52", "jumlah_1"): "3.400",
    ("nota_52", "harga_1"): "35,696",
    ("nota_52", "satuan_1"): "Pcs",
}


def main():
    total = 0
    sebelum_benar = 0
    sesudah_benar = 0
    membaik = 0
    memburuk = 0

    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n===== {nama} =====")
        for f in det["fields"]:
            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                continue
            if f["jenis"] not in ("nomor", "jumlah", "harga", "satuan"):
                continue

            asli = app.kenali_crop(crop, pakai_clahe=False)["teks"]
            rapi = rapikan_field(asli, f["jenis"])

            gt = GT.get((nama, f["id"]))
            tanda = ""
            if gt:
                total += 1
                b = asli == gt
                s = rapi == gt
                sebelum_benar += int(b)
                sesudah_benar += int(s)
                if s and not b:
                    membaik += 1
                    tanda = "  << MEMBAIK"
                elif b and not s:
                    memburuk += 1
                    tanda = "  << MEMBURUK"
                elif not b and not s:
                    tanda = "  (masih salah)"

            print(f"  {f['label']:16s} asli={asli!r:22s} rapi={rapi!r:18s}{tanda}")

    print(f"\nField ber-GT = {total}")
    print(f"Benar sebelum pasca-proses = {sebelum_benar}")
    print(f"Benar sesudah pasca-proses = {sesudah_benar}")
    print(f"Membaik = {membaik} | Memburuk = {memburuk}")


if __name__ == "__main__":
    main()