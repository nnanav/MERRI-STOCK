import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar, simpan_gambar

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]
OUT = BASE / "output" / "cek_field"
OUT.mkdir(parents=True, exist_ok=True)


def ocr(crop):
    if crop is None or not getattr(crop, "size", 0):
        return "-", "-"
    hasil = app.kenali_crop(crop, pakai_clahe=False)
    return hasil.get("teks", ""), hasil.get("n", 0)


def main():
    for nama, path in NOTA:
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n================ {nama} ================")
        for f in det["fields"]:
            if f["jenis"] in ("nomor", "penerima", "proyek",
                              "produk", "jumlah", "satuan", "harga"):
                crop = f["crop_ocr"]
                teks, n = ocr(crop)
                bb = f["bbox"]
                lebar = (bb[2] - bb[0]) if bb else 0
                print(f"  {f['label']:16s} w={lebar:3d} n={n:3d} teks={teks!r}")
                if crop is not None and getattr(crop, "size", 0):
                    simpan_gambar(OUT / f"{nama}_{f['id']}.png", crop)


if __name__ == "__main__":
    main()