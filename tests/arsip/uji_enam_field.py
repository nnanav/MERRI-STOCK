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

OUT = BASE / "output" / "diagnostik"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    for nama, path in NOTA:
        if not path.exists():
            print(nama, "-> file tidak ada")
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        vis = app.gambar_kotak_field(det["kecil"], det["fields"])
        simpan_gambar(OUT / f"{nama}_enam_field_box.png", vis)

        print(f"\n===== {nama} : {len(det['fields'])} field =====")
        for f in det["fields"]:
            crop = f["crop_show"]
            ukuran = crop.shape[:2] if crop is not None else None
            ada = "ADA" if crop is not None and getattr(crop, "size", 0) else "TIDAK"
            print(f"  {f['label']:16s} bbox={f['bbox']} crop={ada} {ukuran}")
            if crop is not None and getattr(crop, "size", 0):
                simpan_gambar(OUT / f"{nama}_{f['id']}.png", crop)


if __name__ == "__main__":
    main()