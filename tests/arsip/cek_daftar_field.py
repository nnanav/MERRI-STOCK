import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app
from processing.pemrosesan_field import baca_gambar

det = app.jalankan_deteksi(baca_gambar(BASE / "data_uji" / "nota_52.jpeg"))
print("Jumlah field:", len(det["fields"]))
for f in det["fields"]:
    print(f"  jenis={f['jenis']:10s} label={f['label']}")
