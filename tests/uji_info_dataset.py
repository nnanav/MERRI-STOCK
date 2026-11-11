"""Memastikan angka informasi dataset dihitung dari data sebenarnya."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app

info = app.info_dataset_latih()

for kunci in (
    "total", "huruf", "angka", "simbol", "kelas_dataset", "kelas_model",
    "dikecualikan", "dipakai", "latih", "validasi",
):
    print(f"{kunci:14s} = {info[kunci]}")

print("status         =", info["status"])
print("folder_kosong  =", info["folder_kosong"])

assert info["total"] == info["huruf"] + info["angka"] + info["simbol"], "grup != total"
assert info["dipakai"] == info["latih"] + info["validasi"], "latih+val != dipakai"
assert info["dipakai"] + info["dikecualikan"] == info["total"], "eksklusi salah"

print("\nSEMUA KONSISTEN")
