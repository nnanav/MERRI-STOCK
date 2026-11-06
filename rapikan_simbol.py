
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
SUMBER = BASE / "dataset" / "original" / "simbol"

KATEGORI = {
    "koma": "koma",
    "kurung_buka": "kurung_buka",
    "kurung_tutup": "kurung_tutup",
    "persen": "persen",
    "petik": "petik",
    "slash": "slash",
    "strip": "strip",
    "titik": "titik",
}

jumlah = 0

for file in list(SUMBER.iterdir()):
    if not file.is_file():
        continue

    if file.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        continue

    nama = file.stem
    kategori = None

    for awalan in sorted(KATEGORI, key=len, reverse=True):
        if nama == awalan or nama.startswith(awalan + "_"):
            kategori = KATEGORI[awalan]
            break

    if kategori is None:
        print("Dilewati:", file.name)
        continue

    tujuan = SUMBER / kategori
    tujuan.mkdir(parents=True, exist_ok=True)

    target = tujuan / file.name

    if not target.exists():
        shutil.copy2(file, target)
        jumlah += 1

print(f"Selesai. {jumlah} simbol dikelompokkan.")

from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
SUMBER = BASE / "dataset" / "original" / "simbol"

KATEGORI = {
    "koma": "koma",
    "kurung_buka": "kurung_buka",
    "kurung_tutup": "kurung_tutup",
    "persen": "persen",
    "petik": "petik",
    "slash": "slash",
    "strip": "strip",
    "titik": "titik",
}

jumlah = 0

for file in list(SUMBER.iterdir()):
    if not file.is_file():
        continue

    if file.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        continue

    nama = file.stem
    kategori = None

    for awalan in sorted(KATEGORI, key=len, reverse=True):
        if nama == awalan or nama.startswith(awalan + "_"):
            kategori = KATEGORI[awalan]
            break

    if kategori is None:
        print("Dilewati:", file.name)
        continue

    tujuan = SUMBER / kategori
    tujuan.mkdir(parents=True, exist_ok=True)

    target = tujuan / file.name

    if not target.exists():
        shutil.copy2(file, target)
        jumlah += 1

print(f"Selesai. {jumlah} simbol dikelompokkan.")
