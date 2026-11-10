
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent

SUMBER = BASE / "dataset" / "augmented" / "huruf" / "simbol"
TUJUAN = BASE / "dataset" / "augmented" / "simbol"

KATEGORI = [
    "kurung_buka",
    "kurung_tutup",
    "koma",
    "persen",
    "petik",
    "slash",
    "strip",
    "titik"
]

jumlah = 0

if not SUMBER.exists():
    print("Folder sumber tidak ditemukan.")
else:
    for file in sorted(SUMBER.iterdir()):
        if not file.is_file():
            continue

        if file.suffix.lower() not in (
            ".png", ".jpg", ".jpeg"
        ):
            continue

        kategori = None

        for nama in KATEGORI:
            if file.name.startswith("aug_" + nama + "_"):
                kategori = nama
                break

        if kategori is None:
            print("Periksa manual:", file.name)
            continue

        folder = TUJUAN / kategori
        folder.mkdir(parents=True, exist_ok=True)

        target = folder / file.name

        if target.exists():
            print("Sudah ada, dilewati:", file.name)
            continue

        shutil.move(str(file), str(target))
        jumlah += 1

print(f"\nSelesai. {jumlah} file simbol dipindahkan.")
