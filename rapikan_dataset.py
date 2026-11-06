
from pathlib import Path
import shutil
import string

BASE = Path(__file__).resolve().parent

SUMBER = BASE / "dataset" / "original" / "huruf"

jumlah = 0

for file in list(SUMBER.iterdir()):
    if not file.is_file():
        continue

    if file.suffix.lower() not in (
        ".png", ".jpg", ".jpeg"
    ):
        continue

    label = file.stem.split("_")[0]

    if len(label) != 1 or label not in string.ascii_letters:
        print("Dilewati:", file.name)
        continue

    if label.isupper():
        tujuan = SUMBER / "uppercase" / label
    else:
        tujuan = SUMBER / "lowercase" / label

    tujuan.mkdir(parents=True, exist_ok=True)

    # Salin, jangan pindahkan atau menghapus file asli
    target = tujuan / file.name

    if target.exists():
        print("Sudah ada:", file.name)
        continue

    shutil.copy2(file, target)
    jumlah += 1

print(f"\nSelesai. {jumlah} file disalin ke folder karakter.")
