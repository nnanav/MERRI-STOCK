
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent

SUMBER_LAMA = Path(
    r"D:\PROJECT MARRIENA\templates\angka"
)

TUJUAN = BASE / "dataset" / "original" / "angka"

TUJUAN.mkdir(parents=True, exist_ok=True)

jumlah = 0

if not SUMBER_LAMA.exists():
    print("Folder template angka lama tidak ditemukan.")
else:
    for file in SUMBER_LAMA.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in (
            ".jpg", ".jpeg", ".png"
        ):
            continue

        label = file.stem.split("_")[0]

        if len(label) != 1 or not label.isdigit():
            print("Dilewati:", file.name)
            continue

        folder_angka = TUJUAN / label
        folder_angka.mkdir(
            parents=True,
            exist_ok=True
        )

        target = folder_angka / file.name

        if target.exists():
            print("Sudah ada:", file.name)
            continue

        shutil.copy2(file, target)
        jumlah += 1

print(f"\nSelesai. {jumlah} template angka disalin.")
