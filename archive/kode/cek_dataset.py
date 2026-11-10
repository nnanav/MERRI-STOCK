
from pathlib import Path
import string

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset" / "original"

print("\n=== REKAP DATASET KARAKTER ASLI ===\n")

total = 0
belum_ada = []

for huruf in string.ascii_letters + string.digits:

    if huruf.isupper():
        folder = DATASET / "huruf" / "uppercase" / huruf
    elif huruf.islower():
        folder = DATASET / "huruf" / "lowercase" / huruf
    else:
        folder = DATASET / "angka" / huruf

    if folder.exists():
        jumlah = len([
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
    else:
        jumlah = 0

    total += jumlah

    if jumlah == 0:
        belum_ada.append(huruf)

    kurang = max(0, 50 - jumlah)

    print(
        f"{huruf} : {jumlah:3d} asli | "
        f"Target 50 | Kurang {kurang}"
    )

print("\nTotal sampel asli:", total)
print("Karakter belum tersedia:", ", ".join(belum_ada) or "Tidak ada")
