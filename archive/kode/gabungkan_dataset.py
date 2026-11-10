
from pathlib import Path
from collections import Counter
import hashlib
import shutil
import string

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"

SUMBER = [
    ("asli", DATASET / "original"),
    ("aug", DATASET / "augmented"),
]

EKSTENSI = {".png", ".jpg", ".jpeg"}

SIMBOL = (
    "kurung_buka",
    "kurung_tutup",
    "garis_miring",
    "titik_dua",
    "koma",
    "titik",
    "strip",
    "minus",
    "slash",
    "persen",
    "petik",
    "apostrof",
    "samadengan",
)

ALIASES = {
    "minus": "strip",
    "garis_miring": "slash",
    "apostrof": "petik",
}

def nama_kelas(file, relatif):
    bagian = relatif.parts
    nama = file.stem

    # Cari kategori huruf dari struktur folder
    if "uppercase" in bagian:
        i = bagian.index("uppercase")
        if i + 1 < len(bagian):
            label = bagian[i + 1]
        else:
            label = nama.split("_")[0]

        if len(label) == 1 and label in string.ascii_uppercase:
            return Path("huruf") / "uppercase" / label

    if "lowercase" in bagian:
        i = bagian.index("lowercase")
        if i + 1 < len(bagian):
            label = bagian[i + 1]
        else:
            label = nama.split("_")[0]

        if len(label) == 1 and label in string.ascii_lowercase:
            return Path("huruf") / "lowercase" / label

    # Angka dari folder atau nama file
    if "angka" in bagian:
        i = bagian.index("angka")
        label = (
            bagian[i + 1]
            if i + 1 < len(bagian)
            else nama.split("_")[0]
        )

        if len(label) == 1 and label in string.digits:
            return Path("angka") / label

    # Simbol dari folder atau nama file
    if "simbol" in bagian:
        i = bagian.index("simbol")

        if i + 1 < len(bagian):
            kategori = bagian[i + 1]
            kategori = ALIASES.get(kategori, kategori)

            if kategori in SIMBOL:
                return Path("simbol") / kategori

        # Menangani file simbol yang masih tercampur
        teks = nama
        if teks.startswith("aug_"):
            teks = teks[4:]

        for simbol in sorted(SIMBOL, key=len, reverse=True):
            if teks == simbol or teks.startswith(simbol + "_"):
                kategori = ALIASES.get(simbol, simbol)
                return Path("simbol") / kategori

    # Huruf lama yang belum dikelompokkan
    if "huruf" in bagian:
        teks = nama
        if teks.startswith("aug_"):
            teks = teks[4:]

        label = teks.split("_")[0]

        if len(label) == 1:
            if label in string.ascii_uppercase:
                return Path("huruf") / "uppercase" / label

            if label in string.ascii_lowercase:
                return Path("huruf") / "lowercase" / label

    return None


def hash_file(file):
    h = hashlib.sha256()

    with open(file, "rb") as f:
        for blok in iter(lambda: f.read(65536), b""):
            h.update(blok)

    return h.hexdigest()


def gabungkan():
    jumlah = Counter()
    dilewati = []
    total_baru = 0

    for jenis, sumber in SUMBER:
        if not sumber.exists():
            print("Tidak ditemukan:", sumber)
            continue

        for file in sorted(sumber.rglob("*")):
            if not file.is_file():
                continue

            if file.suffix.lower() not in EKSTENSI:
                continue

            relatif = file.relative_to(sumber)
            kelas = nama_kelas(file, relatif)

            if kelas is None:
                dilewati.append(str(file))
                continue

            tujuan = DATASET / kelas
            tujuan.mkdir(parents=True, exist_ok=True)

            # Nama berbeda untuk asli dan augmentasi.
            # Hash mencegah tabrakan nama file.
            identitas = hash_file(file)
            nama_baru = (
                f"{jenis}_{file.stem}_"
                f"{identitas[:16]}{file.suffix.lower()}"
            )

            target = tujuan / nama_baru

            if not target.exists():
                shutil.copy2(file, target)
                total_baru += 1

            jumlah[str(kelas)] += 1

    print("\n=== PENGGABUNGAN SELESAI ===")
    print("File baru disalin:", total_baru)

    print("\n=== JUMLAH PER KELAS ===")

    for kelas in sorted(jumlah):
        folder = DATASET / kelas

        file_gambar = [
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in EKSTENSI
        ]

        total = len(file_gambar)
        status = "CUKUP" if total >= 50 else "KURANG"

        print(
            f"{kelas}: {total}/50 - {status}"
        )

    if dilewati:
        print("\n=== FILE PERLU DIPERIKSA ===")
        for nama in dilewati:
            print(nama)


if __name__ == "__main__":
    gabungkan()
