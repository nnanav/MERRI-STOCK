import sys
from pathlib import Path


def ambil(path):
    hasil = []
    for baris in Path(path).read_text(encoding="utf-8").splitlines():
        if baris.startswith("nota"):
            hasil.append(baris)
    return hasil


lama = ambil("cer_rb.txt")
baru = ambil("cer_baru2.txt")

print(f"{'nota':9s} {'field':17s} {'CER lama':9s} {'CER baru':9s} catatan")
print("-" * 74)

lebih_buruk = 0
membaik = 0
for a, b in zip(lama, baru):
    pa = a.split()
    pb = b.split()
    if len(pa) < 4 or len(pb) < 4:
        continue
    nama = pa[0]
    c1 = float(pa[-1])
    c2 = float(pb[-1])

    # nama field: ambil potongan tengah
    i = a.find("'")
    field = a[len(nama):i].strip() if i > 0 else "?"

    catatan = ""
    if c2 > c1 + 1e-9:
        catatan = "LEBIH BURUK"
        lebih_buruk += 1
    elif c2 < c1 - 1e-9:
        catatan = "membaik"
        membaik += 1

    print(f"{nama:9s} {field[:17]:17s} {c1:<9.3f} {c2:<9.3f} {catatan}")

print(f"\nMembaik: {membaik} | Lebih buruk: {lebih_buruk}")