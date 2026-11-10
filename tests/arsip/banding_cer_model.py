import sys
from pathlib import Path

def ambil(path):
    hasil = []
    for baris in Path(path).read_text(encoding="utf-8").splitlines():
        if baris.startswith("nota"):
            bagian = baris.split()
            hasil.append((bagian[0], " ".join(bagian[1:-1]), float(bagian[-1])))
    return hasil

lama = ambil("cer_lama.txt")
baru = ambil("cer_baru.txt")

print(f"{'nota':9s} {'field':26s} {'CER lama':9s} {'CER baru':9s} catatan")
print("-" * 78)
for (n1, f1, c1), (n2, f2, c2) in zip(lama, baru):
    catatan = ""
    if c2 > c1 + 1e-9:
        catatan = "LEBIH BURUK"
    elif c2 < c1 - 1e-9:
        catatan = "membaik"
    print(f"{n1:9s} {f1:26s} {c1:<9.3f} {c2:<9.3f} {catatan}")
