import re
import sys
import zlib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

path = Path(sys.argv[1])
data = path.read_bytes()

teks = []
for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
    blok = m.group(1)
    try:
        isi = zlib.decompress(blok)
    except Exception:
        continue
    for s in re.finditer(rb"\((?:\\.|[^\\()])*\)", isi):
        t = s.group(0)[1:-1]
        t = t.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
        try:
            teks.append(t.decode("latin-1"))
        except Exception:
            pass

gabung = " ".join(teks)
gabung = re.sub(r"\s+", " ", gabung)

keluaran = path.with_suffix(".txt")
keluaran.write_text(gabung, encoding="utf-8")
print("ditulis:", keluaran, len(gabung))
