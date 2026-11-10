import re
import zlib
from pathlib import Path

path = Path(r"C:\Users\ADMIN\Documents\UAT SM - DAVINA.pdf")
data = path.read_bytes()

teks = []
for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
    blok = m.group(1)
    try:
        isi = zlib.decompress(blok)
    except Exception:
        continue
    # Ambil string teks di dalam operator Tj / TJ
    for s in re.finditer(rb"\((?:\\.|[^\\()])*\)", isi):
        t = s.group(0)[1:-1]
        t = t.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
        try:
            teks.append(t.decode("latin-1"))
        except Exception:
            pass

gabung = " ".join(teks)
gabung = gabung.replace("en-US", " ")
gabung = re.sub(r"\s+", " ", gabung)

Path("uat_penuh.txt").write_text(gabung, encoding="utf-8")
print("ditulis:", len(gabung))
