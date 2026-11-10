import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

t = Path("out_uat.txt").read_text(encoding="utf-8", errors="ignore")
t = t.replace("en-US", " ")
t = re.sub(r"\s+", " ", t)

mulai = int(sys.argv[1]) if len(sys.argv) > 1 else 0
akhir = int(sys.argv[2]) if len(sys.argv) > 2 else 7000
print(t[mulai:akhir])
