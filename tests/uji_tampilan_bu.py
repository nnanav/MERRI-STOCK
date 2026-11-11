"""Memeriksa isi halaman Business Understanding yang benar-benar dirender."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
APP = BASE / "app.py"

at = AppTest.from_file(str(APP), default_timeout=180)
at.run()
at.sidebar.radio[0].set_value("Business Understanding").run()

print("exception:", [e.value for e in at.exception])
print("\n=== SUBHEADER ===")
for s in at.subheader:
    print(" -", s.value)

print("\n=== METRIC ===")
for m in at.metric:
    print(f" {m.label:28s} = {m.value}")

print("\n=== CAPTION (10 pertama) ===")
for c in at.caption[:10]:
    print(" -", c.value)

print("\n=== EXPANDER ===")
for e in at.expander:
    print(" -", e.label)

print("\n=== BUTTON ===")
for b in at.button:
    print(" -", b.label, "| key:", b.key)

print("\n=== MARKDOWN ===")
print("jumlah:", len(at.markdown))
for i, md in enumerate(at.markdown):
    print(f"{i:02d} {md.value[:90]}")
