"""Audit kualitas segmentasi berbasis keyakinan prediksi.

Bukan sekadar aturan lebar: sebuah kasus dianggap cacat hanya bila bukti
prediksi mendukungnya.

  GABUNG (2 karakter dalam 1 kotak):
      kotak lebar + setinggi huruf, tetapi keyakinan CNN rendah atau CNN
      dan Template Matching tidak sepakat.
  POTONG (1 karakter terbelah 2 kotak):
      dua kotak sempit bersebelahan tanpa celah, gabungannya selebar satu
      karakter, dan gabungan itu dikenali JAUH lebih yakin daripada
      masing-masing bagian.
"""

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402
from processing.segmentasi_karakter import (  # noqa: E402
    segmentasi_utama, binarisasi, _prediksi_potongan,
)
from processing.pengenalan_karakter import muat_cnn, muat_templates  # noqa: E402

model, labels = muat_cnn()
templates, label_tm = muat_templates()

semua = sorted(
    [p for d in (BASE / "data_nota_asli", BASE / "data_uji")
     for p in d.glob("*.jp*g")],
    key=lambda p: p.name,
)
wajib = {"nota_01.jpeg", "nota_26.jpeg", "nota_52.jpeg"}
sampel = [p for i, p in enumerate(semua) if i % 4 == 0 or p.name in wajib]

print(f"Total nota tersedia: {len(semua)} | sampel: {len(select := sampel)}",
      flush=True)

tot_kotak = tot_gabung = tot_potong = 0
for path in sampel:
    gambar = baca_gambar(path)
    if gambar is None:
        continue
    det = app.jalankan_deteksi(gambar)
    g = p = 0
    detail = []
    for f in det["fields"]:
        crop = f.get("crop_show")
        if crop is None or not getattr(crop, "size", 0):
            continue
        biner = binarisasi(app.ke_tigakanal(crop))
        _, hasil, _ = segmentasi_utama(
            app.ke_tigakanal(crop), model, labels, templates, label_tm
        )
        if not hasil:
            continue
        tot_kotak += len(hasil)
        ta = float(np.median([max(1, s["y2"] - s["y1"]) for s in hasil]))

        # --- GABUNG: lebar + ragu ---
        # Huruf yang memang lebar (M, m, W, w) dan simbol pipih (titik,
        # koma, tanda hubung) dikecualikan karena lebarnya wajar.
        huruf_lebar = {"M", "m", "W", "w"}
        simbol_pipih = {".", ",", "-", "'", '"', "`", ":", ";"}
        for s in hasil:
            w = s["x2"] - s["x1"]
            h = s["y2"] - s["y1"]
            r = w / ta
            if r <= 1.15 or h < 0.75 * ta:
                continue
            if s.get("cnn") in huruf_lebar or s.get("cnn") in simbol_pipih:
                continue
            ragu = s.get("pc", 0) < 0.75 or s.get("tm") != s.get("cnn")
            if ragu:
                g += 1
                detail.append(
                    f"GABUNG w/ta={r:.2f} {s.get('cnn')!r} "
                    f"pc={s.get('pc', 0):.2f} @{f['label']}"
                )

        # --- POTONG: satu karakter terbelah ---
        for a, b in zip(hasil, hasil[1:]):
            wa, wb = a["x2"] - a["x1"], b["x2"] - b["x1"]
            celah = b["x1"] - a["x2"]
            if celah > 1 or wa >= 0.55 * ta or wb >= 0.55 * ta:
                continue
            if wa + wb + celah > 1.15 * ta:
                continue
            y1, y2 = min(a["y1"], b["y1"]), max(a["y2"], b["y2"])
            x1, x2 = a["x1"], b["x2"]
            pot = biner[max(0, y1):y2, max(0, x1):x2]
            gab = _prediksi_potongan(pot, model, labels, templates, label_tm)
            if gab is None:
                continue
            # Terbelah bila gabungan dikenali yakin sementara bagiannya lemah.
            if gab[2] >= 0.80 and min(a.get("pc", 0), b.get("pc", 0)) < 0.60:
                p += 1
                detail.append(
                    f"POTONG {a.get('cnn')!r}+{b.get('cnn')!r} -> {gab[0]!r} "
                    f"({gab[2]:.2f}) @{f['label']}"
                )

    tot_gabung += g
    tot_potong += p
    print(f"{path.name:16s} gabung={g:2d} potong={p:2d}", flush=True)
    for d in detail[:4]:
        print(f"    {d}", flush=True)

print(f"\nTOTAL kotak={tot_kotak} gabung={tot_gabung} potong={tot_potong}")
print(f"Rasio gabung={tot_gabung/max(1,tot_kotak)*100:.2f}% "
      f"potong={tot_potong/max(1,tot_kotak)*100:.2f}%")

