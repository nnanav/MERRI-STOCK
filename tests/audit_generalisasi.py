"""Ukur kesehatan pipeline pada nota yang tidak dipakai untuk tuning.

Tanpa ground truth, kualitas diperiksa dari kewajaran hasil:
  - field terdeteksi lengkap (nomor, penerima, proyek, produk, jumlah, harga)
  - crop tidak kosong/pudar
  - nomor nota berupa 4 digit
  - jumlah & harga hanya berisi angka + pemisah ( . , )
  - satuan pendek dan berupa huruf
"""

import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import app  # noqa: E402
from processing.pemrosesan_field import baca_gambar  # noqa: E402
from processing.segmentasi_karakter import crop_tanpa_teks  # noqa: E402

semua = sorted(
    [p for d in (BASE / "data_nota_asli", BASE / "data_uji")
     for p in d.glob("*.jp*g")],
    key=lambda p: p.name,
)
sampel = semua[::3]  # sepertiga nota, tersebar merata

RE_ANGKA = re.compile(r"^[0-9.,]+$")

print(f"Total nota: {len(semua)} | diuji: {len(sampel)}", flush=True)

stat = {
    "nota": 0, "field": 0, "field_kosong": 0,
    "nomor_ok": 0, "nomor_total": 0,
    "angka_ok": 0, "angka_total": 0,
    "satuan_ok": 0, "satuan_total": 0,
}
rincian = []

for path in sampel:
    gambar = baca_gambar(path)
    if gambar is None:
        continue
    try:
        det = app.jalankan_deteksi(gambar)
    except Exception as e:
        rincian.append(f"{path.name}: GAGAL deteksi ({e})")
        continue
    stat["nota"] += 1

    hasil = {}
    for f in det["fields"]:
        stat["field"] += 1
        crop = f.get("crop_ocr")
        if crop is None or not getattr(crop, "size", 0) or crop_tanpa_teks(crop):
            stat["field_kosong"] += 1
            continue
        try:
            teks = app.kenali_crop(
                crop, pakai_clahe=False, jenis=f.get("jenis")
            )["teks"]
        except Exception:
            teks = ""
        hasil[f["id"]] = (f["jenis"], teks)

    nomor = hasil.get("nomor", (None, ""))[1]
    if nomor:
        stat["nomor_total"] += 1
        stat["nomor_ok"] += int(bool(re.fullmatch(r"\d{4}", nomor)))

    for fid, (jenis, teks) in hasil.items():
        if jenis in ("jumlah", "harga") and teks:
            stat["angka_total"] += 1
            stat["angka_ok"] += int(bool(RE_ANGKA.fullmatch(teks.replace(" ", ""))))
        if jenis == "satuan" and teks:
            stat["satuan_total"] += 1
            stat["satuan_ok"] += int(bool(re.fullmatch(r"[A-Za-z]{2,4}", teks)))

    rincian.append(
        f"{path.name:16s} field={len(det['fields']):2d} "
        f"nomor={nomor!r:8s} "
        f"terbaca={len(hasil)}"
    )

print("\n".join(rincian))
print("\n=== RINGKASAN GENERALISASI ===")
print(f"Nota diproses        : {stat['nota']}")
print(f"Field terdeteksi     : {stat['field']}")
print(f"Field crop kosong    : {stat['field_kosong']} "
      f"({stat['field_kosong']/max(1,stat['field'])*100:.1f}%)")
print(f"Nomor nota 4 digit   : {stat['nomor_ok']}/{stat['nomor_total']} "
      f"({stat['nomor_ok']/max(1,stat['nomor_total'])*100:.1f}%)")
print(f"Jumlah/harga numerik : {stat['angka_ok']}/{stat['angka_total']} "
      f"({stat['angka_ok']/max(1,stat['angka_total'])*100:.1f}%)")
print(f"Satuan wajar         : {stat['satuan_ok']}/{stat['satuan_total']} "
      f"({stat['satuan_ok']/max(1,stat['satuan_total'])*100:.1f}%)")
