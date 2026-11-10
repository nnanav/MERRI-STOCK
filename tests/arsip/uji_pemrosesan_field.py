from pathlib import Path

import cv2
import numpy as np

from uji_deteksi_baris import deteksi_baris
from processing.deteksi_field import (
    deteksi_produk, deteksi_field_lain,
)
from processing.pemrosesan_field import (
    isolasi_nomor, bersihkan_minus, pisah_jumlah_satuan,
    cari_tanggal, baca_gambar, simpan_gambar,
)
from processing.lokasi_field_lama import cari_area_field

BASE = Path(__file__).resolve().parent
OUT = BASE / "diagnostik_v2"

DAFTAR = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
    ("nota_54", Path(r"D:\PROJECT MARRIENA\data_uji\nota_54.jpeg")),
]


def potong(gambar, k):
    H, W = gambar.shape[:2]
    x1 = max(0, int(k["x1"])); y1 = max(0, int(k["y1"]))
    x2 = min(W, int(k["x2"])); y2 = min(H, int(k["y2"]))
    if x2 <= x1 or y2 <= y1:
        return None
    return gambar[y1:y2, x1:x2].copy()


def main():
    for nama, path in DAFTAR:
        print("\n=================", nama, "=================")
        if not path.exists():
            print("  FILE TIDAK ADA:", path)
            continue

        gambar = baca_gambar(path)
        kecil, biner, visual, kotak = deteksi_baris(gambar)
        H, W = kecil.shape[:2]
        folder = OUT / nama
        folder.mkdir(parents=True, exist_ok=True)

        # Nomor nota
        nom = isolasi_nomor(kecil)
        print(f"  [NOMOR] status={nom['status']} | {nom['pesan']}")
        if nom.get("crop_baris") is not None:
            simpan_gambar(folder / "nomor_baris.png", nom["crop_baris"])
        if nom.get("crop_digit") is not None:
            simpan_gambar(folder / "nomor_digit.png", nom["crop_digit"])

        # Produk + minus
        kandidat = [{
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "w": x2 - x1, "h": y2 - y1,
        } for (x1, y1, x2, y2) in kotak]

        produk = deteksi_produk(kandidat, W, H)
        print(f"  [PRODUK] jumlah={len(produk)}")

        for i, p in enumerate(produk, 1):
            crop = potong(kecil, p)
            if crop is None:
                print(f"    produk {i}: crop kosong")
                continue
            hasil = bersihkan_minus(crop)
            simpan_gambar(folder / f"produk_{i}_sebelum.png", crop)
            simpan_gambar(folder / f"produk_{i}_sesudah.png", hasil["crop"])
            print(f"    produk {i}: minus_dihapus={hasil['dihapus']} | {hasil['pesan']}")

        # Jumlah & satuan per produk
        info = deteksi_field_lain(kandidat, W, H, produk)
        for r in info["rincian_produk"]:
            n = r["produk"]
            kj = r["jumlah_satuan"]
            if kj is None:
                print(f"    jumlah/satuan produk {n}: tidak terdeteksi")
                continue
            crop = potong(kecil, kj)
            if crop is None:
                continue
            simpan_gambar(folder / f"jumlahsatuan_{n}_gabungan.png", crop)
            pisah = pisah_jumlah_satuan(crop)
            print(f"    jumlah/satuan produk {n}: status={pisah['status']} | {pisah['pesan']}")
            if pisah.get("jumlah") is not None:
                simpan_gambar(folder / f"jumlah_{n}.png", pisah["jumlah"])
            if pisah.get("satuan") is not None:
                simpan_gambar(folder / f"satuan_{n}.png", pisah["satuan"])

        # Tanggal: pakai wilayah ORB/homografi sebagai region pencarian.
        lok, err = cari_area_field(kecil)
        reg = lok["tanggal"] if (lok and "tanggal" in lok) else None
        tg = cari_tanggal(kecil, lokasi_tanggal=reg)
        print(f"  [TANGGAL] status={tg['status']} asal={tg.get('asal')} | {tg['pesan']}")
        for kunci in ("wilayah", "kandidat", "kandidat_biner",
                      "mask_buang", "bersih_visual"):
            if tg.get(kunci) is not None:
                simpan_gambar(folder / f"tanggal_{kunci}.png", tg[kunci])

        print("  Output:", folder)


if __name__ == "__main__":
    main()