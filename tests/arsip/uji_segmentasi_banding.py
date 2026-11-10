import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import app
from processing.pemrosesan_field import baca_gambar, simpan_gambar
from processing.pengenalan_karakter import (
    muat_cnn, muat_templates, gabung_segmen,
)
from processing.segmentasi_karakter import segmentasi_karakter as seg_baru
from processing.segmentasi_karakter import susun_teks

# Muat modul lama dari arsip untuk perbandingan.
_spes = importlib.util.spec_from_file_location(
    "segmentasi_lama",
    str(BASE / "archive" / "segmentasi_karakter_lama.py"),
)
mod_lama = importlib.util.module_from_spec(_spes)
_spes.loader.exec_module(mod_lama)
seg_lama = mod_lama.segmentasi_karakter

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg", "0086"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg", "0099"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg", "0299"),
]

FIELD_UJI = ["nomor", "produk_1", "jumlah_1", "harga_1", "penerima", "proyek"]
OUT = BASE / "output" / "segmentasi_banding"
OUT.mkdir(parents=True, exist_ok=True)


def ukur(hasil, tinggi_acuan):
    if not hasil:
        return {"n": 0, "lebar": 0.0, "maks": 0.0, "bergabung": 0}
    lebar = [(s["x2"] - s["x1"]) / max(tinggi_acuan, 1) for s in hasil]
    return {
        "n": len(hasil),
        "lebar": float(np.mean(lebar)),
        "maks": float(np.max(lebar)),
        "bergabung": int(sum(1 for z in lebar if z > 1.05)),
    }


def jalankan(seg, crop, model, labels, templates, label_tm):
    img = app.ke_tigakanal(crop)
    biner, hasil = seg(img, model, labels, templates, label_tm)
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    final, _ = gabung_segmen(hasil)
    return biner, hasil, susun_teks(final, "final", ta) if final else "", ta


def gambar_box(crop, hasil, ta):
    vis = app.ke_tigakanal(crop).copy()
    for i, s in enumerate(hasil):
        warna = (0, 170, 0) if s["status"] == "KANDIDAT" else (0, 0, 255)
        cv2.rectangle(vis, (s["x1"], s["y1"]), (s["x2"], s["y2"]), warna, 1)
    return vis


def main():
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    for nama, path, gt in NOTA:
        if not path.exists():
            print(nama, "file tidak ada")
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        print(f"\n################ {nama} ################")
        for fid in FIELD_UJI:
            f = next((x for x in det["fields"] if x["id"] == fid), None)
            if f is None or f["crop_ocr"] is None:
                print(f"  [{fid}] tidak ada crop")
                continue
            crop = f["crop_ocr"]

            b_l, h_l, t_l, ta_l = jalankan(seg_lama, crop, model, labels, templates, label_tm)
            b_b, h_b, t_b, ta_b = jalankan(seg_baru, crop, model, labels, templates, label_tm)

            m_l = ukur(h_l, ta_l)
            m_b = ukur(h_b, ta_b)

            cer_l = app.hitung_cer(gt, t_l) if fid == "nomor" else None
            cer_b = app.hitung_cer(gt, t_b) if fid == "nomor" else None

            print(f"  [{fid}]")
            print(f"    LAMA : n={m_l['n']:2d} lebar_rata={m_l['lebar']:.2f} maks={m_l['maks']:.2f} "
                  f"bergabung={m_l['bergabung']} teks={t_l!r}")
            print(f"    BARU : n={m_b['n']:2d} lebar_rata={m_b['lebar']:.2f} maks={m_b['maks']:.2f} "
                  f"bergabung={m_b['bergabung']} teks={t_b!r}")
            if cer_l is not None:
                print(f"    CER  : lama={cer_l:.3f} baru={cer_b:.3f}")

            simpan_gambar(OUT / f"{nama}_{fid}_asli.png", crop)
            simpan_gambar(OUT / f"{nama}_{fid}_lama_box.png", gambar_box(crop, h_l, ta_l))
            simpan_gambar(OUT / f"{nama}_{fid}_baru_box.png", gambar_box(crop, h_b, ta_b))
            simpan_gambar(OUT / f"{nama}_{fid}_baru_biner.png", b_b)


if __name__ == "__main__":
    main()