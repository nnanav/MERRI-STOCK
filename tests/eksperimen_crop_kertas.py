"""Eksperimen offline A/B. Jalankan python -B tests/eksperimen_crop_kertas.py."""
import argparse
import json
import sys
from pathlib import Path
from types import FunctionType

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app
from processing.ground_truth import GT
from processing.preprocessing_nota import (
    _batas_pratinjau_kertas, crop_dan_luruskan, tingkatkan_kualitas,
)
from processing.segmentasi_karakter import binarisasi

OUT = ROOT / "output" / "eksperimen_crop_kertas"


def berbasis_kertas(foto, dasar, aktif):
    if dasar["bbox"] is None:
        return aktif, "fallback"
    crop = dasar["crop"]
    x1, y1, x2, y2 = _batas_pratinjau_kertas(crop)
    kertas = crop[y1:y2, x1:x2]
    hasil = tingkatkan_kualitas(kertas)
    return {
        "asli": foto, "crop": kertas, "hasil": hasil,
        "offset_crop": (x1, y1), "bbox_kertas_di_kanvas_dasar": dasar["bbox"],
    }, dasar["status"]


def deteksi_dan_mask(foto, preprocessing):
    # Gunakan kode fungsi aplikasi yang sama dengan dependensi preprocessing
    # lokal. Globals aplikasi dan cache Streamlit tidak dimodifikasi.
    mask = None

    def deteksi_baris(citra):
        nonlocal mask
        kecil, mask, visual, kotak = app.deteksi_baris(citra)
        return kecil, mask, visual, kotak

    fungsi = FunctionType(app.jalankan_deteksi.__code__, {
        **app.jalankan_deteksi.__globals__,
        "preprocess_nota": lambda _: preprocessing,
        "deteksi_baris": deteksi_baris,
    })
    hasil = fungsi(foto)
    return hasil, mask


def variasi(foto):
    h, w = foto.shape[:2]
    kecil = cv2.resize(foto, None, fx=.78, fy=.78, interpolation=cv2.INTER_AREA)
    yield "geser_skala", cv2.copyMakeBorder(
        kecil, h // 15, h // 8, w // 7, w // 35,
        cv2.BORDER_CONSTANT, value=(191, 187, 182),
    )
    rotasi = cv2.getRotationMatrix2D((w / 2, h / 2), 4, .9)
    yield "miring", cv2.warpAffine(
        foto, rotasi, (w, h), borderValue=(214, 212, 210),
    )
    yield "kontras_rendah", cv2.convertScaleAbs(foto, alpha=.56, beta=90)


def simpan(folder, nama, gambar):
    folder.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", gambar)
    if not ok:
        raise ValueError(nama)
    png.tofile(str(folder / f"{nama}.png"))


def ukur(acuan, prediksi):
    nilai = [app.hitung_cer_rinci(
        app.normalisasi_teks(gt), app.normalisasi_teks(prediksi.get(fid, "")),
    ) for fid, gt in acuan.items()]
    return dict(
        field=len(nilai), tepat=sum(int(row["sama"]) for row in nilai),
        edit=sum(row["jarak"] for row in nilai), karakter=sum(row["n"] for row in nilai),
    )


def jalankan(foto, nama, kondisi, acuan, visual, ocr):
    dasar = crop_dan_luruskan(foto)
    aktif = app.preprocess_nota(foto)
    kertas, metode = berbasis_kertas(foto, dasar, aktif)
    folder = OUT / "visual" / nama / kondisi
    if visual:
        simpan(folder, "00_foto_masukan", foto)
    for versi, pp, crop_kontras in (
        ("A", aktif, dasar["crop"]),
        ("B", kertas, dasar["crop"] if metode == "fallback" else kertas["crop"]),
    ):
        det, mask = deteksi_dan_mask(foto, pp)
        if versi == "A":
            asli = app.jalankan_deteksi(foto)
            assert np.array_equal(det["kecil"], asli["kecil"])
            for kiri, kanan in zip(det["fields"], asli["fields"], strict=True):
                assert (kiri["id"], kiri["bbox"]) == (kanan["id"], kanan["bbox"])
                assert np.array_equal(kiri["crop_ocr"], kanan["crop_ocr"])
        hasil = app.jalankan_ocr_semua(det, False) if ocr else {}
        pred = {fid: row["tanpa"] for fid, row in hasil.items()}
        if visual:
            folder_versi = folder / versi
            simpan(folder_versi, "01_area_crop_dasar", dasar["crop"])
            simpan(folder_versi, "02_masukan_contrast_stretching" if dasar["bbox"]
                   else "02_contrast_stretching_dilewati", crop_kontras)
            simpan(folder_versi, "03_kanvas_setelah_preprocessing", pp["hasil"])
            simpan(folder_versi, "04_kanvas_kerja_deteksi", det["kecil"])
            simpan(folder_versi, "05_mask_deteksi_setelah_buang_garis", mask)
            simpan(folder_versi, "06_overlay_ilustrasi", app.gambar_kotak_field(
                det["kecil"], det["fields"],
            ))
            for f in det["fields"]:
                crop = f["crop_ocr"]
                if crop is not None and crop.size:
                    simpan(folder_versi / "fields", f"{f['id']}_masukan_segmentasi", crop)
                    simpan(folder_versi / "fields", f"{f['id']}_biner_segmentasi",
                           binarisasi(app.ke_tigakanal(crop)))
        yield dict(
            nota=nama, kondisi=kondisi, pipeline=versi,
            batas=dasar["status"] if versi == "A" else metode,
            ukuran=pp["hasil"].shape[:2][::-1],
            crop_kontras=crop_kontras.shape[:2][::-1],
            ruang_bbox=det["kecil"].shape[:2][::-1],
            produk=len(det["produk"]),
            produk_gt=sum(fid.startswith("produk_") for fid in acuan) if acuan else None,
            tambahan=sorted(f["id"] for f in det["fields"] if f["id"] not in acuan)
            if acuan else None,
            metrik={k: ukur(acuan, {fid: p.get(k, "") for fid, p in pred.items()})
                    for k in ("teks", "cnn", "tm")} if acuan and ocr else None,
            fields={f["id"]: {
                "bbox": f["bbox"],
                **{k: pred.get(f["id"], {}).get(k) for k in ("teks", "cnn", "tm", "status")},
            } for f in det["fields"]},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nota", nargs="*", help="Stem nota untuk menjalankan subset.")
    parser.add_argument("--geometri", action="store_true", help="Lewati inferensi OCR.")
    parser.add_argument("--tanpa-variasi", action="store_true", help="Uji foto asli saja.")
    parser.add_argument("--visual-variasi", action="store_true", help="Simpan PNG variasi sintetis.")
    args = parser.parse_args()
    nama_hasil = "geometri.json" if args.geometri else "hasil.json"
    if args.nota:
        nama_hasil = "subset_" + nama_hasil
    if args.visual_variasi:
        nama_hasil = "visual_" + nama_hasil
    if args.tanpa_variasi:
        nama_hasil = "asli_" + nama_hasil
    cv2.setNumThreads(2)
    semua = sorted(
        path for folder in ("data_uji", "data_nota_asli")
        for path in (ROOT / folder).glob("*.jp*g")
    )
    hasil = []
    OUT.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(semua):
        nama = path.stem
        if args.nota and nama not in args.nota:
            continue
        foto = app.baca_gambar(path)
        acuan = {fid: value for (stem, fid), value in GT.items()
                 if stem == nama and fid.split("_")[0] in app.JENIS_DIEVALUASI}
        # Sampel sistematis dibekukan sebelum eksperimen. Seluruh foto asli
        # tetap masuk audit geometri, tanpa pemilihan berdasarkan hasil.
        sampel = bool(args.nota or acuan) or path.parent.name == "data_uji" or i % 7 == 0
        input_uji = [("asli", foto)]
        if sampel and not args.tanpa_variasi:
            input_uji.extend(variasi(foto))
        for kondisi, citra in input_uji:
            hasil.extend(jalankan(citra, nama, kondisi, acuan,
                                  sampel and (kondisi == "asli" or args.visual_variasi),
                                  sampel and not args.geometri))
        (OUT / nama_hasil).write_text(json.dumps(
            hasil, indent=2, ensure_ascii=False, default=lambda v: v.item(),
        ), encoding="utf-8")
        print(nama, flush=True)


if __name__ == "__main__":
    main()
