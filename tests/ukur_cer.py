"""
Mengukur evaluasi OCR per field utuh terhadap ground truth.

Menampilkan tabel yang sama dengan halaman Evaluasi aplikasi:
Field, Ground Truth, Hasil OCR, CER (%), dan Status.
CER = (S + D + I) / N x 100%. Status: CER 0% -> Benar, CER > 0% -> Salah.

Dipakai untuk membandingkan model lama vs model baru.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "tests") not in sys.path:
    sys.path.insert(0, str(BASE / "tests"))

import app
from processing.pemrosesan_field import baca_gambar
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def main():
    baris = []
    benar = 0
    salah = 0

    for nama, path in NOTA:
        if not path.exists():
            print(f"{nama}: file tidak ada")
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        for f in det["fields"]:
            # Field yang dievaluasi: nomor, penerima, proyek, produk,
            # jumlah, harga satuan (sama dengan halaman Evaluasi).
            if f.get("jenis") not in app.JENIS_DIEVALUASI:
                continue
            gt = GT.get((nama, f["id"]), "")
            if not app.normalisasi_teks(gt):
                continue

            crop = f["crop_ocr"]
            if crop is None or not getattr(crop, "size", 0):
                pred = ""
            else:
                pred = app.kenali_crop(
                    crop, pakai_clahe=False, jenis=f.get("jenis")
                )["teks"]

            r = app.hitung_cer_rinci(gt, pred)
            cocok = r["cer"] == 0.0
            benar += int(cocok)
            salah += int(not cocok)
            baris.append({
                "Field": f"{nama} / {f['label']}",
                "Ground Truth": gt,
                "Hasil OCR": pred,
                "CER (%)": round(r["cer"] * 100, 2),
                "Status": "Benar" if cocok else "Salah",
            })

    lebar = (26, 30, 30, 9, 8)
    print(
        f"{'Field':{lebar[0]}s} {'Ground Truth':{lebar[1]}s} "
        f"{'Hasil OCR':{lebar[2]}s} {'CER (%)':>{lebar[3]}s} "
        f"{'Status':{lebar[4]}s}"
    )
    print("-" * 120)
    for b in baris:
        print(
            f"{b['Field']:{lebar[0]}s} {b['Ground Truth']:{lebar[1]}s} "
            f"{b['Hasil OCR']:{lebar[2]}s} {b['CER (%)']:>{lebar[3]}.2f} "
            f"{b['Status']:{lebar[4]}s}"
        )

    dievaluasi = benar + salah
    akurasi = (benar / dievaluasi * 100.0) if dievaluasi else 0.0
    rata_cer = (
        sum(b["CER (%)"] for b in baris) / len(baris) if baris else 0.0
    )

    print(f"\nJumlah field dievaluasi = {dievaluasi}")
    print(f"Field benar (CER 0%) = {benar}")
    print(f"Field salah (CER > 0%) = {salah}")
    print(f"Exact-match field accuracy = {akurasi:.2f}%")
    print(f"Rata-rata CER = {rata_cer:.2f}%")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()