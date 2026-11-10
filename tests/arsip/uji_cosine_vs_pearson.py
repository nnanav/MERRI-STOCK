"""
Membandingkan skor Template Matching:
- cosine similarity (dipakai kode sekarang)
- Pearson correlation (dipakai jurnal referensi)

Diuji sebagai pengenal karakter (TM-only) pada field nota yang punya GT.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "tests") not in sys.path:
    sys.path.insert(0, str(BASE / "tests"))

import numpy as np
import cv2

import app
from processing.dataset_loader import normalisasi_karakter
from processing.pemrosesan_field import baca_gambar
from processing.pengenalan_karakter import muat_templates
from processing.segmentasi_karakter import segmentasi_karakter, susun_teks
from tests_gt import GT

NOTA = [
    ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
    ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
    ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
]


def siapkan_skor(templates):
    # Cosine: vektor ternormalisasi L2 (seperti kode sekarang).
    cos = templates / np.maximum(
        np.linalg.norm(templates, axis=1, keepdims=True), 1e-8
    )

    # Pearson: vektor di-mean-center lalu dinormalisasi.
    pusat = templates - templates.mean(axis=1, keepdims=True)
    pea = pusat / np.maximum(
        np.linalg.norm(pusat, axis=1, keepdims=True), 1e-8
    )
    return cos, pea


def prediksi(vec, tabel, labels):
    v = vec.astype(np.float32).flatten()
    n = np.linalg.norm(v)
    if n == 0:
        return "", 0.0
    v = v / n
    skor = tabel @ v
    i = int(np.argmax(skor))
    return labels[i], float(skor[i])


def main():
    templates, label_tm = muat_templates()
    tabel = np.array(templates, dtype=np.float32)
    cos, pea = siapkan_skor(tabel)

    model, labels = None, None
    from processing.pengenalan_karakter import muat_cnn
    model, labels = muat_cnn()

    hasil_uji = {"cosine": [], "pearson": []}

    for nama, path in NOTA:
        if not path.exists():
            continue
        det = app.jalankan_deteksi(baca_gambar(path))
        for f in det["fields"]:
            gt = GT.get((nama, f["id"]))
            crop = f["crop_ocr"]
            if not gt or crop is None or not getattr(crop, "size", 0):
                continue

            img = app.ke_tigakanal(crop)
            _, segmen = segmentasi_karakter(
                img, model, labels, tabel, label_tm
            )
            if not segmen:
                continue
            ta = float(np.median([s["y2"] - s["y1"] for s in segmen]))

            for nama_skor, tabel_skor in (("cosine", cos), ("pearson", pea)):
                keluaran = []
                # Koordinat segmen berada di ruang upsampled (SKALA), jadi
                # citra perlu diperbesar agar potongan sejajar.
                from processing.segmentasi_karakter import SKALA
                img_besar = cv2.resize(
                    img, None, fx=SKALA, fy=SKALA,
                    interpolation=cv2.INTER_CUBIC
                )
                for s in segmen:
                    y1 = max(0, min(img_besar.shape[0] - 1, int(s["y1"])))
                    y2 = max(y1 + 1, min(img_besar.shape[0], int(s["y2"])))
                    x1 = max(0, min(img_besar.shape[1] - 1, int(s["x1"])))
                    x2 = max(x1 + 1, min(img_besar.shape[1], int(s["x2"])))
                    potong = img_besar[y1:y2, x1:x2]
                    if potong.size == 0 or potong.shape[0] < 3 or potong.shape[1] < 2:
                        continue
                    if potong.ndim == 3:
                        potong = cv2.cvtColor(potong, cv2.COLOR_BGR2GRAY)
                    normal = normalisasi_karakter(potong)
                    if normal is None:
                        continue
                    ch, _ = prediksi(normal, tabel_skor, label_tm)
                    keluaran.append({
                        "x1": s["x1"], "x2": s["x2"],
                        "y1": s["y1"], "y2": s["y2"],
                        "tm": ch,
                    })

                teks = susun_teks(keluaran, "tm", ta) if keluaran else ""
                hasil_uji[nama_skor].append((nama, f["label"], gt, teks))

    print(f"Field ber-GT: {len(hasil_uji['cosine'])}\n")
    with open("hasil_cos_pea.txt", "w", encoding="utf-8") as f:
        f.write(f"Field ber-GT: {len(hasil_uji['cosine'])}\n\n")
        for nama_skor in ("cosine", "pearson"):
            total = 0.0
            benar = 0
            for nama, label, gt, teks in hasil_uji[nama_skor]:
                total += app.hitung_cer(gt, teks)
                benar += int(teks.strip() == gt.strip())
            n = max(1, len(hasil_uji[nama_skor]))
            f.write(
                f"TM {nama_skor:8s}: CER={total/n:.4f} "
                f"akurasi={(1-total/n)*100:5.2f}% benar={benar}/{n}\n"
            )

        f.write("\nContoh perbedaan:\n")
        for (n1, l1, gt, t1), (n2, l2, gt2, t2) in zip(
            hasil_uji["cosine"], hasil_uji["pearson"]
        ):
            if t1 != t2:
                f.write(f"  {n1} {l1}: GT={gt!r}\n")
                f.write(f"     cosine ={t1!r}\n")
                f.write(f"     pearson={t2!r}\n")
    print("selesai -> hasil_cos_pea.txt")


if __name__ == "__main__":
    main()