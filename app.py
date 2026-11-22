from pathlib import Path

import gc

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from processing.deteksi_baris import deteksi_baris
from processing.deteksi_field import (
    deteksi_produk, deteksi_field_lain,
    deteksi_penerima_proyek, deteksi_harga_satuan,
    pangkas_awalan_tipis, pangkas_hingga_rp,
)
from processing.data_transaksi import (
    buat_csv, simpan_transaksi, validasi_transaksi,
)
from processing.pemrosesan_field import (
    isolasi_nomor, bersihkan_minus, pisah_jumlah_satuan,
    cari_tanggal, baca_gambar, segmentasi_per_karakter,
)
from processing.preprocessing_nota import preprocess_nota
from ui.tema import (
    alur, kepala_halaman, legenda_status, pasang_tema, seksi,
    sidebar_brand, sidebar_label_menu, sidebar_status,
)

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="MERRI-STOCK — Pengolahan Nota",
    page_icon=":material/receipt_long:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tema visual terpusat (CSS di ui/tema.py).
pasang_tema()


def gambar_rgb(img):
    # Samakan ke RGB agar aman ditampilkan Streamlit.
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def potong(img, k, margin=0):
    H, W = img.shape[:2]
    x1 = max(0, int(k["x1"]) - margin)
    y1 = max(0, int(k["y1"]) - margin)
    x2 = min(W, int(k["x2"]) + margin)
    y2 = min(H, int(k["y2"]) + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def ke_tigakanal(img):
    # segmentasi_karakter memakai cvtColor(BGR2GRAY) yang butuh 3 kanal.
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def terapkan_clahe(bgr):
    # CLAHE pada kanal L; dipakai untuk jalur 'dengan CLAHE'.
    bgr = ke_tigakanal(bgr)
    if bgr is None:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def hitung_cer(gt, pred):
    # Character Error Rate = jarak Levenshtein / panjang ground truth.
    gt = (gt or "").strip()
    pred = (pred or "").strip()
    if not gt:
        return None
    sebelumnya = list(range(len(pred) + 1))
    for i, cg in enumerate(gt, 1):
        sekarang = [i]
        for j, cp in enumerate(pred, 1):
            biaya = 0 if cg == cp else 1
            sekarang.append(min(
                sebelumnya[j] + 1,
                sekarang[j - 1] + 1,
                sebelumnya[j - 1] + biaya,
            ))
        sebelumnya = sekarang
    return sebelumnya[-1] / len(gt)


def normalisasi_teks(teks):
    """
    Normalisasi ringan dan konsisten untuk ground truth maupun hasil OCR.

    Hanya spasi yang dirapikan (spasi berlebih menjadi satu, spasi tepi
    dibuang). Huruf, angka, dan tanda baca TIDAK diubah, sehingga
    perbedaan seperti "Pcs" vs "PCS" atau titik pada "39.000" tetap
    terhitung sebagai kesalahan.
    """
    return " ".join(str(teks or "").split())


def hitung_cer_rinci(gt, pred):
    """
    Hitung CER satu field secara utuh beserta rincian operasi edit.

    CER = (S + D + I) / N, dengan
      S = substitution (karakter ground truth diganti karakter lain)
      D = deletion     (karakter ground truth hilang pada hasil OCR)
      I = insertion    (karakter tambahan pada hasil OCR)
      N = jumlah karakter ground truth

    CER adalah satu-satunya metrik evaluasi (tidak ada similarity).

    Mengembalikan dict: n, s, d, i, jarak, cer, sama.
    `cer` berupa pecahan 0..1, atau None bila ground truth kosong
    (tidak ada pembagi sehingga tidak terjadi pembagian dengan nol).
    """
    gt = normalisasi_teks(gt)
    pred = normalisasi_teks(pred)
    n = len(gt)
    m = len(pred)

    if n == 0:
        return {
            "n": 0, "s": 0, "d": 0, "i": 0, "jarak": m,
            "cer": None, "sama": gt == pred,
        }

    # Matriks jarak edit (Levenshtein).
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(n + 1):
        dp[a][0] = a
    for b in range(m + 1):
        dp[0][b] = b
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            biaya = 0 if gt[a - 1] == pred[b - 1] else 1
            dp[a][b] = min(
                dp[a - 1][b - 1] + biaya,  # cocok / substitusi
                dp[a - 1][b] + 1,          # deletion
                dp[a][b - 1] + 1,          # insertion
            )

    # Telusuri balik untuk memisahkan S, D, dan I.
    sub = dele = ins = 0
    a, b = n, m
    while a > 0 or b > 0:
        if (a > 0 and b > 0 and gt[a - 1] == pred[b - 1]
                and dp[a][b] == dp[a - 1][b - 1]):
            a -= 1
            b -= 1
        elif a > 0 and b > 0 and dp[a][b] == dp[a - 1][b - 1] + 1:
            sub += 1
            a -= 1
            b -= 1
        elif a > 0 and dp[a][b] == dp[a - 1][b] + 1:
            dele += 1
            a -= 1
        else:
            ins += 1
            b -= 1

    jarak = sub + dele + ins
    cer = jarak / n
    return {
        "n": n, "s": sub, "d": dele, "i": ins, "jarak": jarak,
        "cer": cer, "sama": jarak == 0,
    }


@st.cache_resource
def muat_semua_model():
    from processing.pengenalan_karakter import muat_cnn, muat_templates
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()
    return model, labels, templates, label_tm


def bersihkan_tepi(teks):
    # Buang sisa tanda baca/label di awal dan akhir hasil OCR.
    if not teks:
        return teks

    teks = teks.strip()
    tepi = ":;=|~_-,. "

    # Buang berulang sampai tidak ada lagi yang bisa dibuang.
    sebelum = None
    while sebelum != teks:
        sebelum = teks
        teks = teks.strip(tepi)

    # Sisa label di awal seperti "P, " atau "P " (dari kata "Proyek").
    import re
    teks = re.sub(r"^[Pp][,\s.]+", "", teks)

    return teks.strip()


def buang_tanda_kali(hasil):
    """
    Buang tanda kali di ujung segmen satuan.

    Pada nota, kolom satuan diikuti tanda kali (mis. "Pcs x"). Tanda itu
    bukan bagian isi satuan, jadi kotak dan teksnya tidak perlu diikutkan.
    """
    if not hasil:
        return hasil
    if str(hasil[-1].get("cnn", "")).strip() in {"x", "X", "*", "×"}:
        return hasil[:-1]
    return hasil


def rapikan_satuan(teks):
    """
    Rapikan hasil OCR kolom satuan.

    Dua koreksi:
    1. Tanda kali di akhir dibuang. Pada nota, kolom satuan diikuti tanda
       kali, sehingga hasil OCR sering menjadi "Pcs x" padahal isinya
       hanya "Pcs".
    2. Penulisan diseragamkan menjadi huruf besar di awal saja
       ("PCS" atau "PcS" menjadi "Pcs"), karena satuan ditulis begitu
       pada nota. Token yang memuat angka (mis. "M3") tidak diubah.
    """
    if not teks:
        return teks
    import re
    teks = re.sub(r"[\s.,]*[xX×\*]+\s*$", "", teks)

    def seragam(token):
        if token.isalpha() and len(token) <= 6:
            return token[0].upper() + token[1:].lower()
        return token

    bagian = re.split(r"(\s+)", teks)
    teks = "".join(b if not b.strip() else seragam(b) for b in bagian)
    return teks.strip()


@st.cache_data(show_spinner=False, max_entries=400)
def _segmentasi_cache(ndim, tinggi, lebar, data_bytes):
    """
    Jalankan segmentasi+klasifikasi untuk satu crop, dengan cache.

    Tanpa cache, halaman Segmentasi menjalankan CNN + Template Matching +
    dynamic programming untuk SETIAP field pada SETIAP rerun Streamlit,
    sehingga memuat halaman terasa sangat lambat (puluhan detik sampai
    menit). Dengan cache, hasil yang sama dipakai ulang selama isi crop
    tidak berubah — termasuk dipakai bersama oleh halaman Segmentasi dan
    Pengenalan karena keduanya memakai crop yang sama.
    """
    from processing.segmentasi_karakter import segmentasi_utama

    arr = np.frombuffer(data_bytes, np.uint8)
    crop = arr.reshape(tinggi, lebar, 3) if ndim == 3 else arr.reshape(
        tinggi, lebar
    )
    model, labels, templates, label_tm = muat_semua_model()
    _, hasil, metode = segmentasi_utama(
        crop, model, labels, templates, label_tm
    )
    return hasil, metode


def segmentasi_crop(crop):
    """Segmentasi satu crop memakai cache. Mengembalikan (hasil, metode)."""
    img = ke_tigakanal(crop)
    if img is None or img.size == 0:
        return [], "-"
    img = np.ascontiguousarray(img)
    return _segmentasi_cache(
        img.ndim, img.shape[0], img.shape[1], img.tobytes()
    )


def kenali_crop(crop, pakai_clahe=False, jenis=None):
    # Satu pipeline OCR: TM menemukan posisi karakter, CNN mengklasifikasi.
    from processing.segmentasi_karakter import susun_teks
    from processing.pengenalan_karakter import gabung_segmen
    img = ke_tigakanal(crop)
    if img is None or img.size == 0:
        return {"teks": "", "cnn": "", "tm": "", "n": 0, "detail": [],
                "metode": "-", "status": "crop_kosong"}
    if pakai_clahe:
        img = terapkan_clahe(img)
    hasil, metode = segmentasi_crop(img)
    if jenis == "satuan":
        hasil = buang_tanda_kali(hasil)
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    hasil_final, detail = gabung_segmen(hasil)
    teks = bersihkan_tepi(susun_teks(hasil_final, "final", ta, jenis))
    if jenis == "satuan":
        teks = rapikan_satuan(teks)
    return {
        "teks": teks,
        "cnn": susun_teks(hasil, "cnn", ta),
        "tm": susun_teks(hasil, "tm", ta),
        "n": len(hasil),
        "detail": detail,
        "metode": metode,
        "status": "ok",
    }


def perluas_kotak_kiri(kecil, box, batas_kiri=0, faktor_gap=0.6):
    # Perluas kotak ke kiri menelusuri tinta yang berdekatan, supaya
    # token jumlah yang terpotong (mis. '3.' pada '3.400') ikut masuk.
    # Berhenti bila menemukan celah lebar atau menyentuh batas_kiri.
    H, W = kecil.shape[:2]
    x1 = int(box["x1"])
    y1 = max(0, int(box["y1"]))
    x2 = int(box["x2"])
    y2 = min(H, int(box["y2"]))
    if x2 <= x1 or y2 <= y1:
        return box

    h = max(1, y2 - y1)
    batas_kiri = max(0, int(batas_kiri))
    if x1 <= batas_kiri:
        return box

    band = cv2.cvtColor(kecil[y1:y2, :], cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.count_nonzero(biner, axis=0)
    ambang = max(3, int(faktor_gap * h))

    baru_x1 = x1
    gap = 0
    nx = x1
    while nx - 1 >= batas_kiri:
        if proj[nx - 1] == 0:
            gap += 1
            if gap > ambang:
                break
        else:
            gap = 0
            baru_x1 = nx - 1
        nx -= 1

    hasil = dict(box)
    hasil["x1"] = int(baru_x1)
    hasil["w"] = int(x2 - baru_x1)
    return hasil


def jalankan_deteksi(gambar_bgr):
    # Preprocessing lebih dulu: kertas nota dipotong, perspektif diluruskan,
    # lalu kualitas gambar ditingkatkan dengan satu metode terpilih.
    # Hasilnya yang diteruskan ke deteksi field dan tahap sesudahnya.
    preproses = preprocess_nota(gambar_bgr)
    gambar_bgr = preproses["hasil"]

    # Deteksi lima field. Koordinat di ruang gambar yang lebarnya 1400.
    kecil, _, _, kotak = deteksi_baris(gambar_bgr)
    H, W = kecil.shape[:2]

    # Crop untuk OCR diambil dari gambar asli, bukan dari gambar proses
    # yang sudah diperkecil ke 1400 px. Pada nota proyek ini tinggi huruf
    # baris transaksi hanya ~11 px di ruang proses, sehingga detail seperti
    # titik pada "10.550" hilang bila crop diambil dari gambar kecil.
    # Kotak field tetap dihitung di ruang proses agar visualisasi konsisten.
    if gambar_bgr is not None and gambar_bgr.shape[1] > W:
        sumber_ocr = gambar_bgr
        skala_ocr = gambar_bgr.shape[1] / float(W)
    else:
        sumber_ocr = kecil
        skala_ocr = 1.0

    def potong_ocr(kotak, margin=0):
        # Potong kotak ruang proses dari sumber resolusi asli.
        if kotak is None:
            return None
        Hs, Ws = sumber_ocr.shape[:2]
        x1 = max(0, int(round(int(kotak["x1"]) * skala_ocr)) - margin)
        y1 = max(0, int(round(int(kotak["y1"]) * skala_ocr)) - margin)
        x2 = min(Ws, int(round(int(kotak["x2"]) * skala_ocr)) + margin)
        y2 = min(Hs, int(round(int(kotak["y2"]) * skala_ocr)) + margin)
        if x2 <= x1 or y2 <= y1:
            return None
        return sumber_ocr[y1:y2, x1:x2].copy()

    def lebar_ke_proses(lebar_piksel):
        # Kembalikan lebar crop (piksel asli) ke ruang proses.
        return int(round(lebar_piksel / skala_ocr))

    def crop_field(kotak, jenis):
        """
        Potong field sesuai karakter isinya.

        Field harga berisi angka dengan titik ribuan yang hanya setebal
        ~1 px di ruang proses, sehingga mudah hilang saat gambar diperkecil
        ke 1400 px. Field harga karena itu dipotong dari gambar asli.
        Field teks (penerima, proyek, produk, satuan) tetap dipotong dari
        gambar proses karena segmentasinya lebih stabil di ukuran itu.
        """
        if kotak is None:
            return None
        if jenis == "harga":
            return potong_ocr(kotak)
        return potong(kecil, kotak)

    kandidat = [{
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
        "w": x2 - x1, "h": y2 - y1,
    } for (x1, y1, x2, y2) in kotak]

    produk = deteksi_produk(kandidat, W, H)
    info = deteksi_field_lain(kandidat, W, H, produk)

    nomor = isolasi_nomor(kecil)

    penerima, proyek = deteksi_penerima_proyek(kandidat, W, H)
    harga = deteksi_harga_satuan(kandidat, W, H, produk)

    fields = []

    fields.append({
        "id": "nomor", "label": "Nomor Nota", "jenis": "nomor",
        "bbox": nomor.get("bbox"),
        "crop_ocr": nomor.get("crop_digit"),
        "crop_show": nomor.get("crop_digit"),
        "status": nomor.get("status"),
        "pesan": nomor.get("pesan", ""),
    })

    def field_dari_kotak(fid, label, jenis, kotak):
        crop = crop_field(kotak, jenis)
        if crop is not None and jenis == "harga":
            crop = pangkas_hingga_rp(crop)
        elif crop is not None and jenis in ("penerima", "proyek"):
            crop = pangkas_awalan_tipis(crop)
        # Bila crop dipangkas, sesuaikan bbox agar kotak di UI ikut pas.
        bbox = None
        if kotak is not None:
            lebar_asli = int(kotak["x2"]) - int(kotak["x1"])
            lebar_crop = lebar_asli
            if crop is not None:
                lebar_crop = crop.shape[1]
                if jenis == "harga":
                    # Crop harga berasal dari gambar asli.
                    lebar_crop = lebar_ke_proses(lebar_crop)
            bbox = (int(kotak["x1"]), int(kotak["y1"]),
                    int(kotak["x1"]) + int(lebar_crop), int(kotak["y2"]))
        return {
            "id": fid, "label": label, "jenis": jenis,
            "bbox": bbox,
            "crop_ocr": crop, "crop_show": crop,
            "status": "ok" if crop is not None else "tidak_terdeteksi",
            "pesan": label,
        }

    fields.append(field_dari_kotak("penerima", "Nama Penerima", "penerima", penerima))
    fields.append(field_dari_kotak("proyek", "Proyek", "proyek", proyek))

    for i, p in enumerate(produk, 1):
        crop = crop_field(p, "produk")
        bersih = bersihkan_minus(crop) if crop is not None else None
        crop_produk = bersih["crop"] if bersih else None

        rincian = info["rincian_produk"][i - 1] if i - 1 < len(info["rincian_produk"]) else {}
        kj = rincian.get("jumlah_satuan")
        kotak_j = crop_j = None
        kotak_s = crop_s = None

        if kj is not None:
            # Perluas ke kiri agar token jumlah yang terpotong ikut masuk.
            kj = perluas_kotak_kiri(kecil, kj, batas_kiri=p["x2"] + 2)
            cjs = crop_field(kj, "jumlah")
            if cjs is not None:
                pisah = pisah_jumlah_satuan(cjs)
                crop_j = pisah.get("jumlah")
                crop_s = pisah.get("satuan")
                xb = pisah.get("x_batas") or kj["w"]
                kotak_j = {"x1": kj["x1"], "y1": kj["y1"],
                           "x2": kj["x1"] + xb, "y2": kj["y2"]}
                kotak_s = {"x1": kj["x1"] + xb, "y1": kj["y1"],
                           "x2": kj["x2"], "y2": kj["y2"]}

        fields.append({
            "id": f"produk_{i}", "label": f"Nama Produk {i}", "jenis": "produk",
            "bbox": (p["x1"], p["y1"], p["x2"], p["y2"]),
            "crop_ocr": crop_produk, "crop_show": crop_produk,
            "status": "ok" if crop_produk is not None else "crop_kosong",
            "pesan": bersih["pesan"] if bersih else "produk tidak terbaca",
        })
        fields.append({
            "id": f"jumlah_{i}", "label": f"Jumlah {i}", "jenis": "jumlah",
            "bbox": (kotak_j["x1"], kotak_j["y1"], kotak_j["x2"], kotak_j["y2"]) if kotak_j else None,
            "crop_ocr": crop_j, "crop_show": crop_j,
            "status": "ok" if crop_j is not None else "tidak_terdeteksi",
            "pesan": "jumlah",
        })
        fields.append({
            "id": f"satuan_{i}", "label": f"Satuan {i}", "jenis": "satuan",
            "bbox": (kotak_s["x1"], kotak_s["y1"], kotak_s["x2"], kotak_s["y2"]) if kotak_s else None,
            "crop_ocr": crop_s, "crop_show": crop_s,
            "status": "ok" if crop_s is not None else "tidak_terdeteksi",
            "pesan": "satuan",
        })
        fields.append(field_dari_kotak(
            f"harga_{i}", f"Harga Satuan {i}", "harga",
            harga[i - 1] if i - 1 < len(harga) else None,
        ))

    # "biner" dan "visual" dari deteksi_baris tidak disimpan: keduanya tidak
    # dipakai halaman mana pun dan hanya membebani memori. Pada mesin ber-RAM
    # kecil, menyimpan citra tambahan per nota bisa memicu kegagalan alokasi
    # OpenCV pada tahap berikutnya.
    return {
        "kecil": kecil,
        "kandidat": kandidat, "nomor": nomor,
        "produk": produk, "fields": fields,
        "preproses": preproses,
    }


def gambar_kotak_field(base, fields, skala=1.0, ox=0.0, oy=0.0):
    # Gambar bounding box tiap field di atas citra `base`.
    # Kotak field disimpan dalam ruang pemrosesan (lebar 1400); parameter
    # skala/ox/oy memetakannya ke ruang citra `base` bila berbeda.
    vis = base.copy()
    warna = {
        "nomor": (0, 0, 255),
        "penerima": (0, 140, 255),
        "proyek": (255, 140, 0),
        "produk": (0, 160, 0),
        "jumlah": (255, 0, 0),
        "satuan": (0, 200, 200),
        "harga": (180, 0, 180),
    }
    H, W = vis.shape[:2]
    for f in fields:
        bb = f.get("bbox")
        if not bb:
            continue
        x1 = int(round(int(bb[0]) / skala - ox))
        y1 = int(round(int(bb[1]) / skala - oy))
        x2 = int(round(int(bb[2]) / skala - ox))
        y2 = int(round(int(bb[3]) / skala - oy))
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        c = warna.get(f["jenis"], (0, 0, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.putText(vis, f["label"], (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return vis


def ambil_deteksi():
    # Hasil deteksi hanya tersedia setelah tombol "Proses Nota" di Upload.
    return st.session_state.get("deteksi")


def segmen_field(crop, jenis=None):
    # Segmentasi per karakter memakai pipeline sebenarnya (CNN+Template
    # Matching + dynamic programming) lewat cache, agar kotak yang
    # ditampilkan sama dengan yang dipakai pengenalan TANPA menjalankan
    # ulang model pada setiap rerun.
    cimg = ke_tigakanal(crop)
    if cimg is None or cimg.size == 0:
        return None, [], None, "", []

    try:
        from processing.segmentasi_karakter import susun_teks, SKALA
        from processing.pengenalan_karakter import gabung_segmen

        hasil, _metode = segmentasi_crop(cimg)
        if jenis == "satuan":
            hasil = buang_tanda_kali(hasil)
        final, detail = gabung_segmen(hasil)
        tinggi = [s["y2"] - s["y1"] for s in hasil]
        ta = float(np.median(tinggi)) if tinggi else 12.0
        teks = susun_teks(final, "final", ta, jenis) if final else ""
        if jenis == "satuan":
            teks = rapikan_satuan(teks)

        # Koordinat hasil berada di ruang upsampled (SKALA x); kembalikan
        # ke ukuran crop asli agar kotak pas.
        h, w = cimg.shape[:2]
        karakter = []
        for s, fin in zip(hasil, final):
            karakter.append({
                "x1": int(s["x1"] / SKALA), "y1": int(s["y1"] / SKALA),
                "x2": min(w, int(s["x2"] / SKALA)),
                "y2": min(h, int(s["y2"] / SKALA)),
                "cnn": s.get("cnn", ""), "tm": s.get("tm", ""),
                "final": fin.get("final", ""),
            })
        return _gambar_kotak(cimg, karakter), karakter, None, teks, detail
    except Exception:
        hasil = segmentasi_per_karakter(cimg)
        karakter = hasil["karakter"]
        return _gambar_kotak(cimg, karakter), karakter, hasil.get("biner"), "", []


def _gambar_kotak(cimg, karakter):
    vis = cimg.copy()
    for i, k in enumerate(karakter):
        warna = (0, 170, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.rectangle(vis, (k["x1"], k["y1"]), (k["x2"], k["y2"]), warna, 1)
    return vis


def halaman_upload():
    kepala_halaman(
        "Upload Nota",
        "Unggah satu gambar nota (JPG/PNG), lalu klik Proses Nota. Proses ini "
        "mendeteksi baris teks dengan adaptive threshold dan mencari lokasi "
        "field memakai connected components.",
    )

    kol_unggah, kol_panduan = st.columns([3, 2], gap="large")
    with kol_unggah:
        berkas = st.file_uploader(
            "Unggah gambar nota", type=["png", "jpg", "jpeg"],
            key="unggah_nota",
        )
    with kol_panduan:
        with st.container(border=True):
            st.markdown("**Panduan unggah**")
            st.caption(
                "Format didukung: JPG, JPEG, dan PNG. Unggah satu nota per "
                "berkas, pastikan seluruh bagian nota terlihat, dan hindari "
                "gambar yang terlalu gelap atau buram."
            )

    if berkas is not None:
        baru = st.session_state.get("nama_file") != berkas.name
        if baru:
            # Kosongkan hasil nota sebelumnya LEBIH DULU agar memori lega
            # sebelum gambar baru didekode. Pada mesin ber-RAM kecil,
            # menyimpan nota lama + mendekode nota baru sekaligus bisa
            # membuat OpenCV gagal alokasi ("Insufficient memory").
            st.session_state["deteksi"] = None
            st.session_state["ocr_semua"] = None
            st.session_state["ocr_kunci"] = None
            st.session_state["preproses"] = None
            st.session_state["gambar_bgr"] = None
            # Form Hasil Ekstraksi ikut direset: kunci widgetnya memakai
            # nama berkas, dan data terkonfirmasi nota lama dibuang.
            st.session_state["ekstraksi_tersimpan"] = None
            gc.collect()

        gambar_baru = None
        data = None
        try:
            data = np.frombuffer(berkas.getvalue(), np.uint8)
            gambar_baru = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except cv2.error as galat:
            st.error(
                "Gambar gagal didekode karena memori komputer tidak cukup. "
                "Tutup aplikasi lain (mis. tab browser atau editor) lalu "
                "unggah ulang nota. Detail: " + str(galat).splitlines()[0]
            )
            return
        except MemoryError:
            st.error(
                "Memori komputer habis saat membaca gambar. Tutup aplikasi "
                "lain lalu unggah ulang nota."
            )
            return
        except Exception as galat:  # jaring pengaman: jangan tampilkan traceback
            st.error(
                "Berkas gagal dibaca. Pastikan berkas berupa gambar JPG/PNG "
                f"yang utuh. Detail: {type(galat).__name__}: {galat}"
            )
            return
        finally:
            data = None
            gc.collect()

        if gambar_baru is None:
            st.error("Gambar gagal dibaca. Coba berkas lain.")
            return
        if baru:
            st.session_state["nama_file"] = berkas.name
            st.session_state["gambar_bgr"] = gambar_baru

    gambar = st.session_state.get("gambar_bgr")
    nama = st.session_state.get("nama_file")
    if gambar is None:
        st.info("Silakan unggah gambar nota untuk memulai.")
        return

    st.success(f"Nota: {nama}")
    with st.container(border=True):
        st.subheader("Pratinjau Nota")
        st.image(gambar_rgb(gambar), width="stretch")
        st.caption(
            "Gambar asli hasil unggahan, belum dipotong maupun diperbaiki "
            "kualitasnya. Perbaikan dilakukan pada halaman Preprocessing."
        )

    with st.container(border=True):
        st.subheader("Proses Nota")
        st.caption(
            "Proses Nota memotong kertas nota, meluruskan perspektif, lalu "
            "menyimpan hasil deteksi field untuk halaman berikutnya."
        )
        kol = st.columns([1, 2])
        if kol[0].button("Proses Nota", type="primary", key="proses_nota"):
            with st.spinner("Memproses nota (deteksi field)..."):
                st.session_state["deteksi"] = jalankan_deteksi(gambar)
                st.session_state["ocr_semua"] = None
                st.session_state["ocr_kunci"] = None

        if st.session_state.get("deteksi") is not None:
            det = st.session_state["deteksi"]
            kol[1].success(
                f"Nota berhasil diproses — {len(det['fields'])} field terdeteksi."
            )
            st.info(
                "Lanjutkan ke halaman Preprocessing, Segmentasi Karakter, "
                "Pengenalan Karakter, Hasil Ekstraksi, lalu Evaluasi."
            )


def halaman_preprocessing():
    kepala_halaman(
        "Preprocessing",
        "Menyiapkan gambar sebelum pengenalan: kertas nota dipotong dan "
        "diluruskan, kontras dinaikkan, lalu citra dibinerkan agar tulisan "
        "terpisah dari latar.",
    )
    gambar = st.session_state.get("gambar_bgr")
    if gambar is None:
        st.info("Unggah nota di halaman Upload terlebih dahulu.")
        return

    nama = st.session_state.get("nama_file")
    pp = st.session_state.get("preproses")
    if pp is None or pp.get("nama") != nama:
        with st.spinner("Menyiapkan preprocessing (crop + peningkatan kualitas)..."):
            pp = preprocess_nota(gambar)
            pp["nama"] = nama
            st.session_state["preproses"] = pp

    hasil = pp["hasil"]
    tampil = pp.get("tampil", hasil)

    st.subheader("1. Crop dan koreksi perspektif")
    st.caption(
        "Batas kertas dicari dengan ambang Otsu dan cadangan tepi Canny, lalu "
        "gambar dipotong dan diluruskan memakai transformasi perspektif empat "
        "titik sudut agar teks sejajar."
    )
    with st.container(border=True):
        a, b = st.columns(2)
        a.image(gambar_rgb(pp["asli"]), caption="Nota asli", width="stretch")
        b.image(gambar_rgb(pp["crop"]), caption="Hasil crop kertas nota",
                width="stretch")

    st.subheader("2. Peningkatan kualitas gambar")
    st.caption(
        f"Metode yang dipakai: {pp['metode']}. Kontras dinaikkan pada kanal "
        "luminansi agar tulisan lebih tegas tanpa memperkuat noise kertas."
    )
    with st.container(border=True):
        st.image(gambar_rgb(tampil), caption="Hasil akhir preprocessing",
                 width="stretch")

    st.subheader("3. Binerisasi")
    st.caption(
        "Citra diubah ke grayscale lalu dibinerkan dengan ambang Otsu, "
        "sehingga tinta menjadi hitam dan latar putih sebagai dasar "
        "pemisahan karakter."
    )
    abu = cv2.cvtColor(tampil, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.image(abu, caption="Grayscale", width="stretch")
        c2.image(otsu, caption="Binarisasi Otsu", width="stretch")


# Nama tampilan dan urutan jenis field. Dipakai bersama oleh halaman
# Segmentasi dan Pengenalan agar kedua halaman menampilkan field yang sama.
# Field "satuan" sengaja tidak ditampilkan: isinya tidak dipakai pada data
# transaksi (jumlah dan satuan sudah dipisah saat deteksi, dan yang disimpan
# hanya jumlah), sehingga tab Satuan hanya menambah halaman tanpa manfaat.
LABEL_JENIS = {
    "nomor": "Nomor Nota",
    "penerima": "Nama Penerima",
    "proyek": "Proyek",
    "produk": "Nama Produk",
    "jumlah": "Jumlah",
    "harga": "Harga Satuan",
}

URUTAN_JENIS = (
    "nomor", "penerima", "proyek", "produk", "jumlah", "harga",
)

TAB_FIELD = [(jenis, LABEL_JENIS[jenis]) for jenis in URUTAN_JENIS]

# Jenis field yang ditampilkan pada halaman Segmentasi dan Pengenalan.
JENIS_TAMPIL = tuple(jenis for jenis, _ in TAB_FIELD)


def halaman_segmentasi():
    kepala_halaman(
        "Segmentasi Karakter",
        "Teks tiap field dipisah menjadi satu kotak per karakter memakai "
        "projection profile, yaitu hitungan piksel tinta per kolom. Bila "
        "karakter menempel, Dynamic Programming memilih batas potong terbaik.",
    )
    st.caption(
        "Setiap potongan karakter di bawah ini menjadi masukan halaman "
        "Pengenalan Karakter."
    )

    if st.session_state.get("gambar_bgr") is None:
        st.info(
            "Unggah nota dan klik 'Proses Nota' di halaman Upload terlebih "
            "dahulu."
        )
        return

    det = ambil_deteksi()
    if det is None:
        st.info("Klik 'Proses Nota' di halaman Upload terlebih dahulu.")
        return

    # Segmentasi seluruh field dihitung sekali di awal. Hasilnya sudah
    # di-cache, sehingga memakai tabs tidak menambah waktu proses.
    # Field di luar JENIS_TAMPIL (mis. satuan) dilewati agar tidak dihitung
    # percuma karena tidak ditampilkan di halaman ini.
    data = {}
    for f in det["fields"]:
        if f.get("jenis") not in JENIS_TAMPIL:
            continue
        crop = f.get("crop_show")
        if crop is None or not getattr(crop, "size", 0):
            data[f["id"]] = None
            continue
        vis, karakter, _biner, teks, _detail = segmen_field(
            crop, f.get("jenis")
        )
        data[f["id"]] = {
            "crop": crop,
            "vis": vis,
            "karakter": karakter,
            "teks": teks,
        }

    per_jenis = {}
    for f in det["fields"]:
        per_jenis.setdefault(f.get("jenis"), []).append(f)

    tabs = st.tabs([judul for _, judul in TAB_FIELD])

    for tab, (jenis, label) in zip(tabs, TAB_FIELD):
        with tab:
            daftar = per_jenis.get(jenis, [])
            if not daftar:
                st.caption(
                    f"Field {label} tidak terdeteksi pada nota ini."
                )
                continue

            for urutan, f in enumerate(daftar, 1):
                if urutan > 1:
                    st.divider()
                _tampilkan_field_segmen(
                    f, data.get(f["id"]), label, len(daftar), urutan
                )


def _tampilkan_field_segmen(f, d, label, jumlah_field, urutan):
    """Tampilkan satu field: informasi, citra, dan potongan tiap karakter."""
    # Satu field dibungkus satu panel agar hasil tiap field mudah dibaca.
    with st.container(border=True):
        # Bila satu jenis field muncul beberapa kali (mis. beberapa baris
        # produk), beri nomor agar tidak tertukar.
        st.subheader(f"{label} {urutan}" if jumlah_field > 1 else label)

        if d is None:
            st.caption("Crop field ini tidak tersedia.")
            return

        karakter = d["karakter"]
        st.caption(f"{len(karakter)} karakter terdeteksi")

        kiri, kanan = st.columns(2)
        with kiri:
            st.image(
                gambar_rgb(d["crop"]),
                caption="Citra asli field",
                width="stretch",
            )
        with kanan:
            if d["vis"] is not None:
                st.image(
                    gambar_rgb(d["vis"]),
                    caption="Bounding box karakter",
                    width="stretch",
                )
            else:
                st.caption("Bounding box tidak tersedia.")

        st.markdown("**Hasil Pemisahan Karakter**")

        if not karakter:
            st.caption("Tidak ada karakter yang tersegmentasi.")
            return

        # Maksimal 6 potongan per baris. Jumlah kolom dibuat tetap agar lebar
        # gambar antar baris konsisten.
        per_baris = 6
        for mulai in range(0, len(karakter), per_baris):
            irisan = karakter[mulai:mulai + per_baris]
            kolom = st.columns(per_baris)
            for slot, k in enumerate(irisan):
                with kolom[slot]:
                    potong = d["crop"][k["y1"]:k["y2"], k["x1"]:k["x2"]]
                    nomor_karakter = mulai + slot + 1
                    if getattr(potong, "size", 0):
                        st.image(
                            gambar_rgb(potong),
                            caption=f"Karakter {nomor_karakter}",
                            width="stretch",
                        )
                    else:
                        st.caption(f"Karakter {nomor_karakter}")


def kenali_satu(crop, pakai_clahe, jenis=None):
    try:
        return kenali_crop(crop, pakai_clahe=pakai_clahe, jenis=jenis)
    except Exception as error:
        return {"cnn": "", "tm": "", "n": 0, "status": f"gagal: {error}"}


def jalankan_ocr_semua(det, sertakan_clahe):
    kosong = {"cnn": "", "tm": "", "n": 0, "status": "crop_kosong"}
    hasil = {}
    for f in det["fields"]:
        crop = f["crop_ocr"]
        if crop is None or getattr(crop, "size", 0) == 0:
            hasil[f["id"]] = {"tanpa": kosong, "clahe": kosong}
            continue
        jenis = f.get("jenis")
        tanpa = kenali_satu(crop, False, jenis)
        clahe = kenali_satu(crop, True, jenis) if sertakan_clahe else \
            {"cnn": "", "tm": "", "n": 0, "status": "-"}
        hasil[f["id"]] = {"tanpa": tanpa, "clahe": clahe}
    return hasil


def halaman_pengenalan():
    kepala_halaman(
        "Pengenalan Karakter",
        "Tiap potongan karakter diseragamkan menjadi 32x32 piksel, lalu "
        "dikenali CNN sebagai penentu label dengan Template Matching sebagai "
        "pendukung. Hasilnya disusun menjadi teks per field.",
    )

    if st.session_state.get("gambar_bgr") is None:
        st.info("Unggah nota di halaman Upload terlebih dahulu.")
        return

    det = ambil_deteksi()
    if det is None:
        st.info("Klik 'Proses Nota' di halaman Upload terlebih dahulu.")
        return

    kunci = (st.session_state.get("nama_file"),)
    if st.session_state.get("ocr_kunci") != kunci or st.session_state.get("ocr_semua") is None:
        with st.spinner("Mengenali semua field (segmentasi + TM + CNN)..."):
            st.session_state["ocr_semua"] = jalankan_ocr_semua(det, False)
            st.session_state["ocr_kunci"] = kunci

    hasil = st.session_state["ocr_semua"]

    st.caption(
        "Pilih tab field untuk melihat teks hasil OCR. Teks ini yang dipakai "
        "pada halaman Hasil Ekstraksi dan Evaluasi."
    )

    per_jenis = {}
    for f in det["fields"]:
        per_jenis.setdefault(f.get("jenis"), []).append(f)

    tabs = st.tabs([judul for _, judul in TAB_FIELD])

    for tab, (jenis, label) in zip(tabs, TAB_FIELD):
        with tab:
            daftar = per_jenis.get(jenis, [])
            if not daftar:
                st.caption(f"Field {label} tidak terdeteksi pada nota ini.")
                continue

            for urutan, f in enumerate(daftar, 1):
                if urutan > 1:
                    st.divider()
                t = hasil.get(f["id"], {}).get("tanpa", {})
                _tampilkan_field_kenal(f, t, label, len(daftar), urutan)


def _tampilkan_field_kenal(f, t, label, jumlah_field, urutan):
    """Tampilkan satu field: crop, teks hasil OCR, dan jumlah karakter."""
    # Satu field dibungkus satu panel agar hasil pembacaan mudah dibaca.
    with st.container(border=True):
        # Bila satu jenis field muncul beberapa kali (mis. beberapa baris
        # produk), beri nomor agar tidak tertukar.
        st.subheader(f"{label} {urutan}" if jumlah_field > 1 else label)

        kiri, kanan = st.columns([1, 2], gap="large")
        with kiri:
            crop = f.get("crop_show")
            if crop is not None and getattr(crop, "size", 0):
                st.image(
                    gambar_rgb(crop),
                    caption="Citra field",
                    width="stretch",
                )
            else:
                st.caption("Crop field tidak tersedia.")

        with kanan:
            st.markdown("**Hasil OCR**")
            teks = t.get("teks", "") or ""
            if teks:
                st.code(teks, language=None)
            else:
                st.caption("Belum terbaca.")
            st.caption(f"Jumlah karakter: {t.get('n', 0)}")

            # Keluaran kedua metode ditampilkan apa adanya (tidak diubah):
            # CNN sebagai penentu label, Template Matching sebagai pendukung.
            cnn = str(t.get("cnn") or "").strip()
            tm = str(t.get("tm") or "").strip()
            if cnn or tm:
                st.caption(
                    f"CNN: {cnn or '-'}  ·  "
                    f"Template Matching: {tm or '-'}"
                )

            if t.get("status", "").startswith("gagal"):
                st.warning(t["status"])


def _teks_ocr(hasil, field_id):
    """Ambil teks hasil OCR satu field apa adanya (tidak diubah)."""
    h = (hasil or {}).get(field_id) or {}
    return str(((h.get("tanpa") or {}).get("teks") or "")).strip()


def _produk_dari_ocr(det, hasil):
    """
    Susun baris produk awal untuk halaman Hasil Ekstraksi.

    Satu baris mewakili satu produk, diambil dari field produk_i, jumlah_i,
    dan harga_i pada hasil OCR. Nilai OCR hanya dipakai sebagai nilai awal;
    pengguna boleh mengoreksinya tanpa mengubah hasil OCR asli.
    """
    baris = []
    for f in det["fields"]:
        if f.get("jenis") != "produk":
            continue
        urutan = str(f["id"]).split("_")[-1]
        baris.append({
            "Nama Produk": _teks_ocr(hasil, f"produk_{urutan}"),
            "Jumlah": _teks_ocr(hasil, f"jumlah_{urutan}"),
            "Harga Satuan": _teks_ocr(hasil, f"harga_{urutan}"),
        })
    return baris


def _sel_teks(nilai):
    """Ubah isi data editor menjadi teks; sel kosong jadi string kosong."""
    if nilai is None or pd.isna(nilai):
        return ""
    return str(nilai).strip()


def _gambar_pratinjau(det, nama):
    """
    Ambil gambar nota untuk pratinjau halaman Hasil Ekstraksi.

    Yang dipakai adalah hasil crop kertas nota (sudah dikoreksi perspektif
    dan ditingkatkan kualitasnya) agar tulisan mudah dibaca, bukan foto
    unggahan apa adanya. Bila hasil crop tidak tersedia, dipakai gambar
    unggahan sebagai cadangan.

    Mengembalikan (gambar, keterangan).
    """
    pp = st.session_state.get("preproses")
    if not pp or pp.get("nama") != nama:
        # Hasil preprocessing juga disimpan pada hasil deteksi, sehingga
        # halaman ini tetap benar walau halaman Preprocessing belum dibuka.
        pp = (det or {}).get("preproses")

    if pp:
        for kunci in ("tampil", "crop", "hasil"):
            gambar = pp.get(kunci)
            if gambar is not None and getattr(gambar, "size", 0):
                return gambar, "Hasil crop kertas nota"

    return st.session_state.get("gambar_bgr"), "Gambar nota asli"


def halaman_hasil_ekstraksi():
    st.header("Hasil Pembacaan Nota")
    st.caption("Periksa dan koreksi informasi transaksi sebelum disimpan.")

    if st.session_state.get("gambar_bgr") is None:
        st.info("Unggah nota di halaman Upload terlebih dahulu.")
        return

    det = ambil_deteksi()
    if det is None:
        st.info("Klik 'Proses Nota' di halaman Upload terlebih dahulu.")
        return

    hasil = st.session_state.get("ocr_semua")
    if not hasil:
        # Halaman ini tidak menjalankan OCR sendiri. Hasil OCR dihitung
        # sekali di halaman Pengenalan Karakter lalu dipakai bersama.
        st.info(
            "Hasil OCR belum tersedia. Buka halaman Pengenalan Karakter "
            "sekali agar hasil OCR dihitung, lalu kembali ke halaman ini."
        )
        return

    nama = st.session_state.get("nama_file") or "nota"

    # Status ini hanya menyatakan OCR selesai, bukan bahwa hasilnya benar.
    st.success(
        "Proses OCR selesai. Hasil di bawah belum tentu benar, jadi periksa "
        "dan koreksi dulu sebelum disimpan."
    )

    # Kunci widget memakai nama berkas, sehingga mengunggah nota baru
    # otomatis menampilkan form berisi hasil OCR nota baru itu.
    kunci = f"ekstraksi_{nama}"
    pratinjau, keterangan = _gambar_pratinjau(det, nama)

    kol_kiri, kol_kanan = st.columns([1, 1.6], gap="large")

    with kol_kiri:
        with st.container(border=True):
            st.subheader("Pratinjau Nota")
            st.image(
                gambar_rgb(pratinjau),
                caption=f"{keterangan} — {nama}",
                width="stretch",
            )
            st.caption(
                "Gambar hasil crop kertas nota setelah koreksi perspektif dan "
                "peningkatan kualitas, bukan foto unggahan asli."
            )

    with kol_kanan:
        with st.container(border=True):
            st.subheader("Informasi Transaksi")
            st.caption(
                "Nilai awal diambil dari hasil OCR; perbaiki yang tidak sesuai "
                "dengan nota. Nomor Nota, Nama Penerima, dan Proyek wajib diisi."
            )

            nomor = st.text_input(
                "Nomor Nota", value=_teks_ocr(hasil, "nomor"),
                key=f"{kunci}_nomor",
            )
            kol_a, kol_b = st.columns(2)
            penerima = kol_a.text_input(
                "Nama Penerima", value=_teks_ocr(hasil, "penerima"),
                key=f"{kunci}_penerima",
            )
            proyek = kol_b.text_input(
                "Proyek", value=_teks_ocr(hasil, "proyek"),
                key=f"{kunci}_proyek",
            )

        with st.container(border=True):
            st.subheader("Produk")
            st.caption(
                "Satu baris untuk satu produk; tambah atau hapus baris sesuai "
                "isi nota. Jumlah harus angka positif dan harga satuan tidak "
                "negatif, dengan format Indonesia seperti 9.000 atau 10.550."
            )

            awal = _produk_dari_ocr(det, hasil)
            if not awal:
                awal = [{"Nama Produk": "", "Jumlah": "", "Harga Satuan": ""}]

            # Isi tabel disimpan di session_state agar hasil koreksi pengguna
            # tidak hilang saat Streamlit rerun (mis. setelah tombol Simpan).
            kunci_tabel = f"{kunci}_produk_data"
            if kunci_tabel not in st.session_state:
                st.session_state[kunci_tabel] = pd.DataFrame(
                    awal, columns=["Nama Produk", "Jumlah", "Harga Satuan"]
                )

            edit = st.data_editor(
                st.session_state[kunci_tabel],
                key=f"{kunci}_produk",
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Nama Produk": st.column_config.TextColumn(
                        "Nama Produk", width="large", required=True
                    ),
                    "Jumlah": st.column_config.TextColumn(
                        "Jumlah", help="Contoh: 400 atau 9.000"
                    ),
                    "Harga Satuan": st.column_config.TextColumn(
                        "Harga Satuan", help="Contoh: 10.550"
                    ),
                },
            )
            st.session_state[kunci_tabel] = edit.copy()

    # Data yang divalidasi: nilai isian saat ini, bukan hasil OCR asli.
    data = {
        "nomor": nomor,
        "penerima": penerima,
        "proyek": proyek,
        "produk": [
            {
                "nama": _sel_teks(baris["Nama Produk"]),
                "jumlah": _sel_teks(baris["Jumlah"]),
                "harga": _sel_teks(baris["Harga Satuan"]),
            }
            for _, baris in edit.iterrows()
        ],
    }

    st.divider()
    with st.container(border=True):
        st.markdown("**Simpan dan Unduh**")
        st.caption(
            "Simpan Data memvalidasi isian lalu menyimpannya ke basis data "
            "SQLite. Unduh Hasil menghasilkan CSV satu baris per produk dan "
            "aktif setelah data tersimpan."
        )
        kol_simpan, kol_unduh = st.columns(2)

        if kol_simpan.button(
            "Simpan Data", type="primary", key=f"{kunci}_simpan"
        ):
            bersih, galat = validasi_transaksi(data)
            if galat:
                for pesan in galat:
                    st.error(pesan)
                st.caption("Data belum disimpan. Perbaiki isian di atas.")
            else:
                try:
                    id_transaksi, baru = simpan_transaksi(
                        bersih, nama_file=nama
                    )
                except Exception as galat_db:
                    st.error(
                        "Data gagal disimpan: "
                        f"{type(galat_db).__name__}: {galat_db}"
                    )
                else:
                    # Data terkonfirmasi disimpan terpisah dari hasil OCR dan
                    # dipakai untuk tombol unduh.
                    st.session_state["ekstraksi_tersimpan"] = {
                        "kunci": nama, "id": id_transaksi, "data": bersih,
                    }
                    if baru:
                        st.success(
                            "Data transaksi berhasil disimpan "
                            f"(ID {id_transaksi}) dengan "
                            f"{len(bersih['produk'])} produk."
                        )
                    else:
                        st.warning(
                            "Data yang sama sudah tersimpan sebelumnya "
                            f"(ID {id_transaksi}), jadi tidak disimpan ulang."
                        )

        tersimpan = st.session_state.get("ekstraksi_tersimpan") or {}
        siap_unduh = tersimpan.get("kunci") == nama and tersimpan.get("data")

        if siap_unduh:
            st.info(
                f"Data terkonfirmasi: ID {tersimpan['id']} — "
                f"{len(tersimpan['data']['produk'])} produk. Siap diunduh."
            )
            kol_unduh.download_button(
                "Unduh Hasil",
                data=buat_csv(tersimpan["data"]),
                file_name=f"transaksi_{Path(nama).stem}.csv",
                mime="text/csv",
                key=f"{kunci}_unduh",
            )
        else:
            kol_unduh.download_button(
                "Unduh Hasil",
                data=b"",
                file_name=f"transaksi_{Path(nama).stem}.csv",
                mime="text/csv",
                key=f"{kunci}_unduh",
                disabled=True,
                help="Simpan data dulu agar hasil yang terkonfirmasi bisa "
                     "diunduh.",
            )


RASIO_VALIDASI = 0.20  # mengikuti konstanta VALIDASI di scripts/latih_cnn.py


@st.cache_data(show_spinner=False)
def info_dataset_latih():
    """
    Menghitung informasi dataset langsung dari isi folder.

    Pengelompokan kelas mengikuti processing/dataset_loader.py
    (daftar_folder_kelas), sedangkan pengecualian data dan pembagian
    latih/validasi mengikuti scripts/latih_cnn.py. Dengan begitu angka
    pada halaman ini ikut berubah bila isi dataset berubah.
    """

    from collections import Counter
    import csv
    import json

    from processing.dataset_loader import EKSTENSI, daftar_folder_kelas

    dataset = BASE / "dataset"

    # Berkas yang ditandai bermasalah oleh scripts/validasi_dataset.py.
    bermasalah = set()
    status = Counter()
    laporan_ada = False

    csv_path = BASE / "output" / "laporan_dataset" / "hasil_validasi.csv"

    if csv_path.exists():
        laporan_ada = True
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for baris in csv.DictReader(f):
                nilai = (baris.get("status") or "").strip()
                status[nilai] += 1
                if nilai and nilai != "OK":
                    berkas = (baris.get("file") or "").replace("\\", "/")
                    bermasalah.add(berkas)

    per_kelas = Counter()
    per_kelas_bersih = Counter()
    kelompok = Counter()
    folder_kosong = []

    for folder, label in daftar_folder_kelas():

        if not folder.exists():
            folder_kosong.append(folder.name)
            continue

        for berkas in folder.iterdir():

            if not berkas.is_file():
                continue

            if berkas.suffix.lower() not in EKSTENSI:
                continue

            per_kelas[label] += 1
            kelompok[folder.parent.name] += 1

            relatif = berkas.relative_to(dataset).as_posix()

            if relatif in bermasalah:
                continue

            per_kelas_bersih[label] += 1

    total = sum(per_kelas.values())
    dipakai = sum(per_kelas_bersih.values())
    dikecualikan = total - dipakai

    # Tiru pembagian stratified per kelas pada scripts/latih_cnn.py.
    latih = 0
    validasi = 0

    for jumlah in per_kelas_bersih.values():

        if jumlah <= 0:
            continue

        n_validasi = max(1, int(round(jumlah * RASIO_VALIDASI)))
        validasi += n_validasi
        latih += jumlah - n_validasi

    kelas_model = 0
    label_path = BASE / "models" / "labels.json"

    if label_path.exists():
        try:
            kelas_model = len(
                json.loads(label_path.read_text(encoding="utf-8"))
            )
        except (ValueError, OSError):
            kelas_model = 0

    return {
        "total": total,
        "huruf": kelompok.get("uppercase", 0) + kelompok.get("lowercase", 0),
        "angka": kelompok.get("angka", 0),
        "simbol": kelompok.get("simbol", 0),
        "kelas_dataset": len(per_kelas),
        "kelas_model": kelas_model,
        "dikecualikan": dikecualikan,
        "dipakai": dipakai,
        "latih": latih,
        "validasi": validasi,
        "laporan_ada": laporan_ada,
        "status": dict(status),
        "folder_kosong": sorted(folder_kosong),
    }


def halaman_business_understanding():
    kepala_halaman(
        "Business Understanding",
        "Gambaran masalah, tujuan, alur sistem, dan data latih yang dipakai.",
    )

    # 1. Latar Belakang
    st.subheader("1. Latar Belakang")
    with st.container(border=True):
        st.write(
            "MERRI-STOCK digunakan untuk mengenali informasi pada nota "
            "transaksi CV Merriena Jaya. Nota yang sebelumnya dibaca dan "
            "diketik manual kini diubah menjadi teks oleh sistem, sehingga "
            "pengambilan data transaksi menjadi lebih cepat dan konsisten."
        )
        st.write(
            "Hasil pengenalan teks sangat dipengaruhi oleh kondisi gambar "
            "nota. Perbedaan kualitas gambar, tingkat kontras, serta bentuk "
            "karakter yang tidak seragam membuat karakter pada nota sulit "
            "dikenali. Karena itu kualitas gambar perlu diperbaiki sebelum "
            "proses pengenalan karakter dilakukan."
        )

    # 2. Tujuan Sistem
    st.subheader("2. Tujuan Sistem")
    with st.container(border=True):
        st.write(
            "Sistem ini bertujuan menganalisis pengaruh image preprocessing "
            "terhadap ketepatan pengenalan teks nota transaksi. Pengenalan "
            "karakter dilakukan dengan CNN dan Template Matching yang "
            "digabungkan dalam satu proses pengenalan karakter, lalu "
            "hasilnya dinilai per field pada halaman Evaluasi."
        )
        st.info(
            "Image preprocessing menjadi variabel yang dianalisis, sedangkan "
            "CNN dan Template Matching adalah metode pengenalan yang bekerja "
            "bersama pada satu proses yang sama."
        )

    # 3. Alur Kerja Sistem
    st.subheader("3. Alur Kerja Sistem")
    st.caption(
        "Urutan pengolahan satu nota dari gambar unggahan sampai data "
        "transaksi yang siap disimpan."
    )
    alur([
        "Upload Nota", "Preprocessing", "Deteksi Field",
        "Segmentasi Karakter", "Pengenalan", "Hasil Ekstraksi", "Evaluasi",
    ])

    tahap = [
        ("Upload Nota",
         "Gambar nota transaksi (jpg/png) dimasukkan ke aplikasi."),
        ("Image Preprocessing",
         "Ukuran gambar diseragamkan, warna diubah ke grayscale, lalu "
         "citra dibinerkan dengan Otsu agar tulisan terpisah dari latar."),
        ("Deteksi Field",
         "Sistem mencari lokasi bagian penting nota: nomor, penerima, "
         "proyek, produk, jumlah, dan harga satuan."),
        ("Segmentasi Karakter",
         "Teks tiap field dipisah per karakter memakai projection profile "
         "dan Dynamic Programming."),
        ("Pengenalan CNN + Template Matching",
         "Template Matching mencocokkan bentuk potongan dengan template, "
         "CNN mengenali tiap karakter hasil segmentasi."),
        ("Hasil Ekstraksi",
         "Hasil OCR diperiksa dan dikoreksi manual, lalu disimpan sebagai "
         "data transaksi beserta detail produknya."),
        ("Evaluasi",
         "Teks hasil OCR dibandingkan dengan ground truth per field, lalu "
         "diberi status Benar atau Salah."),
    ]

    for i in range(0, len(tahap), 3):
        kol = st.columns(3)
        for j, (nama, ket) in enumerate(tahap[i:i + 3]):
            with kol[j]:
                with st.container(border=True):
                    st.markdown(f"**{i + j + 1}. {nama}**")
                    st.caption(ket)

    with st.expander("Rincian metode pengenalan"):
        metode = [
            ("Projection", "Menghitung jumlah piksel tinta per kolom untuk "
                           "menemukan batas antar karakter."),
            ("Dynamic Programming", "Memilih kombinasi batas karakter terbaik "
                                    "dari banyak kemungkinan."),
            ("Template Matching", "Mencocokkan bentuk tiap karakter dengan "
                                  "template pada dataset untuk mengenalinya."),
            ("CNN", "Model pembelajaran mesin yang mengenali karakter "
                    "berdasarkan pola bentuk yang telah dilatih."),
        ]
        for nama, ket in metode:
            st.markdown(f"**{nama}** — {ket}")

    # 4. Informasi Data Latih
    st.subheader("4. Informasi Data Latih")

    info = info_dataset_latih()

    if not info["total"]:
        st.warning("Folder dataset tidak ditemukan atau masih kosong.")
    else:
        kol = st.columns(4)
        kol[0].metric("Total gambar karakter", f"{info['total']:,}")
        kol[1].metric("Kelas karakter model", info["kelas_model"])
        kol[2].metric("Data latih (80%)", f"{info['latih']:,}")
        kol[3].metric("Data validasi (20%)", f"{info['validasi']:,}")

        st.caption(
            f"Dataset berisi {info['total']:,} gambar karakter "
            f"({info['huruf']:,} huruf, {info['angka']:,} angka, "
            f"{info['simbol']:,} simbol) pada {info['kelas_model']} kelas. "
            f"Sebanyak {info['dipakai']:,} gambar dipakai untuk pelatihan, "
            f"dibagi menjadi {info['latih']:,} data latih dan "
            f"{info['validasi']:,} data validasi."
        )

    # 5. Enam field target
    st.subheader("5. Enam Field Target")
    st.caption(
        "Enam field berikut adalah informasi transaksi yang ingin diekstraksi "
        "dari nota. Setiap field dipotong dari nota, lalu teksnya dikenali "
        "satu per satu."
    )

    field = [
        ("Nomor Nota", "Nomor urut nota (contoh: 0299)"),
        ("Nama Penerima", "Nama pemesan atau perusahaan penerima barang"),
        ("Proyek", "Nama proyek atau lokasi pengiriman"),
        ("Nama Produk", "Nama barang pada baris transaksi"),
        ("Jumlah", "Banyaknya barang yang dibeli"),
        ("Harga Satuan", "Harga per satuan barang (contoh: 139.500)"),
    ]

    for i in range(0, len(field), 3):
        kol = st.columns(3)
        for j, (nama, ket) in enumerate(field[i:i + 3]):
            with kol[j]:
                with st.container(border=True):
                    st.markdown(f"**{nama}**")
                    st.caption(ket)


# Jenis field yang dinilai pada halaman Evaluasi, sesuai daftar field
# penelitian: nomor nota, nama penerima, proyek, nama produk, jumlah,
# dan harga satuan. Field "satuan" tidak termasuk daftar tersebut.
JENIS_DIEVALUASI = ("nomor", "penerima", "proyek", "produk", "jumlah", "harga")


def _isi_gt(kunci_widget, kunci_gt):
    """
    Gabungkan Ground Truth tersimpan dengan suntingan terbaru data_editor.

    Nilai GT disimpan pada kunci session_state tersendiri (bukan kunci
    widget), karena nilai widget tidak boleh diubah lewat st.session_state.
    Hasilnya dibaca SEBELUM tabel dirender, sehingga Status dan ringkasan
    ikut berubah pada rerun yang sama saat pengguna mengetik.

    Mengembalikan dict {indeks baris: teks Ground Truth}.
    """
    tersimpan = st.session_state.get(kunci_gt)
    isi = {}
    if isinstance(tersimpan, dict):
        isi = {int(k): str(v or "") for k, v in tersimpan.items()}

    state = st.session_state.get(kunci_widget)
    if isinstance(state, dict):
        for indeks, ubah in (state.get("edited_rows") or {}).items():
            if isinstance(ubah, dict) and "Ground Truth" in ubah:
                isi[int(indeks)] = str(ubah["Ground Truth"] or "")

    st.session_state[kunci_gt] = isi
    return isi


def _data_evaluasi(det, hasil, isi_gt):
    """
    Susun baris evaluasi dari hasil deteksi dan hasil OCR.

    Ground Truth TIDAK diambil otomatis dari file acuan: kolomnya dibiarkan
    kosong dan diisi sendiri oleh pengguna dengan membaca nota asli. Status
    baru ditentukan setelah Ground Truth terisi.

    Hasil OCR tidak pernah diubah di sini.
    """
    baris = []
    for f in det["fields"]:
        # Hanya enam jenis field yang dinilai; field lain dilewati.
        if f.get("jenis") not in JENIS_DIEVALUASI:
            continue

        t = (hasil.get(f["id"], {}) or {}).get("tanpa", {}) or {}
        teks = t.get("teks", "") or ""
        gt = str(isi_gt.get(len(baris), "") or "")

        # Status ditentukan dari perbandingan teks utuh: sama persis atau
        # tidak. Normalisasi hanya merapikan spasi, bukan mengubah huruf,
        # angka, atau tanda baca.
        gt_norm = normalisasi_teks(gt)
        if not gt_norm:
            status = "Belum diisi"
        elif gt_norm == normalisasi_teks(teks):
            status = "Benar"
        else:
            status = "Salah"

        baris.append({
            "Field": f["label"],
            "Ground Truth": gt,
            "Hasil OCR": teks or "(kosong)",
            "Status": status,
        })
    return baris


def halaman_evaluasi():
    kepala_halaman(
        "Evaluasi Hasil Pengenalan OCR",
        "Perbandingan hasil OCR dengan Ground Truth nota asli per field; "
        "status Benar bila seluruh isi field sama persis.",
    )

    if st.session_state.get("gambar_bgr") is None:
        st.info("Unggah nota di halaman Upload terlebih dahulu.")
        return

    det = ambil_deteksi()
    if det is None:
        st.info("Klik 'Proses Nota' di halaman Upload terlebih dahulu.")
        return

    hasil = st.session_state.get("ocr_semua")
    if not hasil:
        st.info("Buka halaman Pengenalan dulu agar hasil OCR tersedia.")
        return

    nama = st.session_state.get("nama_file", "")
    kunci_base = f"evaluasi_{nama}_tabel"
    kunci_gt = f"evaluasi_{nama}_gt"
    kunci_versi = f"evaluasi_{nama}_versi"

    # Versi dipakai agar tombol Kosongkan GT bisa memakai widget baru dengan
    # state kosong, tanpa mengubah nilai widget yang sudah dirender.
    versi = int(st.session_state.get(kunci_versi, 0) or 0)
    kunci_widget = f"{kunci_base}_v{versi}"

    st.subheader("Hasil evaluasi per field")
    st.caption(
        "Ground Truth diisi sendiri dengan mengetik isi field yang terbaca "
        "dari nota asli; Hasil OCR adalah keluaran sistem. Status Benar bila "
        "isinya sama persis."
    )
    legenda_status()

    isi_gt = _isi_gt(kunci_widget, kunci_gt)

    # Tombol dijalankan SEBELUM data_editor dibuat, karena state widget tidak
    # boleh diubah setelah widgetnya dirender pada run yang sama.
    kol_info, kol_reset = st.columns([4, 1])
    with kol_reset:
        if st.button(
            "Kosongkan GT",
            key=f"{kunci_base}_kosongkan",
            disabled=not any(normalisasi_teks(v) for v in isi_gt.values()),
            help="Hapus semua Ground Truth yang sudah diketik agar bisa "
                 "diisi ulang dari awal.",
        ):
            st.session_state[kunci_gt] = {}
            versi += 1
            st.session_state[kunci_versi] = versi
            kunci_widget = f"{kunci_base}_v{versi}"
            isi_gt = {}

    baris = _data_evaluasi(det, hasil, isi_gt)

    tabel = pd.DataFrame([
        {k: b[k] for k in ("Field", "Ground Truth", "Hasil OCR", "Status")}
        for b in baris
    ])

    st.data_editor(
        tabel,
        key=kunci_widget,
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        disabled=["Field", "Hasil OCR", "Status"],
        column_config={
            "Field": st.column_config.TextColumn("Field", width="medium"),
            "Ground Truth": st.column_config.TextColumn(
                "Ground Truth",
                width="large",
                help="Klik sel lalu ketik isi field sesuai nota asli. "
                     "Sel kosong berarti belum diisi.",
            ),
            "Hasil OCR": st.column_config.TextColumn(
                "Hasil OCR", width="large"
            ),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

    terisi = sum(1 for b in baris if normalisasi_teks(b["Ground Truth"]))
    if terisi:
        kol_info.info(
            f"{terisi} dari {len(baris)} field sudah diisi Ground Truth. "
            "Status dan ringkasan di bawah dihitung dari isian tersebut."
        )
    else:
        kol_info.caption(
            "Belum ada Ground Truth yang diisi. Klik sel pada kolom Ground "
            "Truth untuk mengisinya sesuai nota asli."
        )

    benar = sum(1 for b in baris if b["Status"] == "Benar")
    salah = sum(1 for b in baris if b["Status"] == "Salah")
    belum = sum(1 for b in baris if b["Status"] == "Belum diisi")

    dievaluasi = benar + salah
    akurasi = (benar / dievaluasi * 100.0) if dievaluasi else 0.0

    st.subheader("Ringkasan")
    st.caption(
        "Ringkasan dihitung dari exact match per field. Field yang Ground "
        "Truth-nya belum diisi tidak ikut dihitung."
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Field benar", benar)
    k2.metric("Field salah", salah)
    k3.metric("Persentase benar", f"{akurasi:.2f}%")

    if belum:
        st.info(
            f"{belum} field belum diisi Ground Truth sehingga tidak dihitung "
            "dalam ringkasan. Isi kolom Ground Truth pada tabel di atas."
        )


# ---------------------------------------------------------------- Routing
# Nama halaman disimpan sebagai konstanta dan dipakai untuk DAFTAR menu
# maupun untuk routing. Sebelumnya keduanya ditulis terpisah sehingga
# pernah tidak cocok ("Segmentasi Karakter" di menu vs "Segmentasi" di
# routing) dan halaman menjadi kosong tanpa pesan apa pun.
HALAMAN_BU = "Business Understanding"
HALAMAN_UPLOAD = "Upload Nota"
HALAMAN_PREPROSES = "Preprocessing"
HALAMAN_SEGMEN = "Segmentasi Karakter"
HALAMAN_KENAL = "Pengenalan Karakter"
HALAMAN_EKSTRAKSI = "Hasil Ekstraksi"
HALAMAN_EVAL = "Evaluasi"

HALAMAN = [
    HALAMAN_BU,
    HALAMAN_UPLOAD,
    HALAMAN_PREPROSES,
    HALAMAN_SEGMEN,
    HALAMAN_KENAL,
    HALAMAN_EKSTRAKSI,
    HALAMAN_EVAL,
]

# ---------------------------------------------------------------- Sidebar
sidebar_brand()
sidebar_label_menu("Menu")

halaman = st.sidebar.radio(
    "Halaman", HALAMAN, key="navigasi", label_visibility="collapsed"
)

sidebar_status(st.session_state.get("nama_file"))

if halaman == HALAMAN_BU:
    halaman_business_understanding()
elif halaman == HALAMAN_UPLOAD:
    halaman_upload()
elif halaman == HALAMAN_PREPROSES:
    halaman_preprocessing()
elif halaman == HALAMAN_SEGMEN:
    halaman_segmentasi()
elif halaman == HALAMAN_KENAL:
    halaman_pengenalan()
elif halaman == HALAMAN_EKSTRAKSI:
    halaman_hasil_ekstraksi()
elif halaman == HALAMAN_EVAL:
    halaman_evaluasi()
else:
    # Jaring pengaman: bila nama halaman tidak dikenali, tampilkan pesan
    # agar tidak pernah lagi muncul halaman kosong tanpa penjelasan.
    st.error(
        f"Halaman {halaman!r} tidak dikenali. Pilih halaman lain pada menu."
    )
