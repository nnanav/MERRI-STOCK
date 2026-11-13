"""
Preprocessing gambar nota: crop area kertas dan peningkatan kualitas.

Modul ini hanya menangani persiapan gambar SEBELUM deteksi field.
Segmentasi karakter, pengenalan (CNN + Template Matching), dan evaluasi
tidak disentuh oleh modul ini.

Alur:
    gambar unggahan -> deteksi batas kertas -> crop + koreksi perspektif
    -> peningkatan kualitas (satu metode terpilih) -> hasil dipakai tahap
    berikutnya.
"""

import cv2
import numpy as np


# =========================================
# DETEKSI BATAS KERTAS
# =========================================

def _urutkan_titik(kotak):
    """Urutkan 4 titik menjadi (kiri-atas, kanan-atas, kanan-bawah, kiri-bawah)."""
    titik = np.asarray(kotak, dtype="float32").reshape(4, 2)
    jumlah = titik.sum(axis=1)
    selisih = np.diff(titik, axis=1).ravel()

    urut = np.zeros((4, 2), dtype="float32")
    urut[0] = titik[np.argmin(jumlah)]
    urut[2] = titik[np.argmax(jumlah)]
    urut[1] = titik[np.argmin(selisih)]
    urut[3] = titik[np.argmax(selisih)]
    return urut


def _ukuran_target(titik):
    """Lebar dan tinggi persegi panjang tujuan dari 4 titik sudut."""
    (tl, tr, br, bl) = titik
    lebar = max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    )
    tinggi = max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    )
    return int(round(lebar)), int(round(tinggi))


def deteksi_batas_kertas(gambar):
    """
    Mencari empat sudut kertas nota pada gambar.

    Dua strategi dipakai berurutan:
      1. Pemisahan intensitas (Otsu): cocok bila latar lebih gelap dari
         kertas (mis. meja abu-abu), sehingga kertas menjadi satu wilayah
         terang yang utuh.
      2. Tepi Canny: cadangan bila kontras latar-kertas rendah.

    Mengembalikan dict berisi:
      kotak  : ndarray (4, 2) titik sudut atau None bila tidak ditemukan
      status : "ketemu" | "tidak_ketemu"
      pesan  : keterangan diagnostik
    """
    tinggi, lebar = gambar.shape[:2]
    luas = tinggi * lebar

    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    abu = cv2.GaussianBlur(abu, (5, 5), 0)

    # Strategi 1: kertas = wilayah terang yang dipisah Otsu dari latar.
    _, biner = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    porsi_terang = float(np.count_nonzero(biner)) / luas

    if 0.20 < porsi_terang < 0.92:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bersih = cv2.morphologyEx(biner, cv2.MORPH_CLOSE, kernel,
                                  iterations=3)
        bersih = cv2.morphologyEx(bersih, cv2.MORPH_OPEN, kernel,
                                  iterations=1)
        kotak = _kotak_dari_mask(bersih, luas)
        if kotak is not None:
            return {
                "kotak": kotak,
                "status": "ketemu",
                "pesan": "Batas kertas dipisahkan dari latar berdasarkan "
                         "intensitas (Otsu).",
            }

    # Strategi 2: tepi Canny untuk latar yang kontrasnya rendah.
    tepi = cv2.Canny(abu, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tepi = cv2.dilate(tepi, kernel, iterations=2)
    tepi = cv2.morphologyEx(tepi, cv2.MORPH_CLOSE, kernel, iterations=2)
    kotak = _kotak_dari_mask(tepi, luas)
    if kotak is not None:
        return {
            "kotak": kotak,
            "status": "ketemu",
            "pesan": "Batas kertas ditemukan dari peta tepi (Canny).",
        }

    return {
        "kotak": None,
        "status": "tidak_ketemu",
        "pesan": "Batas kertas tidak dapat dipisahkan dari latar.",
    }


def _kotak_dari_mask(mask, luas):
    """Ambil segi empat terbesar dari mask biner; None bila tidak layak."""
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return None

    kontur = max(kontur, key=cv2.contourArea)
    area = cv2.contourArea(kontur)
    if area < 0.20 * luas:
        return None

    peri = cv2.arcLength(kontur, True)
    aproks = cv2.approxPolyDP(kontur, 0.02 * peri, True)
    if len(aproks) == 4:
        return _urutkan_titik(aproks.reshape(4, 2))

    rect = cv2.minAreaRect(kontur)
    return _urutkan_titik(cv2.boxPoints(rect))


def _longgarkan(titik, gambar, faktor=0.01):
    """
    Geser keempat sudut sedikit ke luar agar tepi dan isi nota tidak
    terpotong saat warp.
    """
    tinggi, lebar = gambar.shape[:2]
    pusat = titik.mean(axis=0)
    longgar = pusat + (titik - pusat) * (1.0 + faktor)
    longgar[:, 0] = np.clip(longgar[:, 0], 0, lebar - 1)
    longgar[:, 1] = np.clip(longgar[:, 1], 0, tinggi - 1)
    return longgar.astype("float32")
def crop_dan_luruskan(gambar):
    """
    Crop area kertas nota dan luruskan bila miring/perspektif.

    Mengembalikan dict:
      gambar : hasil crop (atau gambar asli bila deteksi gagal)
      kotak  : titik sudut yang dipakai atau None
      status : "koreksi_perspektif" | "gagal_pakai_asli"
      pesan  : keterangan diagnostik
    """
    tinggi, lebar = gambar.shape[:2]
    det = deteksi_batas_kertas(gambar)
    kotak = det["kotak"]

    if kotak is None:
        return {
            "gambar": gambar.copy(),
            "crop": gambar.copy(),
            "bbox": None,
            "kotak": None,
            "status": "gagal_pakai_asli",
            "pesan": "Deteksi batas kertas gagal; gambar asli dipakai. "
                     + det["pesan"],
        }

    kotak = _longgarkan(kotak, gambar)
    w_target, h_target = _ukuran_target(kotak)

    # Penjagaan agar isi nota tidak hilang: hasil warp harus masuk akal
    # ukurannya dibanding gambar sumber.
    if (
        w_target < 0.40 * lebar
        or h_target < 0.40 * tinggi
        or w_target < 8 or h_target < 8
    ):
        return {
            "gambar": gambar.copy(),
            "crop": gambar.copy(),
            "bbox": None,
            "kotak": None,
            "status": "gagal_pakai_asli",
            "pesan": (
                f"Ukuran hasil warp tidak masuk akal "
                f"({w_target}x{h_target}); gambar asli dipakai."
            ),
        }

    rasio = w_target / max(1, h_target)
    if not 0.5 <= rasio <= 3.0:
        return {
            "gambar": gambar.copy(),
            "crop": gambar.copy(),
            "bbox": None,
            "kotak": None,
            "status": "gagal_pakai_asli",
            "pesan": (
                f"Rasio sisi hasil warp tidak wajar ({rasio:.2f}); "
                "gambar asli dipakai."
            ),
        }

    # Warp kertas ke posisi tegak TANPA memindahkannya: keempat sudut
    # dipetakan ke persegi panjang seukuran kotak pembungkusnya di kanvas
    # asli. Latar di luar kertas menjadi putih dan kemiringan hilang,
    # tetapi posisi serta proporsi isi nota terhadap kanvas tetap sama,
    # sehingga parameter deteksi field yang memakai fraksi tinggi/lebar
    # gambar tidak berubah perilaku.
    x_min = int(max(0, np.floor(kotak[:, 0].min())))
    y_min = int(max(0, np.floor(kotak[:, 1].min())))
    x_max = int(min(lebar, np.ceil(kotak[:, 0].max())))
    y_max = int(min(tinggi, np.ceil(kotak[:, 1].max())))

    matriks = cv2.getPerspectiveTransform(
        kotak,
        np.array(
            [[x_min, y_min], [x_max - 1, y_min],
             [x_max - 1, y_max - 1], [x_min, y_max - 1]],
            dtype="float32",
        ),
    )
    kanvas = cv2.warpPerspective(
        gambar, matriks, (lebar, tinggi),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    # warpPerspective ikut memetakan latar di luar kertas, jadi area di
    # luar persegi panjang kertas ditimpa putih agar benar-benar hanya
    # kertas yang tersisa.
    kanvas[:y_min, :] = 255
    kanvas[y_max:, :] = 255
    kanvas[:, :x_min] = 255
    kanvas[:, x_max:] = 255
    crop = kanvas[y_min:y_max, x_min:x_max].copy()

    return {
        "gambar": kanvas,
        "crop": crop,
        "bbox": (x_min, y_min, x_max, y_max),
        "kotak": kotak,
        "status": "koreksi_perspektif",
        "pesan": (
            "Kertas nota diluruskan dan latar di luarnya diganti putih; "
            "crop ketat kertas tersedia untuk tampilan."
        ),
    }


# =========================================
# PENINGKATAN KUALITAS GAMBAR
# =========================================
# Ketiga kandidat diterapkan pada kanal L (ruang LAB) supaya hue warna
# tetap utuh; dengan begitu mask merah nomor nota tidak berubah perilaku.

def _pisah_lab(gambar):
    lab = cv2.cvtColor(gambar, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l, a, b


def _gabung_lab(a, b, l_baru):
    return cv2.cvtColor(cv2.merge((l_baru, a, b)), cv2.COLOR_LAB2BGR)


def contrast_stretching(gambar):
    """Regangan kontras: rentang intensitas ditarik ke 0-255."""
    l, a, b = _pisah_lab(gambar)
    lo = float(np.percentile(l, 1))
    hi = float(np.percentile(l, 99))
    if hi - lo < 1:
        return gambar.copy()
    l_baru = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return _gabung_lab(a, b, l_baru.astype(np.uint8))


def histogram_equalization(gambar):
    """Perataan histogram global pada kanal luminansi."""
    l, a, b = _pisah_lab(gambar)
    return _gabung_lab(a, b, cv2.equalizeHist(l))


def clahe(gambar):
    """Adaptive histogram equalization (CLAHE) pada kanal luminansi."""
    l, a, b = _pisah_lab(gambar)
    l_baru = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return _gabung_lab(a, b, l_baru)


# Metode terpilih: Contrast Stretching (normalisasi persentil) pada kanal L.
#
# Alasan pemilihan (diukur pada area kertas nota CV Merriena Jaya):
# - Contrast Stretching menaikkan pemisahan teks-latar pada semua nota
#   (mis. 143 -> 163 dan 185 -> 200 pada skala 0-255) sehingga tulisan
#   tipis lebih terbaca, dengan kenaikan noise latar yang kecil.
# - CLAHE menurunkan pemisahan teks-latar pada nota berlatar terang
#   (185 -> 130) karena penyetaraan per-ubin mengangkat tekstur kertas
#   dan tembusan tulisan belakang menjadi seperti tinta.
# - Histogram Equalization meratakan histogram global sehingga tekstur
#   latar ikut dikuatkan drastis (noise latar naik ~3x) dan binerisasi
#   karakter menjadi tidak stabil.
# Metode diterapkan hanya pada area kertas yang terdeteksi; bila batas
# kertas tidak ditemukan (fallback), gambar dipakai apa adanya agar
# clutter latar tidak ikut direntangkan.
METODE_PENINGKATAN = "Contrast Stretching"


def tingkatkan_kualitas(gambar):
    """Terapkan satu metode peningkatan kualitas yang terpilih."""
    return contrast_stretching(gambar)


# =========================================
# ALUR LENGKAP
# =========================================

def preprocess_nota(gambar):
    """
    Jalankan alur preprocessing lengkap.

    Mengembalikan dict:
      asli   : gambar unggahan apa adanya
      crop   : hasil crop + koreksi perspektif (sebelum peningkatan)
      hasil  : gambar akhir yang diteruskan ke deteksi field
      kotak  : titik sudut kertas atau None
      status : status deteksi kertas
      pesan  : keterangan diagnostik
      metode : nama metode peningkatan kualitas
    """
    potong = crop_dan_luruskan(gambar)
    bbox = potong.get("bbox")

    if bbox is None:
        # Fallback: batas kertas tidak terpisah dari latar, sehingga
        # peningkatan kualitas dilewati agar clutter latar tidak ikut
        # direntangkan menjadi noise.
        hasil = potong["gambar"]
    else:
        x_min, y_min, x_max, y_max = bbox
        hasil = potong["gambar"].copy()
        hasil[y_min:y_max, x_min:x_max] = tingkatkan_kualitas(potong["crop"])

    return {
        "asli": gambar.copy(),
        "crop": potong.get("crop", potong["gambar"]),
        "hasil": hasil,
        "kotak": potong["kotak"],
        "status": potong["status"],
        "pesan": potong["pesan"],
        "metode": METODE_PENINGKATAN,
    }

