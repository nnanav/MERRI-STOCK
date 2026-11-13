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
                "cara": "intensitas",
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
            "cara": "tepi",
            "pesan": "Batas kertas ditemukan dari peta tepi (Canny).",
        }

    # Strategi 3: bila kertas dan latar sama-sama terang sehingga tepinya
    # tidak terpisah, pakai batas isi nota (blok teks) plus margin sebagai
    # pengganti batas kertas, sekaligus memperkirakan kemiringannya.
    temu_isi = _kotak_dari_isi(gambar)
    if temu_isi is not None:
        kotak, sudut = temu_isi
        return {
            "kotak": kotak,
            "status": "ketemu",
            "cara": "isi",
            "pesan": (
                f"Tepi kertas tidak terpisah dari latar; batas crop diambil "
                f"dari area isi nota plus margin dan kemiringan {sudut:.1f}° "
                "diluruskan."
            ),
        }

    return {
        "kotak": None,
        "status": "tidak_ketemu",
        "cara": None,
        "pesan": "Batas kertas tidak dapat dipisahkan dari latar.",
    }


def _bersihkan_latar(crop):
    """
    Putihkan sisa latar yang menempel pada crop.

    Wilayah yang jauh lebih gelap dari kertas (meja, bayangan) atau
    berwarna pekat (strip folder) dan membentuk komponen besar yang
    menyentuh tepi crop dianggap latar, lalu diganti putih. Tinta tulisan
    tidak tersentuh karena komponennya kecil dan tidak menempel tepi.
    """
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return crop

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    tengah = hsv[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]
    if tengah.size == 0:
        return crop
    v_ref = float(np.median(tengah[..., 2]))

    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    # Latar = abu-abu sedang (meja/bayangan) atau berwarna pekat (strip).
    # Tinta tulisan jauh lebih gelap sehingga tidak ikut terpilih.
    mask = (
        ((v > 0.45 * v_ref) & (v < 0.85 * v_ref)) | (s > 45)
    ).astype(np.uint8)

    # Satukan piksel latar yang berdekatan agar menjadi komponen utuh.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        menyentuh = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
        if menyentuh and area >= 0.0015 * h * w:
            crop[lab == i] = (255, 255, 255)
    return crop



def _trim_latar(crop, maks=0.22):
    """
    Pangkas pita latar yang masih menempel di tepi crop.

    Sisi crop digeser ke dalam selama pita tepinya terlihat seperti latar:
    jauh lebih gelap dari kertas (meja abu-abu, bayangan) atau berwarna
    pekat (strip folder). Kertas putih/terang tidak ikut terpotong.
    Mengembalikan (crop_terpotong, (kiri, atas, kanan, bawah)).
    """
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return crop, (0, 0, 0, 0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    tengah = hsv[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]
    if tengah.size == 0:
        return crop, (0, 0, 0, 0)
    v_ref = float(np.median(tengah[..., 2]))

    langkah = max(2, int(0.005 * min(h, w)))

    def buruk(band):
        if band.size == 0:
            return False
        v = float(np.median(band[..., 2]))
        s = float(np.median(band[..., 1]))
        return v < 0.82 * v_ref or s > 35

    x0 = y0 = 0
    x1, y1 = w, h
    while x0 < int(maks * w) and buruk(hsv[y0:y1, x0:x0 + langkah]):
        x0 += langkah
    while x1 > w - int(maks * w) and buruk(hsv[y0:y1, x1 - langkah:x1]):
        x1 -= langkah
    while y0 < int(maks * h) and buruk(hsv[y0:y0 + langkah, x0:x1]):
        y0 += langkah
    while y1 > h - int(maks * h) and buruk(hsv[y1 - langkah:y1, x0:x1]):
        y1 -= langkah

    if (x0, y0, x1, y1) == (0, 0, w, h):
        return crop, (0, 0, 0, 0)
    return crop[y0:y1, x0:x1].copy(), (x0, y0, w - x1, h - y1)


def _kotak_dari_isi(gambar):
    """
    Perkirakan kotak nota dari isi cetakannya.

    Teks dan garis nota digabung menjadi blok; kotak pembungkus isi
    diambil, lalu kemiringan cetak diperkirakan lewat profil proyeksi
    baris (sudut yang membuat baris teks paling rapi). Kotak sumber
    dibuat sebagai kotak isi yang diputar sebesar kemiringan tersebut,
    sehingga warp berikutnya meluruskan cetak di tempatnya.
    """
    tinggi, lebar = gambar.shape[:2]
    luas = tinggi * lebar

    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    abu = cv2.GaussianBlur(abu, (3, 3), 0)

    # Ambang adaptif tahan terhadap pencahayaan yang tidak merata.
    tinta = cv2.adaptiveThreshold(
        abu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10,
    )
    tinta = cv2.morphologyEx(
        tinta, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    # Gabungkan karakter menjadi blok baris, lalu antarbaris menyatu.
    blok = cv2.dilate(
        tinta, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)),
        iterations=2,
    )
    blok = cv2.morphologyEx(
        blok, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 35)),
        iterations=2,
    )

    kontur, _ = cv2.findContours(blok, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    berarti = []
    for c in kontur:
        if cv2.contourArea(c) < 0.002 * luas:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Buang artefak tepi scan: pita vertikal tipis (mis. tepi folder
        # atau strip warna) yang menjulang setinggi halaman bukan isi nota.
        if h > 0.50 * tinggi and w < 0.20 * lebar:
            continue
        # Buang blok berwarna pekat (strip folder) yang bukan isi cetak.
        if cv2.mean(hsv[y:y + h, x:x + w])[1] > 40:
            continue
        berarti.append(c)
    if not berarti:
        return None

    x_min = min(cv2.boundingRect(c)[0] for c in berarti)
    y_min = min(cv2.boundingRect(c)[1] for c in berarti)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]
                for c in berarti)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]
                for c in berarti)

    if (x_max - x_min) < 0.30 * lebar or (y_max - y_min) < 0.30 * tinggi:
        return None

    sudut = _estimasi_sudut(abu, (x_min, y_min, x_max, y_max))
    if sudut is None:
        return None

    pusat = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    pojok = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype="float32",
    )
    m = cv2.getRotationMatrix2D(pusat, -sudut, 1.0)
    pojok_sumber = pojok @ m[:, :2].T + m[:, 2]
    return _urutkan_titik(pojok_sumber), sudut


def _estimasi_sudut(abu, bbox, kandidat=None):
    """
    Cari kemiringan cetak (derajat) lewat profil proyeksi baris.

    Sudut terbaik membuat tinta terkonsentrasi pada baris-baris tertentu,
    sehingga ragam profil proyeksi baris maksimum.
    """
    x_min, y_min, x_max, y_max = bbox
    area = abu[y_min:y_max, x_min:x_max]
    if area.size == 0:
        return None
    if kandidat is None:
        kandidat = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    _, biner = cv2.threshold(
        area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    tinggi, lebar = biner.shape[:2]
    pusat = (lebar / 2.0, tinggi / 2.0)

    terbaik, skor_terbaik = None, -1.0
    for sudut in kandidat:
        m = cv2.getRotationMatrix2D(pusat, sudut, 1.0)
        putar = cv2.warpAffine(
            biner, m, (lebar, tinggi),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Ambil bagian tengah agar tepi hasil rotasi tidak ikut dihitung.
        inti = putar[int(0.1 * tinggi):int(0.9 * tinggi),
                     int(0.1 * lebar):int(0.9 * lebar)]
        profil = np.count_nonzero(inti, axis=1).astype(np.float64)
        skor = float(np.var(profil))
        if skor > skor_terbaik:
            skor_terbaik, terbaik = skor, sudut
    return terbaik


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
    cara = det.get("cara")

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

    # Margin lebih longgar bila kotak berasal dari isi nota, supaya tepi
    # kertas dan elemen tepi (logo, tanda tangan) tidak terpotong.
    faktor = 0.06 if cara == "isi" else 0.01
    kotak = _longgarkan(kotak, gambar, faktor)
    if cara == "isi":
        # Batas isi hanya dipakai untuk memotong dan membersihkan latar;
        # cetak tidak diputar agar posisi baris tetap sama dengan yang
        # diharapkan oleh deteksi field.
        x_min = int(max(0, np.floor(kotak[:, 0].min())))
        y_min = int(max(0, np.floor(kotak[:, 1].min())))
        x_max = int(min(lebar, np.ceil(kotak[:, 0].max())))
        y_max = int(min(tinggi, np.ceil(kotak[:, 1].max())))
        kanvas = gambar.copy()
        kanvas[:y_min, :] = 255
        kanvas[y_max:, :] = 255
        kanvas[:, :x_min] = 255
        kanvas[:, x_max:] = 255

        return {
            "gambar": kanvas,
            "crop": kanvas[y_min:y_max, x_min:x_max].copy(),
            "bbox": (x_min, y_min, x_max, y_max),
            "kotak": kotak,
            "status": "crop_saja",
            "pesan": (
                "Area isi nota dipotong dan latar di luarnya diganti putih; "
                "cetak tidak diputar agar posisi baris tetap konsisten."
            ),
        }


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
    crop_base = potong.get("crop", potong["gambar"])

    if bbox is None:
        # Fallback: batas kertas tidak terpisah dari latar, sehingga
        # peningkatan kualitas dilewati agar clutter latar tidak ikut
        # direntangkan menjadi noise.
        hasil = potong["gambar"]
        crop_tampil = crop_base
        tampil = hasil
    else:
        x_min, y_min, x_max, y_max = bbox
        kertas = tingkatkan_kualitas(crop_base)

        # Gambar untuk proses: isi yang sudah ditingkatkan dikembalikan ke
        # kanvas berukuran asli pada posisi semula. Posisi baris terhadap
        # kanvas harus tetap sama karena deteksi field memakai fraksi
        # tinggi/lebar gambar; area di luar kertas sudah putih.
        hasil = potong["gambar"].copy()
        hasil[y_min:y_max, x_min:x_max] = kertas

        # Gambar untuk tampilan: crop ketat yang dirapikan (sisa latar di
        # tepinya dibuang) agar nota terlihat terpotong penuh.
        crop_tampil = _bersihkan_latar(_trim_latar(crop_base)[0])
        tampil = _bersihkan_latar(_trim_latar(kertas)[0])

    return {
        "asli": gambar.copy(),
        "crop": crop_tampil,
        "hasil": hasil,
        "tampil": tampil,
        "kotak": potong["kotak"],
        "status": potong["status"],
        "pesan": potong["pesan"],
        "metode": METODE_PENINGKATAN,
    }

