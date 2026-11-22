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


def _trim_latar(crop, maks=0.22):
    """
    Pangkas latar yang masih menempel di tepi crop (dua iterasi).

    Iterasi kedua perlu karena sudut kertas yang miring sering menyisakan
    latar tipis di dua sisi setelah pemangkasan pertama.

    Mengembalikan (crop_terpotong, (kiri, atas, kanan, bawah)) dengan
    offset dihitung terhadap crop masukan.
    """
    h0, w0 = crop.shape[:2]
    sisa, (kx, ky, kx2, ky2) = _trim_sekali(crop, maks)
    if (kx, ky, kx2, ky2) == (0, 0, 0, 0):
        return sisa, (0, 0, 0, 0)

    sisa2, (jx, jy, jx2, jy2) = _trim_sekali(sisa, maks)
    return sisa2, (kx + jx, ky + jy, kx2 + jx2, ky2 + jy2)


def _trim_sekali(crop, maks=0.22):
    """
    Satu kali pemangkasan latar tepi.

    Bekerja per kolom (sisi kiri/kanan) dan per baris (sisi atas/bawah),
    bukan per pita lebar, karena latar sering hanya menutupi sebagian
    tepi. Sebuah kolom/baris dianggap latar bila kecerahannya jelas di
    bawah kertas (atau berwarna pekat seperti strip folder) dan bukan
    bagian dari teks.

    Pemangkasan dibatasi `maks` dari lebar/tinggi agar isi nota tidak
    hilang. Mengembalikan (crop_terpotong, (kiri, atas, kanan, bawah)).
    """
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return crop, (0, 0, 0, 0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)

    tengah = v[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]
    if tengah.size == 0:
        return crop, (0, 0, 0, 0)
    v_ref = float(np.median(tengah))

    # Peta tinta: piksel yang jauh lebih gelap dari kertas.
    abu = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    tinta = (abu < 0.72 * v_ref).astype(np.uint8)

    prof_v_kol = np.median(v, axis=0)
    prof_s_kol = np.median(s, axis=0)
    prof_t_kol = tinta.mean(axis=0)

    prof_v_bar = np.median(v, axis=1)
    prof_s_bar = np.median(s, axis=1)
    prof_t_bar = tinta.mean(axis=1)

    def latar(vv, ss, tt):
        # Penilaian utama memakai kecerahan: kolom/baris latar jauh lebih
        # gelap dari kertas. Tinta hanya dipakai sebagai penjaga terakhir
        # dengan ambang tinggi, karena bayangan tepi kertas juga terhitung
        # sebagai piksel gelap sehingga tidak boleh memblokir pemangkasan.
        if tt > 0.60:
            return False
        if vv < 0.96 * v_ref:
            return True
        return ss > 30 and vv < 0.995 * v_ref

    x0, x1 = 0, w
    while x0 < int(maks * w) and latar(
        prof_v_kol[x0], prof_s_kol[x0], prof_t_kol[x0]
    ):
        x0 += 1
    while x1 > w - int(maks * w) and latar(
        prof_v_kol[x1 - 1], prof_s_kol[x1 - 1], prof_t_kol[x1 - 1]
    ):
        x1 -= 1

    y0, y1 = 0, h
    while y0 < int(maks * h) and latar(
        prof_v_bar[y0], prof_s_bar[y0], prof_t_bar[y0]
    ):
        y0 += 1
    while y1 > h - int(maks * h) and latar(
        prof_v_bar[y1 - 1], prof_s_bar[y1 - 1], prof_t_bar[y1 - 1]
    ):
        y1 -= 1

    # Jangan sampai hasil crop terlalu kecil.
    if (x1 - x0) < 0.5 * w or (y1 - y0) < 0.5 * h:
        return crop, (0, 0, 0, 0)

    if (x0, y0, x1, y1) == (0, 0, w, h):
        return crop, (0, 0, 0, 0)

    potong = crop[y0:y1, x0:x1]

    # Penjaga isi: bila pemangkasan ikut membuang tinta (teks) lebih dari
    # 5 persen, pemangkasan dibatalkan agar isi nota tidak hilang.
    t0 = int(np.count_nonzero(abu < 128))
    t1 = int(np.count_nonzero(
        cv2.cvtColor(potong, cv2.COLOR_BGR2GRAY) < 128
    ))
    if t0 > 0 and (t0 - t1) / t0 > 0.05:
        return crop, (0, 0, 0, 0)

    return potong.copy(), (x0, y0, w - x1, h - y1)


def _putihkan_latar_tepi(crop, toleransi=0.93):
    """
    Ganti latar yang menempel di tepi crop menjadi putih.

    Dipakai setelah warp perspektif, karena sudut kertas yang miring
    membuat latar ikut terpetakan ke dalam kotak crop sebagai pita tipis
    di tepi. Latar dikenali sebagai piksel yang jelas lebih gelap dari
    kertas DAN tersambung ke tepi gambar, sehingga tulisan di bagian
    tengah nota tidak pernah ikut diputihkan.

    Geometri gambar tidak diubah; hanya warna latar tepi.
    """
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return crop

    abu = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    tengah = abu[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]
    if tengah.size == 0:
        return crop
    v_ref = float(np.median(tengah))
    if v_ref < 40:
        return crop

    gelap = (abu < toleransi * v_ref).astype(np.uint8)
    if not gelap.any():
        return crop

    # Beri label komponen, lalu putihkan komponen yang menyentuh tepi.
    n, label = cv2.connectedComponents(gelap, connectivity=8)
    if n <= 1:
        return crop

    tepi = np.concatenate([
        label[0, :], label[-1, :], label[:, 0], label[:, -1],
    ])
    kandidat = set(int(x) for x in np.unique(tepi) if x != 0)
    if not kandidat:
        return crop

    # Batasi agar tidak memutihkan hampir seluruh gambar.
    area = np.isin(label, list(kandidat))
    if area.mean() > 0.45:
        return crop

    hasil = crop.copy()
    hasil[area] = (255, 255, 255)
    return hasil


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

    # Batas isi diambil dari seluruh piksel tinta, bukan dari blok yang
    # disaring, agar baris kecil (mis. "Nomor Nota") dan tanda tangan di
    # bagian bawah tetap masuk ke crop.
    kolom = np.count_nonzero(tinta, axis=0)
    baris = np.count_nonzero(tinta, axis=1)
    # Pita vertikal setinggi halaman adalah artefak scan (tepi folder),
    # bukan isi nota, jadi tidak dipakai menentukan batas.
    kolom[kolom > 0.50 * tinggi] = 0

    xs = np.nonzero(kolom > max(2, int(0.002 * tinggi)))[0]
    ys = np.nonzero(baris > max(2, int(0.002 * lebar)))[0]
    if xs.size == 0 or ys.size == 0:
        return None

    x_min, x_max = int(xs[0]), int(xs[-1]) + 1
    y_min, y_max = int(ys[0]), int(ys[-1]) + 1

    # Margin kecil supaya tepi huruf paling luar tidak terpotong.
    mx = max(3, int(0.02 * (x_max - x_min)))
    my = max(3, int(0.02 * (y_max - y_min)))
    x_min = max(0, x_min - mx)
    x_max = min(lebar, x_max + mx)
    y_min = max(0, y_min - my)
    y_max = min(tinggi, y_max + my)

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
        # Kertas dan latar sama-sama putih (scan berlatar putih), sehingga
        # batas kertas dicari dari garis tipis tepi kertas: puncak gradien
        # kolom di sepertiga kiri gambar. Crop dimulai tepat di garis itu
        # supaya latar putih di kiri tidak ikut, tanpa memotong isi nota.
        abu_f = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad = np.abs(cv2.Sobel(abu_f, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        tengah = int(0.35 * lebar)
        kiri = int(np.argmax(grad[:tengah]))
        v_kol = float(abu_f[:, kiri].mean())
        v_ref = float(np.median(abu_f[int(0.3 * tinggi):int(0.7 * tinggi),
                                       int(0.3 * lebar):int(0.7 * lebar)]))
        if grad[kiri] > max(30.0, 6.0 * float(np.median(grad))) \
                and v_kol < 0.94 * v_ref:
            x_min = max(0, kiri - 2)
        else:
            x_min = 0

        # Sisi lain mengikuti batas kertas yang terdeteksi (kotak), bukan
        # seluruh tinggi/lebar foto. Sebelumnya y_max dipaksa sama dengan
        # tinggi gambar, sehingga pita latar di bawah kertas ikut masuk dan
        # hasil crop tampak melenceng ke atas.
        y_min = int(max(0, np.floor(kotak[:, 1].min())))
        y_max = int(min(tinggi, np.ceil(kotak[:, 1].max())))
        x_max = int(min(lebar, np.ceil(kotak[:, 0].max())))

        # Jaga agar hasil crop tetap wajar; bila batas kertas terlalu kecil,
        # pakai seluruh tinggi/lebar gambar seperti sebelumnya.
        if (x_max - x_min) < 0.30 * lebar:
            x_min, x_max = 0, lebar
        if (y_max - y_min) < 0.30 * tinggi:
            y_min, y_max = 0, tinggi

        kanvas = gambar.copy()

        return {
            "gambar": kanvas,
            "crop": kanvas[y_min:y_max, x_min:x_max].copy(),
            "bbox": (x_min, y_min, x_max, y_max),
            "kotak": kotak,
            "status": "crop_saja",
            "pesan": (
                "Kertas dan latar sama-sama putih; crop mengikuti garis tepi "
                "kertas yang terdeteksi."
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
        trim_tampil = (0, 0, 0, 0)
    else:
        x_min, y_min, x_max, y_max = bbox
        kertas = tingkatkan_kualitas(crop_base)

        # Gambar untuk proses: isi yang sudah ditingkatkan dikembalikan ke
        # kanvas berukuran asli pada posisi semula. Posisi baris terhadap
        # kanvas harus tetap sama karena deteksi field memakai fraksi
        # tinggi/lebar gambar; area di luar kertas sudah putih.
        hasil = potong["gambar"].copy()
        hasil[y_min:y_max, x_min:x_max] = kertas

        # Gambar untuk tampilan: crop ketat yang dirapikan tepinya saja.
        # Tidak ada piksel isi yang diubah/dihapus. Setelah dipangkas,
        # sisa latar yang miring (tidak full-height) diputihkan agar tepi
        # crop benar-benar bersih. Geometri tidak berubah, jadi pemetaan
        # kotak field pada halaman Segmentasi tetap sama.
        crop_tampil = _trim_latar(crop_base)[0]
        tampil, trim_tampil = _trim_latar(kertas)
        tampil = _putihkan_latar_tepi(tampil)

    return {
        # "asli" hanya ditampilkan (tidak diubah), jadi cukup rujuk gambar
        # masukan tanpa menyalin. Menyalin citra penuh per nota membebani
        # memori pada mesin ber-RAM kecil.
        "asli": gambar,
        "crop": crop_tampil,
        "hasil": hasil,
        "tampil": tampil,
        # Posisi kertas pada kanvas asli + offset pemangkasan latar pada
        # gambar tampil; dipakai halaman Segmentasi untuk memetakan kotak
        # field (ruang kanvas) ke ruang gambar tampil.
        "bbox_kertas": bbox,
        "trim_tampil": trim_tampil,
        "kotak": potong["kotak"],
        "status": potong["status"],
        "pesan": potong["pesan"],
        "metode": METODE_PENINGKATAN,
    }

