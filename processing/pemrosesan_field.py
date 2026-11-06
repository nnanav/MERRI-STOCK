from pathlib import Path

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent.parent


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan_gambar(path, gambar):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", gambar)
    if ok:
        data.tofile(str(path))


def mask_merah(gambar):
    # Mask teks merah (angka nomor nota dicetak merah).
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 65, 45), (12, 255, 255)),
        cv2.inRange(hsv, (150, 65, 45), (179, 255, 255)),
    )


def _komponen(biner, min_area=20):
    n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    hasil = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        hasil.append([int(x), int(y), int(w), int(h), int(a)])
    return hasil


def isolasi_nomor(kecil):
    """
    Isolasi empat digit nomor nota dari tulisan merah 'NO : xxxx'.

    Mengembalikan dict berisi status, crop baris, crop digit, mask,
    dan pesan diagnostik. Tidak memakai crop persentase tetap.
    """
    H, W = kecil.shape[:2]

    merah = mask_merah(kecil)

    # Batasi ke area atas-kanan tempat nomor nota berada,
    # agar logo merah di kiri tidak ikut terambil.
    wilayah = np.zeros_like(merah)
    wilayah[0:int(0.34 * H), int(0.45 * W):W] = 255
    merah = cv2.bitwise_and(merah, wilayah)

    gabung = cv2.morphologyEx(
        merah, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    )

    komp = _komponen(gabung, min_area=80)
    komp = [k for k in komp if k[3] >= 6]

    if not komp:
        return {
            "status": "tidak_ada_merah",
            "pesan": "Tidak ada teks merah 'NO : xxxx' di area atas-kanan.",
            "crop_baris": None, "crop_digit": None, "mask": merah,
        }

    # Ambil baris merah paling atas sebagai baris nomor nota.
    komp.sort(key=lambda z: z[1])
    y_atas = komp[0][1]
    toleransi = max(8, int(0.02 * H))
    baris = [k for k in komp if abs(k[1] - y_atas) <= toleransi]

    x1 = min(k[0] for k in baris)
    y1 = min(k[1] for k in baris)
    x2 = max(k[0] + k[2] for k in baris)
    y2 = max(k[1] + k[3] for k in baris)

    pad = 6
    X1 = max(0, x1 - pad)
    Y1 = max(0, y1 - pad)
    X2 = min(W, x2 + pad)
    Y2 = min(H, y2 + pad)

    crop_baris = kecil[Y1:Y2, X1:X2].copy()

    digit, pesan, mask_digit = _potong_digit(crop_baris)

    if digit is None:
        return {
            "status": "digit_gagal",
            "pesan": pesan,
            "bbox": (X1, Y1, X2, Y2),
            "crop_baris": crop_baris,
            "crop_digit": None,
            "mask": merah,
            "mask_digit": mask_digit,
        }

    return {
        "status": "ok",
        "pesan": pesan,
        "bbox": (X1, Y1, X2, Y2),
        "crop_baris": crop_baris,
        "crop_digit": digit,
        "mask": merah,
        "mask_digit": mask_digit,
    }


def _potong_digit(crop_baris):
    """
    Buang awalan 'NO' dan titik dua, sisakan hanya bagian digit.

    Titik dua dideteksi sebagai komponen bertinggi kecil yang berada
    di antara huruf dan digit. Digit boleh menyatu, jadi kita hanya
    menentukan batas kiri; pemecahan digit dilakukan tahap segmentasi.
    """
    merah = mask_merah(crop_baris)

    if cv2.countNonZero(merah) == 0:
        return None, "Mask merah kosong pada crop baris nomor.", merah

    komp = _komponen(merah, min_area=12)
    komp = [k for k in komp if k[3] >= 4]

    if len(komp) < 2:
        return None, "Komponen digit tidak cukup untuk dipisah.", merah

    komp.sort(key=lambda z: z[0])
    h_max = max(k[3] for k in komp)
    h_besar = [k for k in komp if k[3] >= 0.5 * h_max]

    if len(h_besar) < 3:
        return None, "Digit tidak terdeteksi (komponen tinggi kurang).", merah

    # Titik dua: komponen rendah di antara huruf (kiri) dan digit (kanan).
    x_huruf = h_besar[1][0] + h_besar[1][2]
    x_akhir_besar = h_besar[-1][0]

    kandidat_titik = [
        k for k in komp
        if k[3] < 0.5 * h_max and x_huruf <= k[0] <= x_akhir_besar
    ]

    if kandidat_titik:
        x_mulai = max(k[0] + k[2] for k in kandidat_titik)
        metode = "setelah titik dua"
    else:
        x_mulai = x_huruf
        metode = "setelah huruf NO"

    # Longgarkan sedikit ke kanan agar tidak memotong digit pertama.
    x_mulai = max(0, x_mulai - 1)

    digit_warna = crop_baris[:, x_mulai:].copy()

    # Rapatkan ke bingkai tinta merah.
    m = mask_merah(digit_warna)

    # Buang tanda baca/awalan di depan digit dengan memulai dari
    # komponen setinggi digit yang pertama. Digit nol di depan tetap
    # dijaga (nol juga setinggi digit), sedangkan titik dua yang pendek
    # (bahkan yang memudar) ikut terbuang.
    komp_d = sorted(_komponen(m, min_area=8), key=lambda z: z[0])
    if komp_d:
        hmax_d = max(k[3] for k in komp_d)
        for k in komp_d:
            # Digit awal harus tinggi DAN cukup lebar; titik dua
            # (sempit) tersaring walau tingginya menempel.
            if k[3] >= 0.55 * hmax_d and k[2] >= 0.30 * hmax_d:
                if k[0] > 0:
                    digit_warna = digit_warna[:, k[0]:]
                    m = m[:, k[0]:]
                break

    nz = cv2.findNonZero(m)
    if nz is None:
        return None, "Bagian digit kosong setelah pemotongan.", m

    x, y, w, h = cv2.boundingRect(nz)
    if w < 6 or h < 6:
        return None, "Bagian digit terlalu kecil.", m

    digit_warna = digit_warna[y:y + h, x:x + w].copy()
    m = m[y:y + h, x:x + w]

    # Hitung jumlah komponen digit sebagai info (boleh < 4 bila menyatu).
    n_digit = len([k for k in _komponen(m, min_area=10)])
    pesan = f"Digit diisolasi ({metode}); komponen tinta merah: {n_digit}."

    return digit_warna, pesan, m


def bersihkan_minus(crop):
    """
    Buang tanda minus pembuka daftar (mis. '- Kanstein Jepit ...')
    SEBELUM segmentasi, dengan menghapus piksel minus pembuka saja.

    Hanya minus yang berdiri sendiri di awal baris yang dihapus,
    sehingga kode produk seperti 'K-300' tetap utuh. Titik atau noise
    kecil di depan minus juga ikut dibersihkan bila ada.
    """
    if crop is None or crop.size == 0:
        return {"crop": crop, "dihapus": False, "pesan": "Crop kosong."}

    if crop.ndim == 3:
        abu = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        abu = crop.copy()

    h, w = abu.shape[:2]
    if h < 4 or w < 6:
        return {"crop": abu, "dihapus": False, "pesan": "Crop terlalu kecil."}

    # Tinta gelap di atas latar terang.
    biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    komp = _komponen(biner, min_area=8)
    if len(komp) < 2:
        return {"crop": abu, "dihapus": False, "pesan": "Komponen tidak cukup."}

    komp.sort(key=lambda z: z[0])

    tinggi_teks = h
    tinggi_besar = [k[3] for k in komp if k[3] >= 0.4 * h]
    if tinggi_besar:
        tinggi_teks = float(np.median(tinggi_besar))

    # Komponen teks pertama (huruf/angka sungguhan) menentukan awal nama.
    idx_teks = None
    for i, k in enumerate(komp):
        if k[2] >= 0.40 * tinggi_teks and k[3] >= 0.55 * tinggi_teks:
            idx_teks = i
            break

    if idx_teks is None:
        return {"crop": abu, "dihapus": False, "pesan": "Tidak ada huruf/angka terdeteksi."}

    x_teks = komp[idx_teks][0]
    if x_teks <= 2:
        return {"crop": abu, "dihapus": False, "pesan": "Tidak ada awalan untuk dibersihkan."}

    # Deteksi awalan memakai ambang lebih longgar karena titik dua/noise
    # sering lebih pudar dari huruf sehingga hilang pada Otsu.
    latar = int(np.median(abu))
    ambang_lembut = max(40, latar - 45)
    lembut = (abu < ambang_lembut).astype(np.uint8) * 255

    awalan_mask = np.zeros_like(lembut)
    awalan_mask[:, :x_teks] = lembut[:, :x_teks]

    komp_awalan = _komponen(awalan_mask, min_area=6)

    ada_minus = False
    for k in komp_awalan:
        x, y, ww, hh, a = k
        tengah_y = y + hh / 2
        dekat_tengah = abs(tengah_y - h / 2) <= 0.45 * h
        if (
            hh <= 0.45 * tinggi_teks
            and 1.3 * hh <= ww <= 1.4 * tinggi_teks
            and dekat_tengah
        ):
            ada_minus = True
            break

    if not ada_minus:
        return {"crop": abu, "dihapus": False, "pesan": "Tidak ada minus pembuka."}

    # Hapus semua awalan (titik dua, titik, noise, minus) sebelum huruf pertama.
    # Huruf pertama dijaga: pemotongan berhenti sedikit sebelum komponen teks.
    hasil = abu.copy()
    awal = max(0, x_teks - max(2, int(0.12 * tinggi_teks)))
    if awal > 0:
        hasil[:, :awal] = latar

    # Rapatkan ke sisa tinta supaya bingkai tidak longgar.
    sisa = cv2.threshold(
        hasil, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    nz = cv2.findNonZero(sisa)
    if nz is not None:
        x, y, ww, hh = cv2.boundingRect(nz)
        if ww > 3 and hh > 3:
            hasil = hasil[y:y + hh, x:x + ww].copy()

    return {
        "crop": hasil,
        "dihapus": True,
        "pesan": "Minus pembuka dihapus sebelum segmentasi.",
    }


def _rapatkan(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bb = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    nz = cv2.findNonZero(bb)
    if nz is None:
        return None
    x, y, ww, hh = cv2.boundingRect(nz)
    if ww < 3 or hh < 3:
        return None
    return img[y:y + hh, x:x + ww].copy()


def pisah_jumlah_satuan(crop):
    """
    Pisahkan satu field 'jumlah satuan' (mis. '3.800 Pcs x') menjadi
    jumlah ('3.800') dan satuan ('Pcs x') berdasarkan celah spasi
    tinta, bukan koordinat tetap.
    """
    kosong = {
        "jumlah": None, "satuan": None,
        "status": "kosong", "pesan": "Crop kosong.",
    }
    if crop is None or crop.size == 0:
        return kosong

    if crop.ndim == 3:
        abu = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        abu = crop.copy()

    h, w = abu.shape[:2]
    if h < 4 or w < 8:
        return kosong

    biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    proyeksi = np.count_nonzero(biner, axis=0)
    ada_tinta = proyeksi > 0

    komp = _komponen(biner, min_area=8)
    if len(komp) < 2:
        return {
            "jumlah": crop, "satuan": None,
            "status": "satu_komponen",
            "pesan": "Hanya satu kelompok tinta; satuan tidak terpisah.",
        }

    tinggi = float(np.median([k[3] for k in komp]))
    ambang_celah = max(3, 0.45 * tinggi)

    celah = []
    mulai = None
    for x in range(w):
        if not ada_tinta[x]:
            if mulai is None:
                mulai = x
        else:
            if mulai is not None:
                celah.append((mulai, x - 1))
                mulai = None
    if mulai is not None:
        celah.append((mulai, w - 1))

    # Celah pertama yang cukup lebar di tengah -> batas jumlah/satuan.
    x_batas = None
    for (a, b) in celah:
        lebar = b - a + 1
        if lebar >= ambang_celah and a > 0 and b < w - 1:
            x_batas = (a + b) // 2
            break

    if x_batas is None:
        return {
            "jumlah": crop, "satuan": None,
            "status": "tidak_terpisah",
            "pesan": "Celah jumlah-satuan tidak ditemukan.",
        }

    jumlah = _rapatkan(crop[:, :x_batas])
    satuan = _rapatkan(crop[:, x_batas:])

    if jumlah is None or satuan is None:
        return {
            "jumlah": crop, "satuan": None,
            "status": "tidak_terpisah",
            "pesan": "Hasil pemisahan kosong; dipakai crop gabungan.",
        }

    return {
        "jumlah": jumlah,
        "satuan": satuan,
        "status": "terpisah",
        "pesan": "Jumlah dan satuan dipisah berdasarkan celah.",
        "x_batas": x_batas,
    }


def segmentasi_per_karakter(crop, skala=2):
    """
    Segmentasi satu baris teks menjadi karakter terpisah.

    Target: satu bounding box untuk satu karakter. Karakter yang menyatu
    dipisah memakai profil proyeksi (lembah tinta), lalu tiap kotak
    dirapatkan ke tinta aslinya. Tidak memakai CNN, jadi cepat dan
    dipakai untuk visualisasi pemisahan huruf/angka/simbol.
    """
    kosong = {"biner": None, "karakter": [], "skala": skala}
    if crop is None or crop.size == 0:
        return kosong

    abu = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h0, w0 = abu.shape[:2]
    if h0 < 4 or w0 < 4:
        return kosong

    up = cv2.resize(abu, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
    _, biner = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    H, W = biner.shape[:2]

    # Buang noise sangat kecil (bukan bagian karakter).
    n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    ambang_luas = max(4, int(0.0006 * H * W))
    for i in range(1, n):
        if stats[i, 4] < ambang_luas:
            x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
            biner[y:y + h, x:x + w] = 0

    ys = np.where(np.count_nonzero(biner, axis=1) > 0)[0]
    tinggi = int(ys.max() - ys.min() + 1) if len(ys) else H
    if tinggi < 3:
        return {"biner": biner, "karakter": [], "skala": skala}

    char_w = max(3.0, 0.62 * tinggi)

    proj = np.count_nonzero(biner, axis=0)
    ambang_ink = max(1, int(0.02 * tinggi))

    # Kelompokkan kolom bertinta menjadi "gugus" (satu atau lebih karakter).
    runs = []
    mulai = None
    for x in range(W):
        if proj[x] > ambang_ink:
            if mulai is None:
                mulai = x
        else:
            if mulai is not None:
                runs.append((mulai, x - 1))
                mulai = None
    if mulai is not None:
        runs.append((mulai, W - 1))

    batas = []
    for a, b in runs:
        lebar = b - a + 1
        n_seg = max(1, int(round(lebar / char_w)))
        if n_seg == 1:
            batas.append((a, b))
            continue
        # Bagi gugus lebar pada lembah proyeksi terdekat titik bagi rata.
        titik = [a]
        for k in range(1, n_seg):
            target = a + round(k * lebar / n_seg)
            lo = max(a + 1, target - int(0.25 * char_w))
            hi = min(b - 1, target + int(0.25 * char_w))
            if hi <= lo:
                titik.append(target)
            else:
                titik.append(lo + int(np.argmin(proj[lo:hi + 1])))
        titik.append(b + 1)
        for k in range(n_seg):
            batas.append((titik[k], titik[k + 1] - 1))

    karakter = []
    for (x1, x2) in batas:
        kolom = biner[:, x1:x2 + 1]
        yy = np.where(np.count_nonzero(kolom, axis=1) > 0)[0]
        if len(yy) == 0:
            continue
        karakter.append({
            "x1": int(x1 / skala),
            "y1": int(yy.min() / skala),
            "x2": int(x2 / skala) + 1,
            "y2": int(yy.max() / skala) + 1,
        })

    karakter.sort(key=lambda z: z["x1"])
    return {"biner": biner, "karakter": karakter, "skala": skala}


def _klaster_baris(komp, toleransi_y):
    # Kelompokkan komponen menjadi baris teks berdasarkan pusat Y.
    baris = []
    for k in sorted(komp, key=lambda z: z[1] + z[3] / 2):
        cy = k[1] + k[3] / 2
        cocok = None
        for grup in baris:
            if abs(grup["cy"] - cy) <= toleransi_y:
                cocok = grup
                break
        if cocok is None:
            baris.append({"cy": cy, "komponen": [k]})
        else:
            cocok["komponen"].append(k)
            cys = [g[1] + g[3] / 2 for g in cocok["komponen"]]
            cocok["cy"] = float(np.mean(cys))
    return baris


def cari_tanggal(kecil, lokasi_tanggal=None):
    """
    Cari baris teks tanggal ('Kota, dd Mmm yyyy') di dekat tanda tangan.

    Hasil transformasi/orb hanya dipakai sebagai wilayah pencarian.
    Baris tanggal dipilih berdasarkan bukti visual (pola komponen teks),
    bukan koordinat final tetap. Bila pembersihan tanda tangan tidak
    aman, crop asli dipertahankan dengan status yang jujur.
    """
    H, W = kecil.shape[:2]

    if lokasi_tanggal is not None and "x" in lokasi_tanggal:
        rx1 = int(lokasi_tanggal["x"])
        ry1 = int(lokasi_tanggal["y"])
        rx2 = rx1 + int(lokasi_tanggal["w"])
        ry2 = ry1 + int(lokasi_tanggal["h"])
        asal = "homografi"
    else:
        # Wilayah pencarian konservatif di kanan, di atas nama direktur.
        rx1, ry1 = int(0.68 * W), int(0.33 * H)
        rx2, ry2 = int(1.00 * W), int(0.52 * H)
        asal = "fallback"

    # Perlebar sedikit untuk toleransi geser.
    pad_x, pad_y = int(0.02 * W), int(0.02 * H)
    rx1 = max(0, rx1 - pad_x)
    ry1 = max(0, ry1 - pad_y)
    rx2 = min(W, rx2 + pad_x)
    ry2 = min(H, ry2 + pad_y)

    if rx2 - rx1 < 20 or ry2 - ry1 < 12:
        return {
            "status": "wilayah_kosong",
            "pesan": "Wilayah pencarian tanggal terlalu kecil.",
            "asal": asal,
        }

    wilayah = kecil[ry1:ry2, rx1:rx2].copy()
    abu = cv2.cvtColor(wilayah, cv2.COLOR_BGR2GRAY)
    biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    rh, rw = abu.shape[:2]
    komp = _komponen(biner, min_area=12)
    komp = [k for k in komp if k[2] >= 2 and k[3] >= 4]
    if not komp:
        return {
            "status": "tidak_ada_teks",
            "pesan": "Tidak ada komponen teks pada wilayah tanggal.",
            "asal": asal, "wilayah": wilayah, "wilayah_bbox": (rx1, ry1, rx2, ry2),
        }

    tinggi_med = float(np.median([k[3] for k in komp]))
    toleransi_y = max(4, 0.6 * tinggi_med)
    baris = _klaster_baris(komp, toleransi_y)

    # Skor baris tanggal: banyak komponen mirip digit, ada koma, bukan blob tanda tangan.
    def skor(grup):
        ks = grup["komponen"]
        n = len(ks)
        hh = float(np.median([k[3] for k in ks]))
        ww = max(k[0] + k[2] for k in ks) - min(k[0] for k in ks)
        raksasa = sum(1 for k in ks if k[3] > 1.8 * hh)
        # Komponen mirip digit: tinggi normal, lebar sempit-menengah.
        digit_like = sum(
            1 for k in ks
            if 0.28 * hh <= k[2] <= 0.95 * hh and k[3] >= 0.7 * hh
        )
        # Komponen mirip koma/titik: kecil dan rendah.
        koma_like = sum(
            1 for k in ks
            if k[3] <= 0.4 * hh and k[2] <= 0.5 * hh
        )
        cy = grup["cy"]
        return (
            2.0 * digit_like
            + 1.5 * (1 if koma_like >= 1 else 0)
            + 0.3 * n
            - 4.0 * raksasa
            + 0.001 * ww
            - 0.01 * cy
        )

    baris.sort(key=skor, reverse=True)
    grup = baris[0]
    ks = grup["komponen"]

    x1 = min(k[0] for k in ks)
    y1 = min(k[1] for k in ks)
    x2 = max(k[0] + k[2] for k in ks)
    y2 = max(k[1] + k[3] for k in ks)

    m = 3
    x1 = max(0, x1 - m); y1 = max(0, y1 - m)
    x2 = min(rw, x2 + m); y2 = min(rh, y2 + m)

    kandidat = wilayah[y1:y2, x1:x2].copy()
    kandidat_biner = biner[y1:y2, x1:x2].copy()

    # Coba buang goresan tanda tangan: hanya komponen jelas lebih besar.
    hh_med = float(np.median([k[3] for k in ks]))
    bersih_biner = kandidat_biner.copy()
    mask_buang = np.zeros_like(kandidat_biner)
    dibuang = 0
    n_kecil = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        kandidat_biner, connectivity=8
    )
    for i in range(1, n):
        _, _, ww, hh, a = stats[i]
        if hh > 1.7 * hh_med or ww > 0.75 * (x2 - x1):
            region = (stats[i][0], stats[i][1], ww, hh)
            mask_buang[region[1]:region[1] + hh, region[0]:region[0] + ww] |= (
                kandidat_biner[region[1]:region[1] + hh,
                               region[0]:region[0] + ww]
            )
            dibuang += 1
        else:
            n_kecil += 1

    bersih_biner = cv2.subtract(kandidat_biner, mask_buang)
    sisa = cv2.countNonZero(bersih_biner)
    total = cv2.countNonZero(kandidat_biner)

    if dibuang == 0:
        status = "tanpa_goresan"
        pesan = "Tidak ada goresan tanda tangan terpisah; crop dipakai apa adanya."
    elif sisa < 0.35 * total or n_kecil < 2:
        status = "pembersihan_belum_meyakinkan"
        pesan = "Pembersihan berisiko menghapus tanggal; crop asli dipakai."
        bersih_biner = kandidat_biner
    else:
        status = "dibersihkan"
        pesan = f"{dibuang} goresan tanda tangan dibuang, karakter tanggal dijaga."

    bersih_visual = cv2.bitwise_not(bersih_biner)

    return {
        "status": status,
        "pesan": pesan,
        "asal": asal,
        "wilayah": wilayah,
        "wilayah_bbox": (rx1, ry1, rx2, ry2),
        "kandidat": kandidat,
        "kandidat_biner": kandidat_biner,
        "mask_buang": mask_buang,
        "bersih_biner": bersih_biner,
        "bersih_visual": bersih_visual,
        "kandidat_bbox": (rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2),
    }