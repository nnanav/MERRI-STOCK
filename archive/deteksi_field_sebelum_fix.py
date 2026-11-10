from pathlib import Path
import csv
import sys

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "hasil_deteksi_field"
LEBAR_PROSES = 1400


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan_gambar(path, gambar):
    sukses, data = cv2.imencode(".png", gambar)
    if sukses:
        data.tofile(str(path))


def baca_kandidat(csv_path):

    hasil = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for baris in reader:
            x1 = int(baris["x1"])
            y1 = int(baris["y1"])
            x2 = int(baris["x2"])
            y2 = int(baris["y2"])

            if x2 <= x1 or y2 <= y1:
                continue

            hasil.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "w": x2 - x1,
                "h": y2 - y1
            })

    return hasil

def samakan_ukuran(gambar):
    h, w = gambar.shape[:2]
    skala = LEBAR_PROSES / w

    return cv2.resize(
        gambar,
        (LEBAR_PROSES, max(1, round(h * skala))),
        interpolation=(
            cv2.INTER_AREA if skala < 1
            else cv2.INTER_CUBIC
        )
    )

def kelompokkan_sejajar(kandidat):

    if not kandidat:
        return []

    kandidat = sorted(kandidat, key=lambda k: k["cy"])

    tinggi_median = np.median([k["h"] for k in kandidat])
    toleransi = max(5, min(9, tinggi_median * 0.4))

    kelompok = []

    for kotak in kandidat:
        cocok = None
        jarak_terkecil = float("inf")

        for grup in kelompok:
            tengah = np.median([k["cy"] for k in grup])
            jarak = abs(kotak["cy"] - tengah)

            if jarak <= toleransi and jarak < jarak_terkecil:
                cocok = grup
                jarak_terkecil = jarak

        if cocok is None:
            kelompok.append([kotak])
        else:
            cocok.append(kotak)

    return kelompok

def deteksi_produk(kandidat, lebar, tinggi):

    # Kandidat kolom jumlah
    calon_jumlah = [
        k for k in kandidat
        if 0.57 * lebar <= k["x1"] < 0.70 * lebar
        and k["w"] >= 35
        and 0.17 * tinggi <= k["cy"] <= 0.60 * tinggi
    ]

    # Kandidat nama produk di kiri jumlah
    calon_nama = [
        k for k in kandidat
        if 0.26 * lebar <= k["x1"] < 0.56 * lebar
        and k["w"] >= 75
        and 0.17 * tinggi <= k["cy"] <= 0.60 * tinggi
    ]

    # Kandidat harga di kanan jumlah
    calon_harga = [
        k for k in kandidat
        if k["x1"] >= 0.69 * lebar
        and k["w"] >= 30
        and 0.17 * tinggi <= k["cy"] <= 0.60 * tinggi
    ]

    hasil = []
    nama_terpakai = set()

    # Proses jumlah dari atas ke bawah
    calon_jumlah.sort(key=lambda k: k["cy"])

    for jumlah in calon_jumlah:

        # Harus ada kolom harga di sebelah kanan
        harga_valid = [
            h for h in calon_harga
            if h is not jumlah
            and h["x1"] >= jumlah["x2"] + 8
            and abs(h["cy"] - jumlah["cy"]) <= 22
        ]

        if not harga_valid:
            continue

        # Cari nama produk di kiri jumlah
        nama_valid = []

        for nama in calon_nama:
            identitas = (
                nama["x1"], nama["y1"],
                nama["x2"], nama["y2"]
            )

            if identitas in nama_terpakai:
                continue

            if nama["x2"] >= jumlah["x1"] - 8:
                continue

            selisih_y = abs(
                nama["cy"] - jumlah["cy"]
            )

            if selisih_y > 24:
                continue

            nama_valid.append(nama)

        if not nama_valid:
            continue

        # Pilih nama dengan posisi Y terdekat.
        # Jarak horizontal digunakan sebagai pembeda jika posisi Y hampir sama.
        nama = min(
            nama_valid,
            key=lambda n: (
                abs(n["cy"] - jumlah["cy"]),
                jumlah["x1"] - n["x2"]
            )
        )

        identitas = (
            nama["x1"], nama["y1"],
            nama["x2"], nama["y2"]
        )

        nama_terpakai.add(identitas)
        hasil.append(nama)

    return sorted(
        hasil,
        key=lambda k: k["cy"]
    )


# DETEKSI FIELD TAMBAHAN


def deteksi_field_lain(kandidat, lebar, tinggi, produk):
    """
    Deteksi field tambahan.

    Nomor nota dan tanggal menggunakan
    area pencarian khusus agar tidak
    salah memilih baris lain.

    Deteksi nama produk tetap menggunakan
    fungsi deteksi_produk() yang sudah ada.
    """

    def buat_kotak(x1, y1, x2, y2):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(lebar, int(x2))
        y2 = min(tinggi, int(y2))

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "w": x2 - x1,
            "h": y2 - y1
        }

    # --------------------------------------
    # 1. AREA NOMOR NOTA
    # --------------------------------------
    # Mencakup tulisan NO dan angka di
    # kanan atas, bukan baris "Lembar ke".

    nomor_nota = buat_kotak(
        0.78 * lebar,
        0.055 * tinggi,
        0.95 * lebar,
        0.115 * tinggi
    )

    # --------------------------------------
    # 2. AREA TANGGAL
    # --------------------------------------
    # Mencakup wilayah tanggal di atas
    # tanda tangan, bukan nama direktur.

    tanggal = buat_kotak(
        0.77 * lebar,
        0.68 * tinggi,
        0.97 * lebar,
        0.76 * tinggi
    )

    # --------------------------------------
    # 3. JUMLAH DAN SATUAN
    # --------------------------------------

    kandidat_jumlah = [
        k for k in kandidat
        if (
            0.55 * lebar <= k["x1"] < 0.80 * lebar
            and k["w"] >= 25
        )
    ]

    jumlah_terpakai = set()
    rincian_produk = []

    for nomor, nama in enumerate(produk, start=1):
        calon = []

        for k in kandidat_jumlah:
            identitas = (
                k["x1"], k["y1"],
                k["x2"], k["y2"]
            )

            if identitas in jumlah_terpakai:
                continue

            if k["x1"] <= nama["x2"]:
                continue

            selisih_y = abs(
                k["cy"] - nama["cy"]
            )

            toleransi = max(
                16,
                0.8 * max(nama["h"], k["h"])
            )

            if selisih_y <= toleransi:
                calon.append(k)

        kotak_jumlah = None

        if calon:
            kotak_jumlah = min(
                calon,
                key=lambda k: (
                    abs(k["cy"] - nama["cy"]),
                    k["x1"]
                )
            )

            identitas = (
                kotak_jumlah["x1"],
                kotak_jumlah["y1"],
                kotak_jumlah["x2"],
                kotak_jumlah["y2"]
            )

            jumlah_terpakai.add(identitas)

        rincian_produk.append({
            "produk": nomor,
            "nama_produk": nama,
            "jumlah_satuan": kotak_jumlah
        })

    return {
        "nomor_nota": nomor_nota,
        "tanggal": tanggal,
        "rincian_produk": rincian_produk
    }

def _kelompok_baris(kandidat, toleransi):
    # Kelompokkan kandidat teks menjadi baris berdasarkan pusat Y.
    baris = []
    for k in sorted(kandidat, key=lambda z: z["cy"]):
        cocok = None
        for g in baris:
            if abs(g["cy"] - k["cy"]) <= toleransi:
                cocok = g
                break
        if cocok is None:
            baris.append({"cy": k["cy"], "isi": [k]})
        else:
            cocok["isi"].append(k)
            cocok["cy"] = sum(x["cy"] for x in cocok["isi"]) / len(cocok["isi"])
    return baris


def _kotak_gabung(daftar):
    if not daftar:
        return None
    x1 = min(k["x1"] for k in daftar)
    y1 = min(k["y1"] for k in daftar)
    x2 = max(k["x2"] for k in daftar)
    y2 = max(k["y2"] for k in daftar)
    return {
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
        "w": x2 - x1, "h": y2 - y1,
    }


def _nilai_setelah_label(grup):
    # Label = kandidat paling kiri; nilai = kandidat di kanan label.
    isi = sorted(grup["isi"], key=lambda k: k["x1"])
    if len(isi) < 2:
        return None
    label = isi[0]
    calon = [k for k in isi[1:] if k["x1"] >= label["x2"] - 2]
    if not calon:
        return None
    return _kotak_gabung(calon)


def deteksi_penerima_proyek(kandidat, lebar, tinggi):
    # Berbasis label: baris "Telah diterima dari" dan "Proyek" berada di
    # kolom label kiri. Nilai diambil dari kandidat di kanan label pada
    # baris yang sama, sehingga label tidak ikut serta.
    band = [
        k for k in kandidat
        if 0.12 * tinggi <= k["cy"] <= 0.27 * tinggi
    ]
    if not band:
        return None, None

    tinggi_med = float(np.median([k["h"] for k in band]))

    # Baris header seperti "KWITANSI" berada di tengah dan lebih besar;
    # jadikan penanda untuk mulai mencari baris isi di bawahnya. Bila tidak
    # ada (mis. header tertutup), pakai ambang posisi vertikal.
    header = [k for k in kandidat
              if 0.12 * tinggi <= k["cy"] <= 0.20 * tinggi
              and k["cx"] >= 0.40 * lebar
              and k["h"] >= 1.3 * tinggi_med]

    if header:
        y_mulai = max(k["cy"] for k in header)
    else:
        y_mulai = 0.15 * tinggi

    # Baris di kolom kiri setelah header, ukuran teks normal.
    kiri = [k for k in band
            if k["cy"] > y_mulai
            and k["x1"] <= 0.45 * lebar
            and k["h"] <= 1.6 * tinggi_med]
    if not kiri:
        return None, None

    toleransi = max(4, 0.7 * tinggi_med)
    baris = _kelompok_baris(kiri, toleransi)
    baris.sort(key=lambda g: g["cy"])

    def pilih_baris(grup):
        # Prioritaskan kandidat lebar (nilai gabungan label+nilai) agar
        # nilai yang menempel ke label tetap ikut; buang tanda baca tipis
        # seperti titik dua di awal nilai.
        kandidat_baris = sorted(grup["isi"], key=lambda k: k["x1"])
        kandidat_baris = [k for k in kandidat_baris if k["w"] >= 25]
        if not kandidat_baris:
            return None
        # Nilai berada di kanan label/titik dua (label ~0.26W, ':' ~0.33W).
        pilihan = [k for k in kandidat_baris if k["x1"] >= 0.34 * lebar]
        if pilihan:
            return _kotak_gabung(pilihan)
        return _kotak_gabung(kandidat_baris)

    hasil = [pilih_baris(g) for g in baris[:2]]
    while len(hasil) < 2:
        hasil.append(None)

    # penerima = baris paling atas, proyek = baris berikutnya.
    return hasil[0], hasil[1]


def deteksi_harga_satuan(kandidat, lebar, tinggi, produk):
    # Harga satuan = angka sebelum "= Rp"; bukan total di paling kanan dan
    # bukan tulisan "Rp". Ambil angka murni di area kanan pada baris produk.
    hasil = []
    for p in produk:
        toleransi = max(8, 0.5 * p["h"])
        baris = [
            k for k in kandidat
            if abs(k["cy"] - p["cy"]) <= toleransi
            and k["x1"] >= p["x2"] + 8
            and k["x1"] >= 0.70 * lebar
            and k["w"] >= 28
        ]
        if not baris:
            hasil.append(None)
            continue

        cy_dekat = min(abs(k["cy"] - p["cy"]) for k in baris)
        kandidat_baris = [
            k for k in baris
            if abs(k["cy"] - p["cy"]) <= cy_dekat + 3
        ]
        kandidat_baris.sort(key=lambda k: k["x1"])
        hasil.append(kandidat_baris[0])
    return hasil


def cari_harga_satuan(kecil, produk, lebar):
    """
    Ambil harga satuan langsung dari citra: kelompok tinta pertama
    (paling kiri) di area kanan baris produk. Ini memastikan tulisan
    "= Rp" dan kolom total tidak ikut terambil.
    """
    H, W = kecil.shape[:2]
    hasil = []
    for p in produk:
        y1 = max(0, int(p["y1"]) - 2)
        y2 = min(H, int(p["y2"]) + 2)
        h = y2 - y1
        if h < 4:
            hasil.append(None)
            continue

        band = cv2.cvtColor(kecil[y1:y2, :], cv2.COLOR_BGR2GRAY)
        _, biner = cv2.threshold(
            band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        proj = np.count_nonzero(biner, axis=0)
        ada = proj > 0

        x0 = int(0.66 * W)
        x0 = max(x0, int(p["x2"]) + 4)

        runs = []
        mulai = None
        for x in range(x0, W):
            if ada[x]:
                if mulai is None:
                    mulai = x
            else:
                if mulai is not None:
                    runs.append((mulai, x - 1))
                    mulai = None
        if mulai is not None:
            runs.append((mulai, W - 1))

        # Gabungkan run berdekatan (bagian dari satu angka).
        gab = []
        ambang_gab = max(3, int(0.45 * h))
        for a, b in runs:
            if gab and a - gab[-1][1] <= ambang_gab:
                gab[-1] = (gab[-1][0], b)
            else:
                gab.append((a, b))

        # Kelompok pertama yang cukup lebar = angka harga satuan.
        pilih = None
        for a, b in gab:
            if (b - a + 1) >= 0.85 * h:
                pilih = (a, b)
                break
        if pilih is None:
            hasil.append(None)
            continue

        hasil.append({
            "x1": int(pilih[0]), "y1": int(y1),
            "x2": int(pilih[1]) + 1, "y2": int(y2),
            "cx": (pilih[0] + pilih[1]) / 2, "cy": (y1 + y2) / 2,
            "w": pilih[1] - pilih[0] + 1, "h": h,
        })
    return hasil


def pangkas_awalan_tipis(crop):
    """
    Buang awalan tipis (titik dua, titik, noise) di kiri crop nilai.

    Memakai tinggi tinta per kolom: kolom yang tingginya jauh di bawah
    tinggi teks dianggap titik dua/titik, lalu dipotong sampai kolom
    pertama yang benar-benar bagian huruf/angka.
    """
    if crop is None or crop.size == 0:
        return crop

    abu = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = abu.shape[:2]
    if h < 4 or w < 8:
        return crop

    biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    tinggi_kolom = np.count_nonzero(biner, axis=0)

    n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    tinggi_teks = 0.0
    for i in range(1, n):
        if stats[i, 4] >= 10:
            tinggi_teks = max(tinggi_teks, float(stats[i, 3]))
    if tinggi_teks <= 0:
        tinggi_teks = float(h)

    ambang_tall = 0.55 * tinggi_teks
    batas = 0
    for x in range(w):
        if tinggi_kolom[x] >= ambang_tall:
            break
        batas = x + 1

    if batas <= 0 or batas >= w - 2:
        return crop
    return crop[:, batas:].copy()


def pangkas_hingga_rp(crop):
    """
    Pangkas crop harga agar berhenti sebelum tulisan "= Rp".

    Mendeteksi simbol "=" sebagai komponen pendek tetapi cukup lebar
    (bukan titik/koma yang sangat kecil), lalu memotong tepat di kirinya.
    """
    if crop is None or crop.size == 0:
        return crop

    abu = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = abu.shape[:2]
    if h < 4 or w < 10:
        return crop

    biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)

    komp = []
    for i in range(1, n):
        x, y, ww, hh, a = stats[i]
        if a < 6:
            continue
        komp.append((int(x), int(y), int(ww), int(hh)))
    if len(komp) < 3:
        return crop

    komp.sort(key=lambda z: z[0])
    tinggi = float(np.median([k[3] for k in komp]))

    for i in range(1, len(komp)):
        x, y, ww, hh = komp[i]
        # Simbol "=" : jauh lebih pendek dari angka, tapi cukup lebar.
        if (
            hh <= 0.5 * tinggi
            and ww >= 0.35 * tinggi
            and ww >= 3
            and any(kk[3] >= 0.7 * tinggi for kk in komp[:i])
        ):
            potong = max(1, x - 2)
            if potong >= 4:
                return crop[:, :potong].copy()
            return crop

    return crop


def bersihkan_label_nilai(crop):
    """
    Buang label ("Telah diterima dari", "Proyek") dan titik dua dari crop
    nilai penerima/proyek.

    Nilai sebenarnya selalu berupa SATU kelompok kata di kanan, sedangkan
    label + titik dua berada di kiri dan membentuk beberapa kelompok kata.
    Fungsi memindai dari kiri dan berhenti pada kelompok kata terakhir.
    """
    if crop is None or crop.size == 0:
        return {"crop": crop, "dipangkas": False}

    abu = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = abu.shape[:2]
    if h < 4 or w < 8:
        return {"crop": crop, "dipangkas": False}

    biner = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    n, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    komp = []
    tinggi_list = []
    for i in range(1, n):
        x, y, ww, hh, a = stats[i]
        if a < 8 or hh < 3:
            continue
        komp.append((int(x), int(ww)))
        tinggi_list.append(int(hh))
    if len(komp) < 2:
        return {"crop": crop, "dipangkas": False}

    komp.sort(key=lambda z: z[0])
    tinggi = float(np.median(tinggi_list)) if tinggi_list else float(h)
    ambang_gap = max(3, 0.5 * tinggi)

    # Kelompokkan komponen menjadi kata berdasarkan celah horizontal.
    kata = []
    mulai = komp[0][0]
    akhir = komp[0][0] + komp[0][1]
    for x, ww in komp[1:]:
        if x - akhir > ambang_gap:
            kata.append((mulai, akhir))
            mulai = x
        akhir = max(akhir, x + ww)
    kata.append((mulai, akhir))

    # Kurangi kelompok kata kecil/tanda baca di awal (mis. titik dua).
    kata = [k for k in kata if (k[1] - k[0]) >= 3]

    if len(kata) <= 1:
        return {"crop": crop, "dipangkas": False}

    # Nilai = kelompok kata terakhir. Pangkas dari kirinya.
    x_mulai = max(0, kata[-1][0] - max(2, int(0.15 * tinggi)))
    if x_mulai <= 0:
        return {"crop": crop, "dipangkas": False}

    hasil = crop[:, x_mulai:].copy()
    return {"crop": hasil, "dipangkas": True}


def crop_field(gambar, kotak, margin=4):

    if kotak is None:
        return None

    H, W = gambar.shape[:2]

    x1 = max(0, int(kotak["x1"]) - margin)
    y1 = max(0, int(kotak["y1"]) - margin)

    x2 = min(W, int(kotak["x2"]) + margin)
    y2 = min(H, int(kotak["y2"]) + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    return gambar[y1:y2, x1:x2].copy()

def proses(gambar_path, csv_path):
    gambar = baca_gambar(gambar_path)

    if gambar is None:
        raise ValueError("Gambar nota tidak dapat dibaca.")

    gambar = samakan_ukuran(gambar)
    H, W = gambar.shape[:2]

    kandidat = baca_kandidat(csv_path)
    produk = deteksi_produk(kandidat, W, H)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    visual = gambar.copy()

    # Semua kandidat berwarna abu-abu
    for k in kandidat:
        cv2.rectangle(
            visual,
            (k["x1"], k["y1"]),
            (k["x2"], k["y2"]),
            (170, 170, 170),
            1
        )

    data_produk = []

    for nomor, p in enumerate(produk, start=1):
        # Tambah margin kecil agar huruf tidak terpotong
        x1 = max(0, p["x1"] - 4)
        y1 = max(0, p["y1"] - 4)
        x2 = min(W, p["x2"] + 4)
        y2 = min(H, p["y2"] + 4)

        crop = gambar[y1:y2, x1:x2]

        simpan_gambar(
            OUTPUT / f"nama_produk_{nomor}.png",
            crop
        )

        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            (0, 180, 0),
            2
        )

        cv2.putText(
            visual,
            f"Produk {nomor}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        data_produk.append({
            "produk": nomor,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    simpan_gambar(
        OUTPUT / "deteksi_produk_otomatis.png",
        visual
    )

    with open(
        OUTPUT / "koordinat_produk.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["produk", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(data_produk)

    print("\n=== DETEKSI FIELD OTOMATIS ===")
    print("Jumlah kandidat teks:", len(kandidat))
    print("Calon nama produk:", len(produk))
    print("Folder hasil:", OUTPUT)

    if not produk:
        print(
            "PERINGATAN: Tidak ditemukan baris "
            "yang memenuhi pola kolom produk."
        )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            'Penggunaan: python -m processing.deteksi_field '
            '"path_nota" "path_csv"'
        )
        sys.exit(1)

    proses(Path(sys.argv[1]), Path(sys.argv[2]))
