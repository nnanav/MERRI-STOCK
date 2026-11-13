from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from processing.deteksi_baris import deteksi_baris
from processing.deteksi_field import (
    deteksi_produk, deteksi_field_lain,
    deteksi_penerima_proyek, deteksi_harga_satuan,
    pangkas_awalan_tipis, pangkas_hingga_rp,
)
from processing.pemrosesan_field import (
    isolasi_nomor, bersihkan_minus, pisah_jumlah_satuan,
    cari_tanggal, baca_gambar, mask_merah, segmentasi_per_karakter,
)
from processing.ground_truth import gt_field
from processing.preprocessing_nota import preprocess_nota

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="MERRI-STOCK", layout="wide")


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
    Buang tanda kali di akhir satuan.

    Pada nota, kolom satuan diikuti tanda kali, sehingga hasil OCR sering
    menjadi "Pcs x" padahal isi satuannya hanya "Pcs".
    """
    if not teks:
        return teks
    import re
    teks = re.sub(r"[\s.,]*[xX×\*]+\s*$", "", teks)
    return teks.strip()


def kenali_crop(crop, pakai_clahe=False, jenis=None):
    # Satu pipeline OCR: TM menemukan posisi karakter, CNN mengklasifikasi.
    from processing.segmentasi_karakter import segmentasi_utama, susun_teks
    from processing.pengenalan_karakter import gabung_segmen
    model, labels, templates, label_tm = muat_semua_model()
    img = ke_tigakanal(crop)
    if img is None or img.size == 0:
        return {"teks": "", "cnn": "", "tm": "", "n": 0, "detail": [],
                "metode": "-", "status": "crop_kosong"}
    if pakai_clahe:
        img = terapkan_clahe(img)
    biner, hasil, metode = segmentasi_utama(
        img, model, labels, templates, label_tm
    )
    if jenis == "satuan":
        hasil = buang_tanda_kali(hasil)
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    hasil_final, detail = gabung_segmen(hasil)
    teks = bersihkan_tepi(susun_teks(hasil_final, "final", ta))
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
    kecil, biner, visual, kotak = deteksi_baris(gambar_bgr)
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

    return {
        "kecil": kecil, "biner": biner, "visual": visual,
        "kandidat": kandidat, "nomor": nomor,
        "produk": produk, "fields": fields,
        "preproses": preproses,
    }


def gambar_kotak_field(kecil, fields):
    # Gambar bounding box tiap field di atas nota.
    vis = kecil.copy()
    warna = {
        "nomor": (0, 0, 255),
        "penerima": (0, 140, 255),
        "proyek": (255, 140, 0),
        "produk": (0, 160, 0),
        "jumlah": (255, 0, 0),
        "satuan": (0, 200, 200),
        "harga": (180, 0, 180),
    }
    for f in fields:
        bb = f.get("bbox")
        if not bb:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
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
    # Matching + dynamic programming), agar kotak yang ditampilkan sama
    # dengan yang dipakai pengenalan. Fallback ke proyeksi bila gagal.
    cimg = ke_tigakanal(crop)
    if cimg is None or cimg.size == 0:
        return None, [], None, "", []

    try:
        from processing.segmentasi_karakter import (
            segmentasi_utama, susun_teks, SKALA,
        )
        from processing.pengenalan_karakter import gabung_segmen
        model, labels, templates, label_tm = muat_semua_model()
        biner, hasil, metode = segmentasi_utama(
            cimg, model, labels, templates, label_tm
        )
        if jenis == "satuan":
            hasil = buang_tanda_kali(hasil)
        final, detail = gabung_segmen(hasil)
        tinggi = [s["y2"] - s["y1"] for s in hasil]
        ta = float(np.median(tinggi)) if tinggi else 12.0
        teks = susun_teks(final, "final", ta) if final else ""
        if jenis == "satuan":
            teks = rapikan_satuan(teks)

        # Koordinat hasil berada di ruang upsampled (SKALA x); kembalikan
        # ke ukuran crop asli agar kotak pas.
        h, w = cimg.shape[:2]
        karakter = []
        for s in hasil:
            karakter.append({
                "x1": int(s["x1"] / SKALA), "y1": int(s["y1"] / SKALA),
                "x2": min(w, int(s["x2"] / SKALA)),
                "y2": min(h, int(s["y2"] / SKALA)),
                "cnn": s.get("cnn", ""), "tm": s.get("tm", ""),
                "final": s.get("final", ""),
            })
        biner_tampil = cv2.resize(
            biner, (w, h), interpolation=cv2.INTER_AREA
        ) if biner is not None else None
        return _gambar_kotak(cimg, karakter), karakter, biner_tampil, teks, detail
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
    st.header("Upload Nota")
    st.caption("Unggah satu gambar nota transaksi (jpg/png), lalu klik Proses Nota.")

    berkas = st.file_uploader(
        "Unggah gambar nota", type=["png", "jpg", "jpeg"], key="unggah_nota"
    )

    if berkas is not None:
        data = np.frombuffer(berkas.getvalue(), np.uint8)
        gambar_baru = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if gambar_baru is None:
            st.error("Gambar gagal dibaca. Coba berkas lain.")
            return
        if st.session_state.get("nama_file") != berkas.name:
            st.session_state["nama_file"] = berkas.name
            st.session_state["gambar_bgr"] = gambar_baru
            st.session_state["deteksi"] = None
            st.session_state["ocr_semua"] = None
            st.session_state["ocr_kunci"] = None
            st.session_state["preproses"] = None

    gambar = st.session_state.get("gambar_bgr")
    nama = st.session_state.get("nama_file")
    if gambar is None:
        st.info("Silakan unggah gambar nota untuk memulai.")
        return

    st.success(f"Nota: {nama}")
    st.image(gambar_rgb(gambar), use_container_width=True)

    st.divider()
    kol = st.columns([1, 2])
    if kol[0].button("Proses Nota", type="primary", key="proses_nota"):
        with st.spinner("Memproses nota (deteksi field)..."):
            st.session_state["deteksi"] = jalankan_deteksi(gambar)
            st.session_state["ocr_semua"] = None
            st.session_state["ocr_kunci"] = None

    if st.session_state.get("deteksi") is not None:
        det = st.session_state["deteksi"]
        kol[1].success(f"Nota berhasil diproses — {len(det['fields'])} field terdeteksi.")
        st.info("Lanjutkan ke halaman Preprocessing, Segmentasi, Pengenalan, lalu Evaluasi.")


def halaman_preprocessing():
    st.header("Preprocessing")
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

    st.subheader("1. Crop dan koreksi perspektif")
    st.caption(pp["pesan"])
    a, b = st.columns(2)
    a.image(gambar_rgb(pp["asli"]), caption="Nota asli", width="stretch")
    b.image(gambar_rgb(pp["crop"]), caption="Hasil crop kertas nota",
            width="stretch")

    st.subheader("2. Peningkatan kualitas gambar")
    st.caption(
        f"Metode terpilih: {pp['metode']} pada kanal luminansi. Rentang "
        "intensitas kertas direntangkan sehingga tulisan yang tipis lebih "
        "terpisah dari latar, tanpa mengubah warna dan tanpa menambah "
        "kontras tekstur kertas. Metode diterapkan pada area kertas yang "
        "terdeteksi; bila batas kertas tidak ditemukan, gambar dipakai apa "
        "adanya (fallback)."
    )
    st.image(gambar_rgb(hasil), caption="Hasil akhir preprocessing",
             width="stretch")

    st.subheader("Mask merah nomor nota")
    merah = mask_merah(hasil)
    st.caption(
        "Teks nomor nota dicetak merah. Mask ini menjadi dasar isolasi digit "
        "nomor (huruf 'NO' dan titik dua dibuang, hanya digit yang diambil)."
    )
    c, d = st.columns(2)
    c.image(merah, caption="Mask merah (nomor nota)", width="stretch")
    d.image(gambar_rgb(cv2.bitwise_and(hasil, hasil, mask=merah)),
            caption="Piksel merah saja", width="stretch")

    with st.expander("Binerisasi yang dipakai segmentasi"):
        abu = cv2.cvtColor(hasil, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(
            abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        st.image(abu, caption="Grayscale", width="stretch")
        st.image(otsu, caption="Binarisasi Otsu", width="stretch")
        st.caption(
            "Hasil akhir preprocessing di atas yang diteruskan ke deteksi "
            "field, segmentasi karakter, dan pengenalan."
        )


def halaman_segmentasi():
    st.header("Segmentasi")
    if st.session_state.get("gambar_bgr") is None:
        st.info("Unggah nota dan klik 'Proses Nota' di halaman Upload terlebih dahulu.")
        return

    det = ambil_deteksi()
    if det is None:
        st.info("Klik 'Proses Nota' di halaman Upload terlebih dahulu.")
        return

    st.subheader("Bounding box tiap field")
    st.caption("Setiap kotak menandai lokasi satu field pada nota.")
    st.image(gambar_rgb(gambar_kotak_field(det["kecil"], det["fields"])),
             use_container_width=True)

    st.subheader("Crop dan segmentasi per karakter")
    st.caption(
        "Kotak berasal dari pipeline pengenalan sebenarnya (CNN + Template "
        "Matching), target satu bounding box untuk satu karakter."
    )
    for f in det["fields"]:
        with st.expander(f["label"]):
            crop = f["crop_show"]
            if crop is None or getattr(crop, "size", 0) == 0:
                st.warning("Crop field ini tidak tersedia.")
                continue

            vis, karakter, biner, teks, _detail = segmen_field(
                crop, f.get("jenis")
            )

            kiri, kanan = st.columns(2)
            kiri.caption("Crop asli")
            kiri.image(gambar_rgb(crop), use_container_width=True)
            if biner is not None:
                kanan.caption("Binarisasi")
                kanan.image(biner, use_container_width=True)

            st.caption(f"Bounding box karakter ({len(karakter)} kotak)")
            if vis is not None:
                st.image(gambar_rgb(vis), use_container_width=True)

            if teks:
                st.caption(f"Hasil OCR: {teks}")

            if karakter:
                st.caption("Potongan tiap karakter:")
                n = min(len(karakter), 14)
                kol = st.columns(n)
                for i in range(n):
                    k = karakter[i]
                    seg = crop[k["y1"]:k["y2"], k["x1"]:k["x2"]]
                    if getattr(seg, "size", 0):
                        kol[i].image(gambar_rgb(seg), use_container_width=True)
                        if k.get("final"):
                            kol[i].caption(k["final"])


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
    st.header("Pengenalan")
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

    st.subheader("Hasil pengenalan semua field")
    st.caption(
        "Segmentasi menemukan kandidat karakter, Template Matching mencocokkan "
        "polanya, dan CNN mengenali tiap karakter. Hasil di bawah adalah satu "
        "teks OCR per field."
    )
    baris = []
    for f in det["fields"]:
        h = hasil.get(f["id"], {})
        t = h.get("tanpa", {})
        row = {
            "Field": f["label"],
            "Hasil OCR": t.get("teks", "") or "—",
            "Jumlah karakter": t.get("n", 0),
        }
        baris.append(row)
    st.dataframe(baris, use_container_width=True, hide_index=True)

    st.subheader("Detail hasil per field")
    for f in det["fields"]:
        h = hasil.get(f["id"], {})
        t = h.get("tanpa", {})
        with st.expander(f["label"]):
            if f["crop_show"] is not None and getattr(f["crop_show"], "size", 0):
                st.image(gambar_rgb(f["crop_show"]), width=300)
            st.write("Hasil OCR:", t.get("teks", "") or "—")
            st.caption(f"Jumlah karakter: {t.get('n', 0)}")
            if t.get("status", "").startswith("gagal"):
                st.warning(t["status"])


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
    st.header("Business Understanding")

    # 1. Latar Belakang
    st.subheader("1. Latar Belakang")

    st.write(
        "MERRI-STOCK digunakan untuk mengenali informasi pada nota transaksi "
        "CV Merriena Jaya. Nota yang sebelumnya dibaca dan diketik manual "
        "kini diubah menjadi teks oleh sistem, sehingga pengambilan data "
        "transaksi menjadi lebih cepat dan konsisten."
    )

    st.write(
        "Hasil pengenalan teks sangat dipengaruhi oleh kondisi gambar nota. "
        "Perbedaan kualitas gambar, tingkat kontras, serta bentuk karakter "
        "yang tidak seragam membuat karakter pada nota sulit dikenali. "
        "Karena itu kualitas gambar perlu diperbaiki sebelum proses "
        "pengenalan karakter dilakukan."
    )

    # 2. Tujuan Sistem
    st.subheader("2. Tujuan Sistem")

    st.write(
        "Sistem ini bertujuan menganalisis pengaruh image preprocessing "
        "terhadap akurasi pengenalan teks nota transaksi. Pengenalan "
        "karakter dilakukan dengan CNN dan Template Matching yang "
        "digabungkan dalam satu proses pengenalan karakter, lalu akurasinya "
        "diukur menggunakan CER (Character Error Rate)."
    )

    st.info(
        "Image preprocessing menjadi variabel yang dianalisis, sedangkan CNN "
        "dan Template Matching adalah metode pengenalan yang bekerja "
        "bersama pada satu proses yang sama."
    )

    # 3. Alur Kerja Sistem
    st.subheader("3. Alur Kerja Sistem")

    st.markdown(
        """
        ```
        Upload Nota  →  Image Preprocessing  →  Deteksi Field
              →  Segmentasi Karakter  →  Pengenalan CNN + Template Matching
              →  Evaluasi CER
        ```
        """
    )

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
        ("Evaluasi CER",
         "Teks hasil OCR dibandingkan dengan ground truth, lalu diukur "
         "dengan CER."),
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
    st.caption(
        "Angka pada bagian ini dihitung otomatis dari isi folder dataset "
        "dan label model, sehingga ikut berubah bila isi dataset berubah."
    )

    _, kanan = st.columns([5, 1])
    with kanan:
        if st.button("Hitung ulang", key="muat_ulang_dataset"):
            info_dataset_latih.clear()
            st.rerun()

    info = info_dataset_latih()

    if not info["total"]:
        st.warning("Folder dataset tidak ditemukan atau masih kosong.")
    else:
        atas = st.columns(4)
        atas[0].metric("Total gambar karakter", f"{info['total']:,}")
        atas[1].metric("Huruf", f"{info['huruf']:,}")
        atas[2].metric("Angka", f"{info['angka']:,}")
        atas[3].metric("Simbol", f"{info['simbol']:,}")

        bawah = st.columns(4)
        bawah[0].metric("Kelas karakter model", info["kelas_model"])
        bawah[1].metric("Dipakai untuk pelatihan", f"{info['dipakai']:,}")
        bawah[2].metric("Data latih (80%)", f"{info['latih']:,}")
        bawah[3].metric("Data validasi (20%)", f"{info['validasi']:,}")

        st.caption(
            f"Tersedia {info['total']:,} gambar karakter. Sebanyak "
            f"{info['dikecualikan']:,} gambar dikecualikan karena tidak lolos "
            f"pemeriksaan dataset, sehingga {info['dipakai']:,} gambar "
            f"benar-benar dipakai. Dari jumlah tersebut {info['latih']:,} "
            f"menjadi data latih dan {info['validasi']:,} menjadi data "
            f"validasi (pembagian stratified {100 - int(RASIO_VALIDASI * 100)}:"
            f"{int(RASIO_VALIDASI * 100)} per kelas, yaitu latih:validasi)."
        )

        with st.expander("Rincian pemeriksaan dataset"):
            st.markdown("**Status berkas menurut laporan validasi**")

            if info["laporan_ada"]:
                tabel = [
                    {"Status": nama or "(kosong)", "Jumlah": jumlah}
                    for nama, jumlah in sorted(info["status"].items())
                ]
                st.dataframe(tabel, hide_index=True)
            else:
                st.caption(
                    "Laporan validasi belum tersedia. Jalankan "
                    "`python scripts/validasi_dataset.py` agar jumlah data "
                    "yang dikecualikan ikut terhitung."
                )

            st.markdown("**Kelas karakter**")
            st.caption(
                f"Dataset berisi {info['kelas_dataset']} kelas dan model "
                f"mengenali {info['kelas_model']} kelas: huruf A-Z, "
                f"huruf a-z, angka 0-9, dan simbol."
            )

            if info["folder_kosong"]:
                st.caption(
                    "Folder simbol tanpa gambar (belum menjadi kelas): "
                    + ", ".join(info["folder_kosong"])
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


def halaman_evaluasi():
    st.header("Evaluasi (CER)")
    st.caption(
        "Tulis ground truth tiap field. Program membandingkan hasil OCR "
        "dengan ground truth menggunakan CER."
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
    baris = []

    for f in det["fields"]:
        h = hasil.get(f["id"], {})
        t = h.get("tanpa", {})
        # Ground truth diisi otomatis bila nota punya acuan, supaya CER
        # langsung terhitung. Nilainya tetap bisa diubah manual.
        bawaan = gt_field(nama, f["id"])
        gt = st.text_input(
            f"Ground truth — {f['label']}", value=bawaan, key=f"gt_{f['id']}"
        )
        teks = t.get("teks", "")
        cer = hitung_cer(gt, teks) if gt.strip() else None
        row = {
            "Field": f["label"],
            "Ground Truth": gt,
            "Hasil OCR": teks or "—",
            "CER": None if cer is None else round(cer, 4),
        }
        baris.append(row)

    st.subheader("Hasil OCR vs Ground Truth")
    st.dataframe(baris, use_container_width=True, hide_index=True)

    st.subheader("Rata-rata CER")
    st.caption("Hanya field yang ground truth-nya diisi dan hasilnya tidak kosong.")

    def rata(kunci):
        nilai = [r[kunci] for r in baris if r.get(kunci) is not None]
        return (round(sum(nilai) / len(nilai), 4), len(nilai)) if nilai else (None, 0)

    ringkas = []
    for label, kunci in (
        ("OCR (CNN + Template Matching)", "CER"),
    ):
        if any(kunci in r for r in baris):
            nilai, jumlah = rata(kunci)
            ringkas.append({
                "Metode": label, "Rata-rata CER": nilai, "Jumlah field": jumlah,
            })

    st.dataframe(ringkas, use_container_width=True, hide_index=True)


# Routing enam halaman
st.sidebar.title("MERRI-STOCK")
st.sidebar.caption("Sistem Pengenalan Teks Nota Transaksi CV Merriena Jaya")

HALAMAN = [
    "Business Understanding",
    "Upload Nota",
    "Preprocessing",
    "Segmentasi",
    "Pengenalan",
    "Evaluasi",
]

halaman = st.sidebar.radio("Halaman", HALAMAN, key="navigasi")

if st.session_state.get("nama_file"):
    st.sidebar.info(f"Nota aktif: {st.session_state['nama_file']}")

if halaman == "Business Understanding":
    halaman_business_understanding()
elif halaman == "Upload Nota":
    halaman_upload()
elif halaman == "Preprocessing":
    halaman_preprocessing()
elif halaman == "Segmentasi":
    halaman_segmentasi()
elif halaman == "Pengenalan":
    halaman_pengenalan()
elif halaman == "Evaluasi":
    halaman_evaluasi()