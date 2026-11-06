from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from uji_deteksi_baris import deteksi_baris
from processing.deteksi_field import deteksi_produk, deteksi_field_lain
from processing.lokasi_field_lama import cari_area_field
from processing.pemrosesan_field import (
    isolasi_nomor, bersihkan_minus, pisah_jumlah_satuan,
    cari_tanggal, baca_gambar, mask_merah, segmentasi_per_karakter,
)

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


def kenali_crop(crop, pakai_clahe=False):
    # Segmentasi karakter + pengenalan CNN dan Template Matching.
    from processing.segmentasi_karakter import segmentasi_karakter, susun_teks
    model, labels, templates, label_tm = muat_semua_model()
    img = ke_tigakanal(crop)
    if img is None or img.size == 0:
        return {"cnn": "", "tm": "", "n": 0, "status": "crop_kosong"}
    if pakai_clahe:
        img = terapkan_clahe(img)
    biner, hasil = segmentasi_karakter(img, model, labels, templates, label_tm)
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    return {
        "cnn": susun_teks(hasil, "cnn", ta),
        "tm": susun_teks(hasil, "tm", ta),
        "n": len(hasil),
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
    # Deteksi lima field. Koordinat di ruang gambar yang lebarnya 1400.
    kecil, biner, visual, kotak = deteksi_baris(gambar_bgr)
    H, W = kecil.shape[:2]

    kandidat = [{
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
        "w": x2 - x1, "h": y2 - y1,
    } for (x1, y1, x2, y2) in kotak]

    produk = deteksi_produk(kandidat, W, H)
    info = deteksi_field_lain(kandidat, W, H, produk)

    nomor = isolasi_nomor(kecil)

    lok, err = cari_area_field(kecil)
    region_tanggal = lok["tanggal"] if (lok and "tanggal" in lok) else None
    tanggal = cari_tanggal(kecil, lokasi_tanggal=region_tanggal)

    fields = []

    fields.append({
        "id": "nomor", "label": "Nomor Nota", "jenis": "nomor",
        "bbox": nomor.get("bbox"),
        "crop_ocr": nomor.get("crop_digit"),
        "crop_show": nomor.get("crop_digit"),
        "status": nomor.get("status"),
        "pesan": nomor.get("pesan", ""),
    })

    fields.append({
        "id": "tanggal", "label": "Tanggal", "jenis": "tanggal",
        "bbox": tanggal.get("kandidat_bbox"),
        "crop_ocr": tanggal.get("kandidat"),
        "crop_show": tanggal.get("kandidat"),
        "status": tanggal.get("status"),
        "pesan": tanggal.get("pesan", ""),
    })

    for i, p in enumerate(produk, 1):
        crop = potong(kecil, p)
        bersih = bersihkan_minus(crop) if crop is not None else None
        crop_produk = bersih["crop"] if bersih else None

        rincian = info["rincian_produk"][i - 1] if i - 1 < len(info["rincian_produk"]) else {}
        kj = rincian.get("jumlah_satuan")
        kotak_j = kotak_s = crop_j = crop_s = None

        if kj is not None:
            # Perluas ke kiri agar token jumlah yang terpotong ikut masuk.
            kj = perluas_kotak_kiri(kecil, kj, batas_kiri=p["x2"] + 2)
            cjs = potong(kecil, kj)
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

    return {
        "kecil": kecil, "biner": biner, "visual": visual,
        "kandidat": kandidat, "nomor": nomor, "tanggal": tanggal,
        "produk": produk, "fields": fields, "err_tanggal": err,
    }


def gambar_kotak_field(kecil, fields):
    # Gambar bounding box tiap field di atas nota.
    vis = kecil.copy()
    warna = {
        "nomor": (0, 0, 255),
        "tanggal": (0, 140, 255),
        "produk": (0, 160, 0),
        "jumlah": (255, 0, 0),
        "satuan": (180, 0, 180),
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


def gambar_kotak_karakter(crop):
    # Gambar bounding box per karakter (target: satu kotak satu karakter).
    cimg = ke_tigakanal(crop)
    if cimg is None or cimg.size == 0:
        return None, []
    hasil = segmentasi_per_karakter(cimg)
    karakter = hasil["karakter"]
    vis = cimg.copy()
    for i, k in enumerate(karakter):
        warna = (0, 170, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.rectangle(vis, (k["x1"], k["y1"]), (k["x2"], k["y2"]), warna, 1)
    return vis, karakter


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

    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe_bgr = terapkan_clahe(gambar)
    merah = mask_merah(gambar)

    st.subheader("Mask merah nomor nota")
    st.caption(
        "Teks nomor nota dicetak merah. Mask ini menjadi dasar isolasi digit "
        "nomor (huruf 'NO' dan titik dua dibuang, hanya digit yang diambil)."
    )
    a, b = st.columns(2)
    a.image(merah, caption="Mask merah (nomor nota)", use_container_width=True)
    b.image(gambar_rgb(cv2.bitwise_and(gambar, gambar, mask=merah)),
            caption="Piksel merah saja", use_container_width=True)

    st.subheader("Tahapan preprocessing")
    c, d = st.columns(2)
    c.image(abu, caption="Grayscale", use_container_width=True)
    d.image(otsu, caption="Binarisasi Otsu", use_container_width=True)
    e, f = st.columns(2)
    e.image(gambar_rgb(gambar), caption="Asli", use_container_width=True)
    f.image(gambar_rgb(clahe_bgr), caption="Dengan CLAHE", use_container_width=True)

    st.caption(
        "Jalur tanpa CLAHE dan dengan CLAHE memakai lokasi field yang sama, "
        "sehingga perbedaan hasil berasal dari CLAHE."
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
    st.caption("Target segmentasi: satu bounding box untuk satu karakter.")
    for f in det["fields"]:
        with st.expander(f["label"]):
            crop = f["crop_show"]
            if crop is None or getattr(crop, "size", 0) == 0:
                st.warning("Crop field ini tidak tersedia.")
                continue

            kiri, kanan = st.columns(2)
            kiri.caption("Hasil crop")
            kiri.image(gambar_rgb(crop), use_container_width=True)

            vis, karakter = gambar_kotak_karakter(crop)
            kanan.caption(f"Segmentasi per karakter ({len(karakter)} kotak)")
            if vis is not None:
                kanan.image(gambar_rgb(vis), use_container_width=True)

            if karakter:
                st.caption("Potongan tiap karakter:")
                n = min(len(karakter), 14)
                kol = st.columns(n)
                for i in range(n):
                    k = karakter[i]
                    seg = crop[k["y1"]:k["y2"], k["x1"]:k["x2"]]
                    if getattr(seg, "size", 0):
                        kol[i].image(gambar_rgb(seg), use_container_width=True)


def kenali_satu(crop, pakai_clahe):
    try:
        return kenali_crop(crop, pakai_clahe=pakai_clahe)
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
        tanpa = kenali_satu(crop, False)
        clahe = kenali_satu(crop, True) if sertakan_clahe else \
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

    sertakan = st.checkbox(
        "Sertakan jalur CLAHE (perbandingan)", value=True, key="cek_clahe"
    )

    kunci = (st.session_state.get("nama_file"), bool(sertakan))
    if st.session_state.get("ocr_kunci") != kunci or st.session_state.get("ocr_semua") is None:
        with st.spinner("Mengenali semua field (CNN + Template Matching)..."):
            st.session_state["ocr_semua"] = jalankan_ocr_semua(det, sertakan)
            st.session_state["ocr_kunci"] = kunci

    hasil = st.session_state["ocr_semua"]

    st.subheader("Hasil pengenalan semua field")
    baris = []
    for f in det["fields"]:
        h = hasil.get(f["id"], {})
        t = h.get("tanpa", {})
        c = h.get("clahe", {})
        row = {
            "Field": f["label"],
            "CNN": t.get("cnn", "") or "—",
            "Template Matching": t.get("tm", "") or "—",
            "Jumlah karakter": t.get("n", 0),
        }
        if sertakan:
            row["CNN (CLAHE)"] = c.get("cnn", "") or "—"
            row["TM (CLAHE)"] = c.get("tm", "") or "—"
        baris.append(row)
    st.dataframe(baris, use_container_width=True, hide_index=True)

    st.subheader("Detail per field")
    for f in det["fields"]:
        h = hasil.get(f["id"], {})
        t = h.get("tanpa", {})
        with st.expander(f["label"]):
            if f["crop_show"] is not None and getattr(f["crop_show"], "size", 0):
                st.image(gambar_rgb(f["crop_show"]), width=300)
            st.write("CNN:", t.get("cnn", "") or "—")
            st.write("Template Matching:", t.get("tm", "") or "—")
            if t.get("status", "").startswith("gagal"):
                st.warning(t["status"])


def halaman_preparation():
    st.header("Preparation")

    st.subheader("Latar belakang")
    st.write(
        " Analisis Pengaruh Image Preprocessing terhadap Akurasi "
        "Pengenalan Teks Nota Transaksi pada CV Merriena Jaya."
    )
    st.write(
        "MERRI-STOCK merupakan aplikasi pengenalan teks pada nota transaksi CV Merriena Jaya. "
        "Penelitian ini menganalisis pengaruh image preprocessing terhadap hasil pengenalan "
        "karakter menggunakan CNN dan Template Matching sebagai pembanding. Hasil pengenalan "
        "dievaluasi berdasarkan ground truth nota."
    )

    st.subheader("Jumlah dataset karakter")
    dataset = BASE / "dataset"
    ekstensi = (".png", ".jpg", ".jpeg")
    total = 0
    rincian = {}
    if dataset.exists():
        for jenis in ("huruf", "angka", "simbol"):
            folder = dataset / jenis
            if folder.exists():
                n = len([
                    p for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in ekstensi
                ])
                rincian[jenis] = n
                total += n
    kol = st.columns(4)
    kol[0].metric("Total gambar", total)
    kol[1].metric("Huruf", rincian.get("huruf", 0))
    kol[2].metric("Angka", rincian.get("angka", 0))
    kol[3].metric("Simbol", rincian.get("simbol", 0))

    st.subheader("Model dan metode")
    st.write("- Kelas karakter CNN: 70 (26 huruf besar, 26 huruf kecil, 10 angka, 8 simbol).")
    st.write("- Model: `models/cnn_karakter.keras` ")
    st.write("- Pembanding: Template Matching (cosine similarity).")
    st.write("- Image preprocessing dianalisis pengaruhnya terhadap hasil pengenalan karakter.")
    st.write("- Hasil pengenalan CNN dan Template Matching dievaluasi terhadap ground truth menggunakan Character Error Rate (CER)")

    st.subheader("Lima field target")
    st.write("Nomor nota, Tanggal, Nama produk, Jumlah, Satuan.")


GT_CONTOH = {
    "nota_01.jpeg": "0086",
    "nota_26.jpeg": "0099",
    "nota_52.jpeg": "0299",
    "nota_54.jpeg": "0292",
}


def halaman_evaluasi():
    st.header("Evaluasi (CER)")
    st.caption(
        "Tulis ground truth tiap field. Program langsung membandingkannya "
        "dengan hasil pengenalan CNN dan Template Matching (CER)."
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
        c = h.get("clahe", {})
        bawaan = GT_CONTOH.get(nama, "") if f["jenis"] == "nomor" else ""
        gt = st.text_input(
            f"Ground truth — {f['label']}", value=bawaan, key=f"gt_{f['id']}"
        )
        cnn = t.get("cnn", "")
        tm = t.get("tm", "")
        cer_c = hitung_cer(gt, cnn) if gt.strip() else None
        cer_t = hitung_cer(gt, tm) if gt.strip() else None
        row = {
            "Field": f["label"],
            "Ground Truth": gt,
            "CNN": cnn or "—",
            "CER CNN": None if cer_c is None else round(cer_c, 4),
            "TM": tm or "—",
            "CER TM": None if cer_t is None else round(cer_t, 4),
        }
        if c.get("cnn") is not None and (c.get("cnn") or c.get("tm")):
            cc = hitung_cer(gt, c.get("cnn", "")) if gt.strip() else None
            ct = hitung_cer(gt, c.get("tm", "")) if gt.strip() else None
            row["CER CNN (CLAHE)"] = None if cc is None else round(cc, 4)
            row["CER TM (CLAHE)"] = None if ct is None else round(ct, 4)
        baris.append(row)

    st.subheader("Perbandingan per field")
    st.dataframe(baris, use_container_width=True, hide_index=True)

    st.subheader("Rata-rata CER")
    st.caption("Hanya field yang ground truth-nya diisi dan hasilnya tidak kosong.")

    def rata(kunci):
        nilai = [r[kunci] for r in baris if r.get(kunci) is not None]
        return (round(sum(nilai) / len(nilai), 4), len(nilai)) if nilai else (None, 0)

    ringkas = []
    for label, kunci in (
        ("CNN (tanpa CLAHE)", "CER CNN"),
        ("Template Matching (tanpa CLAHE)", "CER TM"),
        ("CNN (dengan CLAHE)", "CER CNN (CLAHE)"),
        ("Template Matching (dengan CLAHE)", "CER TM (CLAHE)"),
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
    "Preparation",
    "Upload Nota",
    "Preprocessing",
    "Segmentasi",
    "Pengenalan",
    "Evaluasi",
]

halaman = st.sidebar.radio("Halaman", HALAMAN, key="navigasi")

if st.session_state.get("nama_file"):
    st.sidebar.info(f"Nota aktif: {st.session_state['nama_file']}")

if halaman == "Preparation":
    halaman_preparation()
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