from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from uji_deteksi_baris import deteksi_baris
from processing.deteksi_field import deteksi_produk, deteksi_field_lain
from processing.lokasi_field_lama import cari_area_field
from processing.pemrosesan_field import (
    isolasi_nomor, bersihkan_minus, pisah_jumlah_satuan,
    cari_tanggal, baca_gambar,
)

BASE = Path(__file__).resolve().parent
FOLDER_NOTA = {
    "data_nota_asli": BASE / "data_nota_asli",
    "data_uji": BASE / "data_uji",
}

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


def terapkan_clahe(bgr):
    # CLAHE pada kanal L; dipakai untuk jalur 'dengan CLAHE'.
    if bgr is None:
        return None
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
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
    # Dimuat sekali; hanya dipanggil saat halaman Pengenalan.
    from processing.pengenalan_karakter import muat_cnn, muat_templates
    model, labels = muat_cnn()
    templates, label_tm = muat_templates()
    return model, labels, templates, label_tm


def jalankan_deteksi(gambar_bgr):
    # Deteksi lima field. Semua koordinat di ruang gambar yang lebarnya
    # sudah disamakan (1400) agar konsisten dengan deteksi_baris.
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

    daftar_produk = []
    for i, p in enumerate(produk):
        crop = potong(kecil, p)
        bersih = bersihkan_minus(crop) if crop is not None else None
        rincian = info["rincian_produk"][i] if i < len(info["rincian_produk"]) else {}
        kj = rincian.get("jumlah_satuan")
        jumlah = satuan = None
        status_js = "tidak_terdeteksi"
        if kj is not None:
            cjs = potong(kecil, kj)
            if cjs is not None:
                pisah = pisah_jumlah_satuan(cjs)
                jumlah = pisah.get("jumlah")
                satuan = pisah.get("satuan")
                status_js = pisah.get("status")
        daftar_produk.append({
            "kotak": p,
            "crop": crop,
            "bersih": bersih,
            "jumlah": jumlah,
            "satuan": satuan,
            "status_js": status_js,
        })

    return {
        "kecil": kecil,
        "biner": biner,
        "visual": visual,
        "kandidat": kandidat,
        "nomor": nomor,
        "tanggal": tanggal,
        "produk": daftar_produk,
        "err_tanggal": err,
    }


def kenali_crop(crop_bgr, pakai_clahe=False):
    # Segmentasi karakter + pengenalan CNN dan Template Matching.
    from processing.segmentasi_karakter import segmentasi_karakter, susun_teks
    model, labels, templates, label_tm = muat_semua_model()
    img = terapkan_clahe(crop_bgr) if pakai_clahe else crop_bgr
    if img is None or img.size == 0:
        return {"cnn": "", "tm": "", "n": 0, "biner": None, "hasil": []}
    biner, hasil = segmentasi_karakter(img, model, labels, templates, label_tm)
    tinggi = [s["y2"] - s["y1"] for s in hasil]
    ta = float(np.median(tinggi)) if tinggi else 12.0
    return {
        "cnn": susun_teks(hasil, "cnn", ta),
        "tm": susun_teks(hasil, "tm", ta),
        "n": len(hasil),
        "biner": biner,
        "hasil": hasil,
    }


def daftar_file_nota():
    hasil = []
    for label, folder in FOLDER_NOTA.items():
        if not folder.exists():
            continue
        for pola in ("*.jpeg", "*.jpg", "*.png"):
            for p in sorted(folder.glob(pola)):
                hasil.append((f"{label}/{p.name}", p))
    return hasil


def halaman_preparation():
    st.header("Preparation")
    st.caption("Ringkasan data dan model sebelum pengujian.")

    kol = st.columns(3)
    jumlah_asli = len(list((FOLDER_NOTA["data_nota_asli"]).glob("*.jpeg"))) \
        if FOLDER_NOTA["data_nota_asli"].exists() else 0
    jumlah_uji = len(list((FOLDER_NOTA["data_uji"]).glob("*.*"))) \
        if FOLDER_NOTA["data_uji"].exists() else 0
    kol[0].metric("Nota asli", jumlah_asli)
    kol[1].metric("Nota uji", jumlah_uji)
    kol[2].metric("Kelas CNN", 70)

    st.subheader("Model dan metode")
    st.write(
        "- Model CNN: `models/cnn_karakter.keras` (70 kelas, tidak dilatih ulang).\n"
        "- Pembanding: Template Matching (cosine similarity).\n"
        "- Evaluasi: Character Error Rate (CER).\n"
        "- Dua jalur preprocessing: tanpa CLAHE dan dengan CLAHE."
    )

    model_path = BASE / "models" / "cnn_karakter.keras"
    label_path = BASE / "models" / "labels.json"
    st.write("Model tersedia:", model_path.exists())
    st.write("Label tersedia:", label_path.exists())

    st.subheader("Lima field target")
    st.write("Nomor nota, Tanggal, Nama produk, Jumlah, Satuan.")


def halaman_upload():
    st.header("Upload Nota")
    sumber = st.radio(
        "Sumber gambar", ["Dataset", "Unggah berkas"],
        horizontal=True, key="sumber_nota",
    )

    gambar = None
    nama = None

    if sumber == "Dataset":
        opsi = daftar_file_nota()
        if not opsi:
            st.warning("Tidak ada nota pada folder dataset.")
            return
        label_list = [l for l, _ in opsi]
        pilihan = st.selectbox("Pilih nota", label_list, key="pilih_nota")
        path = dict(opsi)[pilihan]
        gambar = baca_gambar(path)
        nama = path.name
    else:
        berkas = st.file_uploader(
            "Unggah nota", type=["png", "jpg", "jpeg"], key="unggah_nota"
        )
        if berkas is not None:
            data = np.frombuffer(berkas.read(), np.uint8)
            gambar = cv2.imdecode(data, cv2.IMREAD_COLOR)
            nama = berkas.name

    if gambar is None:
        st.info("Pilih atau unggah nota terlebih dahulu.")
        return

    if st.session_state.get("nama_file") != nama:
        st.session_state["nama_file"] = nama
        st.session_state["gambar_bgr"] = gambar
        st.session_state["deteksi"] = None
        st.session_state["ocr_nomor"] = None
        st.session_state["ocr_hasil"] = {}

    st.success(f"Nota aktif: {nama}")
    st.image(gambar_rgb(gambar), caption=nama, use_container_width=True)
    st.caption("Lanjutkan ke halaman Preprocessing atau Segmentasi.")


def halaman_preprocessing():
    st.header("Preprocessing")
    gambar = st.session_state.get("gambar_bgr")
    if gambar is None:
        st.info("Pilih nota di halaman Upload terlebih dahulu.")
        return

    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe_bgr = terapkan_clahe(gambar)

    from processing.pemrosesan_field import mask_merah
    merah = mask_merah(gambar)

    a, b = st.columns(2)
    a.image(abu, caption="Grayscale", use_container_width=True)
    b.image(otsu, caption="Binarisasi Otsu", use_container_width=True)

    c, d = st.columns(2)
    c.image(gambar_rgb(clahe_bgr), caption="Dengan CLAHE", use_container_width=True)
    d.image(merah, caption="Mask merah (nomor nota)", use_container_width=True)

    st.caption(
        "Jalur tanpa CLAHE dan dengan CLAHE memakai lokasi field yang sama "
        "pada halaman Segmentasi, sehingga perbedaan hasil murni dari CLAHE."
    )


def ambil_deteksi():
    if st.session_state.get("deteksi") is None:
        gambar = st.session_state.get("gambar_bgr")
        if gambar is None:
            return None
        with st.spinner("Mendeteksi lima field..."):
            st.session_state["deteksi"] = jalankan_deteksi(gambar)
    return st.session_state["deteksi"]


def halaman_segmentasi():
    st.header("Segmentasi")
    if st.session_state.get("gambar_bgr") is None:
        st.info("Pilih nota di halaman Upload terlebih dahulu.")
        return

    det = ambil_deteksi()
    if det is None:
        return

    # Nomor nota
    nom = det["nomor"]
    with st.expander("Segmentasi Nomor Nota", expanded=True):
        if nom.get("status") == "ok":
            st.success(nom["pesan"])
            kol = st.columns(3)
            if nom.get("crop_baris") is not None:
                kol[0].image(gambar_rgb(nom["crop_baris"]),
                             caption="Baris nomor (NO : xxxx)",
                             use_container_width=True)
            if nom.get("crop_digit") is not None:
                kol[1].image(gambar_rgb(nom["crop_digit"]),
                             caption="Hanya digit",
                             use_container_width=True)
            if nom.get("mask_digit") is not None:
                kol[2].image(nom["mask_digit"],
                             caption="Mask digit",
                             use_container_width=True)
        else:
            st.warning(nom.get("pesan", "Wilayah belum tersedia."))
            if nom.get("crop_baris") is not None:
                st.image(gambar_rgb(nom["crop_baris"]),
                         caption="Crop diagnostik baris",
                         use_container_width=True)

    # Tanggal
    tg = det["tanggal"]
    with st.expander("Segmentasi Tanggal"):
        st.write("Status:", tg.get("status"), "| Asal wilayah:", tg.get("asal"))
        st.caption(tg.get("pesan", ""))
        if tg.get("err_tanggal") is None:
            kol = st.columns(2)
            if tg.get("wilayah") is not None:
                kol[0].image(gambar_rgb(tg["wilayah"]),
                             caption="Wilayah pencarian",
                             use_container_width=True)
            if tg.get("kandidat") is not None:
                kol[1].image(gambar_rgb(tg["kandidat"]),
                             caption="Kandidat baris tanggal",
                             use_container_width=True)
        if tg.get("mask_buang") is not None:
            c2 = st.columns(2)
            c2[0].image(tg["mask_buang"],
                        caption="Mask gangguan (tanda tangan)",
                        use_container_width=True)
            if tg.get("bersih_biner") is not None:
                c2[1].image(tg["bersih_biner"],
                            caption="Hasil pembersihan (biner)",
                            use_container_width=True)

    # Produk + jumlah + satuan
    for i, p in enumerate(det["produk"], 1):
        with st.expander(f"Produk {i}: Nama, Jumlah, Satuan"):
            kol = st.columns(2)
            if p["crop"] is not None:
                kol[0].image(gambar_rgb(p["crop"]),
                             caption="Sebelum (crop asli)",
                             use_container_width=True)
            if p["bersih"] is not None:
                kol[1].image(gambar_rgb(p["bersih"]["crop"]),
                             caption="Sesudah (minus pembuka dibuang)",
                             use_container_width=True)
                st.caption(p["bersih"]["pesan"])

            kjs = st.columns(2)
            if p["jumlah"] is not None:
                kjs[0].image(gambar_rgb(p["jumlah"]),
                             caption="Jumlah", use_container_width=True)
            else:
                kjs[0].warning("Jumlah: " + str(p["status_js"]))
            if p["satuan"] is not None:
                kjs[1].image(gambar_rgb(p["satuan"]),
                             caption="Satuan", use_container_width=True)
            else:
                kjs[1].warning("Satuan: " + str(p["status_js"]))

    st.subheader("Visual deteksi baris")
    st.image(gambar_rgb(det["visual"]), use_container_width=True)

    st.caption(
        "Deteksi berhasil berbeda dengan OCR akurat. "
        "Status 'crop perlu diperiksa' tetap harus diverifikasi."
    )


def _kumpulkan_field(det):
    # Kumpulan crop per field untuk pengenalan pada halaman Pengenalan.
    opsi = {}
    if det["tanggal"].get("kandidat") is not None:
        opsi["Tanggal"] = det["tanggal"]["kandidat"]
    for i, p in enumerate(det["produk"], 1):
        if p["bersih"] is not None and p["bersih"].get("crop") is not None:
            opsi[f"Produk {i}"] = p["bersih"]["crop"]
        if p["jumlah"] is not None:
            opsi[f"Jumlah {i}"] = p["jumlah"]
        if p["satuan"] is not None:
            opsi[f"Satuan {i}"] = p["satuan"]
    if det["nomor"].get("crop_digit") is not None:
        opsi["Nomor Nota"] = det["nomor"]["crop_digit"]
    return opsi


def halaman_pengenalan():
    st.header("Pengenalan")
    if st.session_state.get("gambar_bgr") is None:
        st.info("Pilih nota di halaman Upload terlebih dahulu.")
        return

    det = ambil_deteksi()
    if det is None:
        return

    st.subheader("Nomor Nota (CNN dan Template Matching)")
    digit = det["nomor"].get("crop_digit")
    if digit is None:
        st.warning(det["nomor"].get("pesan", "Digit nomor belum terisolasi."))
    else:
        st.image(gambar_rgb(digit), caption="Crop digit nomor",
                 width=220)
        jalan = st.button("Proses OCR nomor (tanpa & dengan CLAHE)",
                          key="tombol_ocr_nomor")
        if jalan:
            try:
                with st.spinner("Mengenali digit (CNN + Template Matching)..."):
                    tanpa = kenali_crop(digit, pakai_clahe=False)
                    dengan = kenali_crop(digit, pakai_clahe=True)
                    st.session_state["ocr_nomor"] = {
                        "tanpa": tanpa, "clahe": dengan,
                    }
            except Exception as error:
                st.error(f"OCR nomor gagal: {error}")

        hasil = st.session_state.get("ocr_nomor")
        if hasil:
            tanpa = hasil["tanpa"]
            dengan = hasil["clahe"]
            st.write("Jumlah karakter tersegmentasi:",
                     "tanpa CLAHE =", tanpa["n"],
                     "| dengan CLAHE =", dengan["n"])
            tabel = [
                {"Metode": "CNN",
                 "Tanpa CLAHE": tanpa["cnn"], "Dengan CLAHE": dengan["cnn"]},
                {"Metode": "Template Matching",
                 "Tanpa CLAHE": tanpa["tm"], "Dengan CLAHE": dengan["tm"]},
            ]
            st.dataframe(tabel, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Field lain (opsional)")
    opsi = _kumpulkan_field(det)
    if not opsi:
        st.info("Belum ada field lain yang siap dikenali.")
        return

    nama_field = st.selectbox("Pilih field", list(opsi.keys()), key="pilih_field")
    if st.button("OCR field terpilih", key="tombol_ocr_field"):
        try:
            with st.spinner("Mengenali field..."):
                hasil = kenali_crop(opsi[nama_field], pakai_clahe=False)
                st.session_state.setdefault("ocr_hasil", {})[nama_field] = hasil
        except Exception as error:
            st.error(f"OCR field gagal: {error}")

    hasil_field = st.session_state.get("ocr_hasil", {}).get(nama_field)
    if hasil_field:
        st.image(gambar_rgb(opsi[nama_field]),
                 caption=f"Crop {nama_field}", width=280)
        st.write("CNN:", hasil_field["cnn"] or "—")
        st.write("Template Matching:", hasil_field["tm"] or "—")
        st.write("Jumlah karakter:", hasil_field["n"])
        if hasil_field.get("biner") is not None:
            st.image(hasil_field["biner"], caption="Binarisasi segmentasi",
                     width=280)


GT_CONTOH = {
    "nota_01.jpeg": "0086",
    "nota_26.jpeg": "0099",
    "nota_52.jpeg": "0299",
    "nota_54.jpeg": "0292",
}


def halaman_evaluasi():
    st.header("Evaluasi (CER)")
    st.caption(
        "CER dihitung hanya untuk field yang punya ground truth dan "
        "crop OCR valid. Training accuracy bukan akurasi nota."
    )

    hasil = st.session_state.get("ocr_nomor")
    if not hasil:
        st.info("Jalankan OCR nomor nota di halaman Pengenalan terlebih dahulu.")
        return

    nama = st.session_state.get("nama_file", "")
    bawaan = GT_CONTOH.get(nama, "")
    gt = st.text_input("Ground truth nomor nota", value=bawaan, key="gt_nomor")

    if not gt.strip():
        st.warning("Isi ground truth nomor nota untuk menghitung CER.")
        return

    baris = []
    for metode in ("cnn", "tm"):
        for jalur, kunci in (("Tanpa CLAHE", "tanpa"), ("Dengan CLAHE", "clahe")):
            pred = hasil[kunci][metode]
            cer = hitung_cer(gt, pred)
            baris.append({
                "Metode": "CNN" if metode == "cnn" else "Template Matching",
                "Jalur": jalur,
                "Prediksi": pred or "—",
                "CER": None if cer is None else round(cer, 4),
            })

    st.dataframe(baris, use_container_width=True, hide_index=True)

    st.caption(
        "Bandingkan baris dengan lokasi field sama (crop digit identik) "
        "untuk melihat pengaruh CLAHE. Prediksi kosong tidak dihitung."
    )


# Routing enam halaman
st.sidebar.title("MERRI-STOCK")
st.sidebar.caption(
    "Analisis pengaruh image preprocessing (CLAHE) terhadap "
    "akurasi pengenalan teks nota."
)

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