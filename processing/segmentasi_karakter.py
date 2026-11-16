
from pathlib import Path
import sys
import csv

import cv2
import numpy as np

from processing.dataset_loader import normalisasi_karakter
from processing.pengenalan_karakter import (
    muat_cnn,
    muat_templates
)


SKALA = 3

# Ambang lebar komponen (relatif terhadap tinggi huruf) yang dianggap
# memuat lebih dari satu karakter sehingga perlu dicoba dipisah.
# Batas 1.25 terlalu longgar: huruf yang saling menempel seperti "va",
# "ln", atau "GI" lebarnya hanya ~1.1-1.2 x tinggi huruf, sehingga
# lolos dari pemisahan dan keluar sebagai satu kotak bertanda '?'.
# Nilai 1.10 dipilih dari pengujian: CER terbaik pada rentang 1.08-1.12.
# Pasangan yang lebih sempit dari ini (mis. "in" 1.03, "ei" 1.07) tetap
# ditangani oleh aturan "label mustahil" di _rapikan_gabung_pecah().
AMBANG_LEBAR_KOMPONEN = 1.10


# =========================================
# BACA DAN SIMPAN GAMBAR
# =========================================

def baca_gambar(path):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan_gambar(path, gambar):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, data = cv2.imencode(".png", gambar)

    if not ok:
        raise ValueError(f"Gagal encoding: {path}")

    with open(path, "wb") as f:
        f.write(data.tobytes())


# =========================================
# PREPROCESSING
# =========================================

def crop_tanpa_teks(gambar):
    """
    Deteksi crop yang tidak memuat tinta teks (kosong, pudar, atau hanya
    bayangan/gradasi).

    Pada crop seperti ini Otsu akan mengubah gradasi latar menjadi blob
    raksasa yang menyerupai karakter, sehingga muncul kotak-kotak palsu
    yang tampak seperti beberapa karakter tergabung. Ciri yang dipakai
    bersifat umum dan tidak bergantung pada satu nota tertentu:
      - tidak ada piksel gelap sama sekali (tinta teks selalu gelap), dan
      - variasi intensitas sangat kecil (gambar hampir seragam).

    Crop teks sungguhan pada nota proyek ini memiliki piksel gelap
    (min < 100) dan simpangan baku > 70.
    """
    if gambar is None or getattr(gambar, "size", 0) == 0:
        return True
    abu = gambar if gambar.ndim == 2 else cv2.cvtColor(
        gambar, cv2.COLOR_BGR2GRAY
    )
    return bool(float(abu.min()) > 140 and float(abu.std()) < 25.0)


def binarisasi(gambar):
    """
    Ubah crop abu-abu menjadi citra biner (tinta putih di latar hitam).

    Memakai Otsu global. Varian lain sudah diuji pada 3 nota acuan dan
    hasilnya lebih buruk untuk pengenalan akhir:

      varian                komponen tergabung   CER akhir
      Otsu (dipakai)              110            4.39%   <-- terbaik
      Adaptive threshold           41           10.05%
      Otsu dibatasi adaptive       72              -
      Otsu tanpa blur              93              -

    Adaptive threshold memang memisahkan lebih banyak huruf, tetapi
    merusak bentuk karakter (goresan tipis pecah, mis. "KonStein" menjadi
    "K0nS(e/nS") sehingga pengenalan jadi jauh lebih salah. Otsu dipilih
    karena mempertahankan bentuk karakter; sisa huruf menempel ditangani
    di tahap pemisahan (DP + _rapikan_gabung_pecah).
    """
    abu = cv2.cvtColor(
        gambar, cv2.COLOR_BGR2GRAY
    )

    abu = cv2.resize(
        abu,
        None,
        fx=SKALA,
        fy=SKALA,
        interpolation=cv2.INTER_CUBIC
    )

    abu = cv2.GaussianBlur(
        abu, (3, 3), 0
    )

    _, biner = cv2.threshold(
        abu, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return biner


# =========================================
# DETEKSI KOMPONEN
# =========================================

def binarisasi_batas(gambar):
    """
    CATATAN: pendekatan ini SUDAH DIUJI dan DITOLAK.

    Idenya: cari kotak karakter dari adaptive threshold (pemisahan lebih
    baik, 41 vs 110 karakter tergabung) tetapi ambil piksel dari Otsu
    (bentuk utuh). Kenyataannya CER memburuk dari 4.39% menjadi 9.80%,
    karena kotak hasil adaptive memecah karakter menjadi beberapa bagian
    (mis. titik dan batang 'i' terpisah) sehingga jumlah kotak melebihi
    jumlah karakter sebenarnya.

    Fungsi dipertahankan sebagai dokumentasi dan tidak dipakai.
    """
    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    abu = cv2.resize(abu, None, fx=SKALA, fy=SKALA,
                     interpolation=cv2.INTER_CUBIC)
    abu = cv2.GaussianBlur(abu, (3, 3), 0)

    adaptif = cv2.adaptiveThreshold(
        abu, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10,
    )

    # Sedikit pelebaran agar kotak tidak terlalu ketat dan tepi huruf
    # tidak terpotong saat piksel diambil dari biner Otsu.
    adaptif = cv2.dilate(adaptif, np.ones((3, 3), np.uint8), iterations=1)

    return adaptif


def deteksi_komponen(biner):
    n, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            biner,
            connectivity=8
        )
    )

    hasil = []

    for x, y, w, h, area in stats[1:]:
        if area >= 12 and h >= 4 and w >= 2:
            hasil.append((
                int(x), int(y),
                int(w), int(h)
            ))

    return sorted(
        hasil,
        key=lambda z: (z[0], z[1])
    )


# =========================================
# NORMALISASI KANDIDAT KARAKTER
# =========================================

def normalisasi_potongan(biner):
    titik = cv2.findNonZero(biner)

    if titik is None:
        return None

    x, y, w, h = cv2.boundingRect(titik)

    if w < 2 or h < 3:
        return None

    putih = cv2.bitwise_not(
        biner[y:y+h, x:x+w]
    )

    return normalisasi_karakter(putih)


# =========================================
# MENCARI BEBERAPA CALON BATAS
# =========================================

def calon_batas(blob, tinggi_acuan):
    h, w = blob.shape

    proyeksi = np.count_nonzero(
        blob, axis=0
    ).astype(np.float32)

    halus = np.convolve(
        proyeksi,
        np.ones(3) / 3,
        mode="same"
    )

    batas = {0, w}

    # 1) Kolom tanpa tinta = celah pasti antar karakter. Titik tengah
    #    setiap celah dijadikan batas agar huruf terpisah tidak menyatu.
    #    Celah selebar SATU kolom juga dihitung: pada huruf yang hampir
    #    menempel, pemisahnya sering hanya satu kolom kosong.
    kolom_kosong = proyeksi == 0
    i = 0
    while i < w:
        if kolom_kosong[i]:
            j = i
            while j < w and kolom_kosong[j]:
                j += 1
            if (j - i) >= 1 and 0 < i and j < w:
                batas.add((i + j) // 2)
            i = j
        else:
            i += 1

    # 2) Lembah proyeksi lokal untuk huruf yang benar-benar menempel.
    #
    #    PENTING: pencarian lembah memakai proyeksi MENTAH, bukan yang
    #    sudah dihaluskan. Penghalusan jendela-3 menggeser dan menutupi
    #    lembah sempit. Contoh nyata pada "ei" yang menempel:
    #        proyeksi mentah : ... 2, 1, 3 ...  -> minimum jelas di tengah
    #        setelah halus   : ... 1.67, 2.0, 5.67 ... -> minimum hilang
    #    Karena itu titik potong yang benar tidak pernah ditawarkan ke DP
    #    dan dua huruf tetap menyatu.
    ambang = np.percentile(proyeksi, 30)
    for p in range(1, w - 1):
        if (
            proyeksi[p] <= proyeksi[p - 1]
            and proyeksi[p] <= proyeksi[p + 1]
            and proyeksi[p] <= ambang
        ):
            batas.add(p)

    # 3) Grid tetap sebagai cadangan pilihan DP.
    langkah = max(3, round(tinggi_acuan * 0.12))
    for p in range(langkah, w, langkah):
        batas.add(p)

    return sorted(batas)


# =========================================
# PENILAIAN CNN DAN TEMPLATE MATCHING
# =========================================

def kandidat_potongan(blob, batas, tinggi_acuan):
    """
    Kumpulkan semua potongan kandidat (i, j) beserta citra 32x32-nya.

    Dipisahkan dari penilaian agar potongan dari SEMUA komponen dalam satu
    field dapat diprediksi dalam satu batch Ã¢â‚¬â€ jauh lebih cepat daripada
    memanggil model sekali per komponen.
    """
    min_w = max(2, int(0.10 * tinggi_acuan))

    # Batas lebar maksimum untuk SATU karakter.
    #
    # Sebelumnya 0.95 x tinggi huruf. Nilai itu terlalu sempit sehingga
    # huruf yang memang lebar seperti 'M', 'm', 'W' (rasio 1.0-1.3) selalu
    # dipaksa dipecah ('M' -> 'N'+'I', 'm' -> 'r'+'n'). Batas 1.60 memberi
    # ruang bagi huruf lebar tetap utuh; potongan janggal tetap kena
    # penalti lebar pada tahap penilaian.
    max_w = max(min_w + 1, int(1.60 * tinggi_acuan))

    pasangan = []
    array = []

    for i in range(len(batas) - 1):
        for j in range(i + 1, len(batas)):
            lebar = batas[j] - batas[i]
            if lebar > max_w:
                break
            if lebar < min_w:
                continue
            pot = normalisasi_potongan(blob[:, batas[i]:batas[j]])
            if pot is not None:
                pasangan.append((i, j))
                array.append(pot)

    return pasangan, array


def skor_dari_prediksi(batas, tinggi_acuan, pasangan, prediksi,
                       labels, label_tm, penalti=0.55):
    """
    Hitung skor pemilihan batas DP dari prediksi yang sudah dihitung.

    Skor ini bukan nilai akurasi OCR, melainkan nilai untuk memilih
    kombinasi potongan yang paling masuk akal.
    """
    skor = {}

    for (i, j), pr in zip(pasangan, prediksi):
        if pr is None:
            continue
        c_char, t_char, pc, pt = pr
        panjang = batas[j] - batas[i]

        nilai = (
            0.65 * np.log(max(pc, 1e-6))
            + 0.35 * np.log(max(pt, 1e-6))
            - 0.22 * abs(np.log(
                max(panjang, 1) / max(tinggi_acuan * 0.55, 1)
            ))
            - penalti
        )

        # Penalti tambahan bila potongan jauh lebih lebar dari satu
        # karakter. Koefisien 0.30 cukup menghukum gumpalan dua karakter
        # tanpa menghukum huruf lebar tunggal.
        if panjang > 0.85 * tinggi_acuan:
            nilai -= 0.30 * (panjang / max(tinggi_acuan, 1) - 0.85)

        # Bonus kecil bila CNN dan Template Matching sepakat.
        if c_char == t_char:
            nilai += 0.06

        skor[(i, j)] = (nilai, c_char, t_char, pc, pt)

    return skor


def skor_potongan(blob, batas, tinggi_acuan, model, labels, templates,
                  label_tm, penalti=0.55):
    """
    Versi mandiri untuk SATU blob (dipakai pengujian/kompatibilitas).

    Pipeline utama memakai kandidat_potongan + skor_dari_prediksi agar
    seluruh komponen dapat diprediksi sekaligus.
    """
    pasangan, array = kandidat_potongan(blob, batas, tinggi_acuan)
    if not array:
        return {}
    prediksi = _prediksi_banyak(
        array, model, labels, templates, label_tm, sudah_normal=True
    )
    return skor_dari_prediksi(
        batas, tinggi_acuan, pasangan, prediksi,
        labels, label_tm, penalti
    )


# =========================================
# MEMILIH KOMBINASI PEMOTONGAN
# =========================================

def pilih_batas(batas, skor):
    n = len(batas)

    dp = [-np.inf] * n
    prev = [None] * n

    dp[0] = 0.0

    # Dynamic programming
    for j in range(1, n):
        for i in range(j):

            if (
                (i, j) in skor
                and np.isfinite(dp[i])
            ):
                val = (
                    dp[i]
                    + skor[(i, j)][0]
                )

                if val > dp[j]:
                    dp[j] = val
                    prev[j] = i

    if prev[-1] is None:
        return []

    hasil = []
    j = n - 1

    while j > 0:
        i = prev[j]

        if i is None:
            return []

        hasil.append((
            batas[i],
            batas[j],
            skor[(i, j)]
        ))

        j = i

    return list(reversed(hasil))


# =========================================
# SEGMENTASI UTAMA
# =========================================

def _prediksi_potongan(blob_kolom, model, labels, templates, label_tm):
    # Prediksi satu potongan (CNN + Template Matching).
    pot = normalisasi_potongan(blob_kolom)
    if pot is None:
        return None
    X = pot.astype(np.float32).reshape(1, 32, 32, 1) / 255.0
    prob = model.predict(X, verbose=0)[0]
    c = int(np.argmax(prob))
    pc = float(prob[c])
    V = X.reshape(1, -1)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    sim = V @ templates.T
    t = int(np.argmax(sim[0]))
    pt = float(sim[0, t])
    return labels[c], label_tm[t], pc, pt


def _prediksi_banyak(daftar_potongan, model, labels, templates, label_tm,
                     sudah_normal=False):
    """
    Prediksi BANYAK potongan sekaligus dalam satu panggilan model.

    Memanggil model satu per satu (batch = 1) sangat lambat karena biaya
    tetap setiap panggilan jauh lebih besar daripada biaya menghitung satu
    gambar. Dengan menggabungkan semua potongan menjadi satu batch, waktu
    proses turun drastis dan hasilnya identik.

    sudah_normal=True dipakai bila potongan sudah berbentuk 32x32 hasil
    normalisasi_karakter, sehingga tidak perlu dinormalisasi ulang.

    Mengembalikan list dengan panjang sama seperti masukan; elemen bernilai
    None bila potongan tidak dapat dinormalisasi.
    """
    hasil = [None] * len(daftar_potongan)
    array = []
    indeks = []

    for i, pot in enumerate(daftar_potongan):
        if pot is None or getattr(pot, "size", 0) == 0:
            continue
        if sudah_normal:
            norm = pot
        else:
            norm = normalisasi_potongan(pot)
        if norm is None:
            continue
        array.append(norm)
        indeks.append(i)

    if not array:
        return hasil

    X = np.asarray(array, dtype=np.float32)[..., None] / 255.0
    prob = model.predict(X, batch_size=256, verbose=0)

    V = X[..., 0].reshape(len(X), -1)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    sim = V @ templates.T

    for row, idx in enumerate(indeks):
        c = int(np.argmax(prob[row]))
        t = int(np.argmax(sim[row]))
        hasil[idx] = (
            labels[c],
            label_tm[t],
            float(prob[row, c]),
            float(sim[row, t]),
        )

    return hasil


def _pecah_lebar(blob, a, b, char_w, model, labels, templates, label_tm,
                 kedalaman=0):
    """
    Pecah segmen [a..b] yang jauh lebih lebar dari satu karakter.

    Dipotong rekursif di lembah tinta terdekat titik tengah, sehingga
    huruf yang menempel tetap terpisah. Tiap potongan diprediksi ulang
    agar label tetap sesuai isinya.
    """
    lebar = b - a
    if lebar <= 1.45 * char_w or kedalaman >= 3:
        pr = _prediksi_potongan(blob[:, a:b], model, labels, templates, label_tm)
        if pr is None:
            return []
        c, t, pc, pt = pr
        return [(a, b, ("split", c, t, pc, pt))]

    proyeksi = np.count_nonzero(blob[:, a:b], axis=0).astype(np.float32)
    tengah = lebar // 2
    lo = max(1, tengah - int(0.4 * char_w))
    hi = min(lebar - 1, tengah + int(0.4 * char_w))
    pos = tengah if hi <= lo else lo + int(np.argmin(proyeksi[lo:hi]))

    kiri = _pecah_lebar(blob, a, a + pos, char_w,
                        model, labels, templates, label_tm, kedalaman + 1)
    kanan = _pecah_lebar(blob, a + pos, b, char_w,
                         model, labels, templates, label_tm, kedalaman + 1)
    return kiri + kanan


def _rapikan_gabung_pecah(hasil, biner, tinggi_acuan,
                          model, labels, templates, label_tm):
    """
    Perbaikan akhir atas hasil dynamic programming.

    DP kadang terlalu kasar pada gumpalan beberapa karakter:
      (A) satu karakter berongga (mis. 'O', 'Q') terpotong menjadi dua
          busur sempit ('(' + ')'), atau
      (B) dua karakter menempel (mis. 'GI') dibiarkan menjadi satu kotak
          lebar yang dikenali salah.

    Kedua kasus diperbaiki BERBASIS KEYAKINAN CNN, bukan sekadar lebar:
      (A) dua segmen sempit yang bersebelahan tanpa celah digabung kembali
          HANYA bila potongan gabungannya dikenali lebih yakin daripada
          kedua bagiannya;
      (B) satu segmen lebar dengan keyakinan rendah dipecah di lembah
          proyeksi HANYA bila kedua belahannya dikenali lebih yakin.
    Dengan gerbang keyakinan ini karakter sah (mis. 'M' utuh, atau 'i'+'l'
    yang memang dua karakter) tidak ikut berubah.
    """
    if not hasil or biner is None:
        return hasil

    H, W = biner.shape[:2]
    ta = max(tinggi_acuan, 1.0)
    hasil = sorted(hasil, key=lambda z: (z["x1"], z["y1"]))

    def ambil(x1, y1, x2, y2):
        if x2 <= x1 or y2 <= y1:
            return None
        return biner[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]

    # ---------- (A) GABUNG ----------
    busur = {"(", ")", "[", "]", "C", "c", "K", "G", "0", "O", "o", "Q"}
    huruf_bulat = {"O", "o", "0", "Q", "D", "C", "G", "c", "e", "a", "U"}

    # Kumpulkan semua pasangan bersebelahan yang memenuhi syarat bentuk,
    # lalu prediksi potongan gabungannya SEKALIGUS (satu batch). Tanpa
    # ini, setiap pasangan memanggil model sendiri-sendiri dan proses
    # menjadi sangat lambat.
    kandidat_gabung = {}
    for i in range(len(hasil) - 1):
        cur, nxt = hasil[i], hasil[i + 1]
        w_cur = cur["x2"] - cur["x1"]
        w_nxt = nxt["x2"] - nxt["x1"]
        celah = nxt["x1"] - cur["x2"]
        lebar_gabung = w_cur + w_nxt + celah
        if not (
            celah <= 1
            and w_cur < 0.60 * ta
            and w_nxt < 0.60 * ta
            and lebar_gabung <= 1.15 * ta
            and cur.get("cnn") in busur
            and nxt.get("cnn") in busur
        ):
            continue
        y1 = min(cur["y1"], nxt["y1"])
        y2 = max(cur["y2"], nxt["y2"])
        pot = ambil(cur["x1"], y1, nxt["x2"], y2)
        if pot is not None:
            kandidat_gabung[i] = (y1, y2, pot)

    indeks_gabung = sorted(kandidat_gabung)
    prediksi_gabung = _prediksi_banyak(
        [kandidat_gabung[i][2] for i in indeks_gabung],
        model, labels, templates, label_tm,
    )
    hasil_gabung = dict(zip(indeks_gabung, prediksi_gabung))

    gabung = []
    i = 0
    while i < len(hasil):
        cur = dict(hasil[i])
        merged = hasil_gabung.get(i)
        if merged is not None:
            c, t, pc, pt = merged
            if pc >= 0.55 and c in huruf_bulat:
                y1, y2, _ = kandidat_gabung[i]
                cur = {
                    "x1": cur["x1"], "y1": y1,
                    "x2": hasil[i + 1]["x2"], "y2": y2,
                    "cnn": c, "tm": t, "pc": pc, "pt": pt,
                    "status": "KANDIDAT",
                }
                gabung.append(cur)
                i += 2
                continue
        gabung.append(cur)
        i += 1
    hasil = gabung

    sempit_mustahil = {
        ".", ",", "'", '"', "`", ":", ";", "!", "|",
        "i", "l", "1", "I", "j", "-", "~", "_",
    }

    # ---------- (B) PECAH ----------
    # Tahap 1: tentukan segmen yang perlu diuji beserta posisi potongnya.
    rencana = []
    for idx, s in enumerate(hasil):
        w = s["x2"] - s["x1"]
        pc = float(s.get("pc", 0.0) or 0.0)
        ragu = pc < 0.9 or s.get("tm") != s.get("cnn")
        lebar_ragu = w > 1.05 * ta and ragu
        label_mustahil = (
            s.get("cnn") in sempit_mustahil and w > 0.75 * ta
        )
        # Pemicu ketiga: keyakinan SANGAT rendah.
        #
        # Gabungan huruf sempit tidak bisa dideteksi dari lebar. Contoh
        # nyata: "In" pada "Indra" menyatu menjadi satu kotak selebar
        # 0.47 x tinggi huruf ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â LEBIH SEMPIT dari median karakter,
        # sehingga tidak ada ambang lebar yang bisa menangkapnya.
        # Satu-satunya sinyal yang tersisa adalah keraguan model.
        ragu_kuat = pc < 0.60 and w > 0.40 * ta
        if not (lebar_ragu or label_mustahil or ragu_kuat):
            continue

        pot = ambil(s["x1"], s["y1"], s["x2"], s["y2"])
        if pot is None:
            continue
        proyeksi = np.count_nonzero(pot, axis=0).astype(np.float32)

        tengah = w // 2
        # Jendela 35% dari tengah: cukup untuk pasangan huruf biasa, tanpa
        # membuat potongan asal-asalan pada karakter lebar. Jendela 45%
        # sudah dicoba dan menurunkan akurasi (CER 4.39% -> 5.94%).
        lo = max(1, tengah - int(0.35 * w))
        hi = min(w - 1, tengah + int(0.35 * w))
        if hi <= lo:
            continue

        pos_min = lo + int(np.argmin(proyeksi[lo:hi]))
        posisi = [pos_min]
        jumlah_coba = 5
        for k in range(1, jumlah_coba + 1):
            p = lo + round((hi - lo) * k / (jumlah_coba + 1))
            if p not in posisi:
                posisi.append(p)

        rencana.append({
            "idx": idx, "s": s, "posisi": posisi,
            "pos_min": pos_min, "pc": pc,
            "label_mustahil": label_mustahil, "ragu_kuat": ragu_kuat,
        })

    # Tahap 2: siapkan semua potongan (kiri & kanan) lalu prediksi
    # SEKALIGUS. Ini bagian terbesar penghemat waktu: sebelumnya setiap
    # posisi memanggil model dua kali secara terpisah.
    potongan = []
    for r in rencana:
        s = r["s"]
        for pos in r["posisi"]:
            potongan.append(ambil(s["x1"], s["y1"], s["x1"] + pos, s["y2"]))
            potongan.append(ambil(s["x1"] + pos, s["y1"], s["x2"], s["y2"]))

    prediksi_split = _prediksi_banyak(
        potongan, model, labels, templates, label_tm
    )

    # Tahap 3: susun tabel skor per posisi untuk tiap segmen.
    kursor = 0
    tabel_skor = {}
    for r in rencana:
        tabel = {}
        for pos in r["posisi"]:
            kiri = prediksi_split[kursor]
            kursor += 1
            kanan = prediksi_split[kursor]
            kursor += 1
            if kiri is None or kanan is None:
                continue
            tabel[pos] = (min(kiri[2], kanan[2]), pos, kiri, kanan)
        tabel_skor[r["idx"]] = tabel

    # Tahap 4: putuskan per segmen memakai hasil batch.
    rencana_per_idx = {r["idx"]: r for r in rencana}
    pecah = []
    for idx, s in enumerate(hasil):
        r = rencana_per_idx.get(idx)
        if r is None:
            pecah.append(s)
            continue

        tabel = tabel_skor.get(idx, {})
        terbaik = tabel.get(r["pos_min"])
        # Bila lembah tinta TIDAK menghasilkan dua bagian yang meyakinkan,
        # cari titik lain di sekitar lembah (contoh: "ei" yang menempel).
        if terbaik is None or terbaik[0] < 0.70:
            for calon in tabel.values():
                if terbaik is None or calon[0] > terbaik[0]:
                    terbaik = calon

        if terbaik is None:
            pecah.append(s)
            continue

        skor, pos, kiri, kanan = terbaik
        yakin_lebih_baik = skor > r["pc"]
        # Untuk label yang mustahil, kedua bagian harus wajar (>= 0.80).
        wajar = r["label_mustahil"] and skor >= 0.80
        # Pemicu keraguan menuntut bukti paling kuat.
        if r["ragu_kuat"]:
            terima = skor >= 0.85
        else:
            terima = yakin_lebih_baik or wajar

        if not terima:
            pecah.append(s)
            continue

        for seg, pr in ((0, kiri), (pos, kanan)):
            c, t, pc, pt = pr
            x1 = s["x1"] + seg
            x2 = s["x1"] + pos if seg == 0 else s["x2"]
            pecah.append({
                "x1": x1, "y1": s["y1"],
                "x2": x2, "y2": s["y2"],
                "cnn": c, "tm": t, "pc": pc, "pt": pt,
                "status": "KANDIDAT",
            })

    hasil = sorted(pecah, key=lambda z: (z["x1"], z["y1"]))
    return hasil


def segmentasi_karakter_tm(gambar, model, labels, templates, label_tm):
    """
    Alur utama sesuai metode penelitian:
    Template Matching menentukan posisi/pola karakter, lalu CNN
    mengklasifikasikan setiap kandidat yang ditemukan.

    Koordinat keluaran memakai ruang upsampled (SKALA) agar sama dengan
    segmentasi_karakter sehingga pemanggil tidak perlu diubah.
    """
    from processing.lokalisasi_template import cari_kandidat_karakter

    biner = binarisasi(gambar)
    kandidat = cari_kandidat_karakter(gambar)
    if not kandidat:
        return biner, []

    hasil = []
    for k in kandidat:
        x1 = int(k["x1"] * SKALA)
        y1 = int(k["y1"] * SKALA)
        x2 = int(k["x2"] * SKALA)
        y2 = int(k["y2"] * SKALA)

        y1 = max(0, min(biner.shape[0] - 2, y1))
        y2 = max(y1 + 2, min(biner.shape[0], y2))
        x1 = max(0, min(biner.shape[1] - 2, x1))
        x2 = max(x1 + 2, min(biner.shape[1], x2))

        potongan = biner[y1:y2, x1:x2]
        pr = _prediksi_potongan(potongan, model, labels, templates, label_tm)
        if pr is None:
            continue
        c, t, pc, pt = pr

        hasil.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cnn": c,
            "tm": k.get("template", t),
            "pc": pc,
            "pt": float(k.get("skor", pt)),
            "status": "KANDIDAT",
        })

    hasil.sort(key=lambda z: (z["x1"], z["y1"]))
    return biner, hasil


def segmentasi_utama(gambar, model=None, labels=None, templates=None,
                     label_tm=None, utamakan_tm=False):
    """
    Menjalankan pipeline pengenalan karakter.

    - utamakan_tm=False (default): segmentasi proyeksi + DP, lalu CNN.
      Ini jalur yang terbukti stabil pada data nota proyek ini.
    - utamakan_tm=True: Template Matching menentukan posisi/pola lebih
      dahulu, lalu CNN mengklasifikasi. Disediakan untuk eksperimen
      metode; pada pengujian, hasilnya masih di bawah jalur proyeksi.

    Mengembalikan (biner, hasil, metode).
    """
    if model is None:
        model, labels = muat_cnn()
    if templates is None:
        templates, label_tm = muat_templates()

    if utamakan_tm:
        biner, hasil = segmentasi_karakter_tm(
            gambar, model, labels, templates, label_tm
        )
        if len(hasil) >= 3:
            return biner, hasil, "template_matching"

    biner, hasil = segmentasi_karakter(
        gambar, model, labels, templates, label_tm
    )
    return biner, hasil, "proyeksi"


def segmentasi_karakter(
    gambar,
    model=None,
    labels=None,
    templates=None,
    label_tm=None
):
    if model is None:
        model, labels = muat_cnn()

    if templates is None:
        templates, label_tm = muat_templates()

    biner = binarisasi(gambar)

    # Crop kosong/pudar tidak boleh diproses: Otsu akan mengubah gradasi
    # latar menjadi blob raksasa yang tampak seperti karakter tergabung.
    if crop_tanpa_teks(gambar):
        return biner, []

    komponen = deteksi_komponen(biner)

    if not komponen:
        return biner, []

    tinggi = [
        h for _, _, _, h in komponen
    ]

    h_max = max(tinggi)

    tinggi_acuan = float(
        np.median([
            h for h in tinggi
            if h >= h_max * 0.45
        ])
    )

    # Buang noise yang berada di luar pita vertikal baris teks.
    pusat = [
        y + h / 2 for _, y, _, h in komponen
        if h >= 0.5 * tinggi_acuan
    ]
    if pusat:
        cy_med = float(np.median(pusat))
        komponen = [
            k for k in komponen
            if (cy_med - 0.75 * tinggi_acuan)
            <= (k[1] + k[3] / 2)
            <= (cy_med + 0.75 * tinggi_acuan)
        ]

    # Buang minus pembuka daftar: hanya komponen paling kiri yang pipih
    # dan lebar. Minus di dalam kode produk (mis. K-300) tidak tersentuh
    # karena bukan komponen paling kiri.
    if komponen:
        kiri = min(komponen, key=lambda k: k[0])
        _, _, w_kiri, h_kiri = kiri
        if (
            h_kiri <= 0.45 * tinggi_acuan
            and 1.3 * h_kiri <= w_kiri <= 1.4 * tinggi_acuan
        ):
            komponen = [k for k in komponen if k != kiri]

    if not komponen:
        return biner, []

    # Estimasi lebar satu karakter untuk memecah segmen yang terlalu lebar.
    #
    # Hanya komponen yang JELAS satu karakter (lebar <= 0.90 x tinggi) yang
    # dipakai menghitung median. Komponen yang sudah tergabung tidak boleh
    # ikut, karena akan menggeser estimasi ke atas dan membuat pasangan
    # huruf sempit tidak pernah dikenali sebagai gabungan.
    lebar_satu = [
        w0 for (_, _, w0, h0) in komponen
        if h0 >= 0.55 * tinggi_acuan
        and 0.15 * tinggi_acuan <= w0 <= 0.90 * tinggi_acuan
    ]
    if len(lebar_satu) >= 3:
        char_w_est = float(np.median(lebar_satu))
    else:
        char_w_est = 0.55 * tinggi_acuan

    # ---------------------------------------------------------------
    # Tahap A: tentukan batas tiap komponen dan kumpulkan potongan
    # ---------------------------------------------------------------
    # Seluruh prediksi dijalankan dalam beberapa batch besar, bukan satu
    # per komponen. Inilah sumber utama percepatan: sebelumnya model
    # dipanggil sekali untuk setiap komponen (belasan kali per field).
    rencana = []
    patch_guard = []

    for x, y, w, h in komponen:
        blob = biner[y:y + h, x:x + w]

        # Komponen dicoba dipisah bila lebih lebar dari SATU karakter.
        # Dua ukuran dipakai sekaligus:
        #   - relatif tinggi huruf (1.10): menangkap gabungan yang jelas;
        #   - relatif lebar median karakter (1.35): menangkap gabungan
        #     huruf SEMPIT (mis. "rt", "is") yang secara absolut tidak lebar.
        is_lebar = (
            (w > AMBANG_LEBAR_KOMPONEN * tinggi_acuan
             or w > 1.35 * char_w_est)
            and h > 0.40 * tinggi_acuan
        )

        # Jaring pengaman: pada zona aturan baru, komponen yang sudah
        # dikenali yakin sebagai satu karakter tidak dipecah.
        idx_guard = None
        if is_lebar and w <= AMBANG_LEBAR_KOMPONEN * tinggi_acuan:
            pot_utuh = normalisasi_potongan(blob)
            if pot_utuh is not None:
                idx_guard = len(patch_guard)
                patch_guard.append(pot_utuh)

        rencana.append({
            "x": x, "y": y, "w": w, "h": h, "blob": blob,
            "is_lebar": is_lebar, "idx_guard": idx_guard,
        })

    # Satu batch untuk seluruh pemeriksaan jaring pengaman.
    pred_guard = _prediksi_banyak(
        patch_guard, model, labels, templates, label_tm, sudah_normal=True
    )

    # ---------------------------------------------------------------
    # Tahap B: finalkan batas + kumpulkan kandidat potongan
    # ---------------------------------------------------------------
    semua_potongan = []

    for r in rencana:
        blob, w = r["blob"], r["w"]
        is_lebar = r["is_lebar"]

        if r["idx_guard"] is not None:
            cek = pred_guard[r["idx_guard"]]
            if cek is not None and cek[2] >= 0.90:
                is_lebar = False

        r["is_lebar"] = is_lebar
        batas = calon_batas(blob, tinggi_acuan) if is_lebar else [0, w]
        r["batas"] = batas

        pasangan, array = kandidat_potongan(blob, batas, tinggi_acuan)
        r["pasangan"] = pasangan
        r["awal"] = len(semua_potongan)
        semua_potongan.extend(array)

        # Penalti per-karakter diturunkan untuk blob lebar agar DP rela
        # memecah gumpalan menjadi beberapa karakter.
        rasio_lebar = w / max(tinggi_acuan, 1.0)
        if rasio_lebar > 1.5:
            r["penalti"] = 0.15
        elif rasio_lebar > 1.2:
            r["penalti"] = 0.35
        else:
            r["penalti"] = 0.55

    # Satu batch untuk seluruh kandidat potongan di field ini.
    pred_pasangan = _prediksi_banyak(
        semua_potongan, model, labels, templates, label_tm, sudah_normal=True
    )

    # ---------------------------------------------------------------
    # Tahap C: skor + dynamic programming per komponen
    # ---------------------------------------------------------------
    hasil = []

    for r in rencana:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        blob, batas = r["blob"], r["batas"]
        awal = r["awal"]
        pred_slice = pred_pasangan[awal:awal + len(r["pasangan"])]

        nilai = skor_dari_prediksi(
            batas, tinggi_acuan, r["pasangan"], pred_slice,
            labels, label_tm, penalti=r["penalti"]
        )

        potong = pilih_batas(batas, nilai)

        if not potong:
            # DP tidak menemukan satu pun kandidat valid. Komponen utuh
            # tetap diklasifikasikan agar selalu ada prediksi.
            utuh = _prediksi_potongan(
                blob, model, labels, templates, label_tm
            )

            if utuh is None:
                hasil.append({
                    "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                    "cnn": "?", "tm": "?", "pc": 0.0, "pt": 0.0,
                    "status": "BELUM_TERURAI",
                })
                continue

            c_utuh, t_utuh, pc_utuh, pt_utuh = utuh
            hasil.append({
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "cnn": c_utuh, "tm": t_utuh,
                "pc": pc_utuh, "pt": pt_utuh,
                "status": "KANDIDAT",
            })
            continue

        for a, b, info in potong:
            # Pecah lagi hanya bila segmen memang cukup lebar untuk memuat
            # lebih dari satu karakter. Syarat lebar menjaga agar huruf
            # tunggal yang memang lebar (mis. 'M', 'm', 'W') tidak ikut
            # terpecah. Ambang 45% jendela sudah diuji dan menurunkan
            # akurasi, jadi 35% yang dipakai.
            lebar_seg = b - a
            cukup_lebar = (
                lebar_seg > 1.45 * char_w_est
                and lebar_seg > 1.40 * tinggi_acuan
            )

            if cukup_lebar:
                bagian = _pecah_lebar(
                    blob, a, b, char_w_est,
                    model, labels, templates, label_tm
                )
            else:
                bagian = [(a, b, info)]

            for aa, bb, info2 in bagian:
                _, c, t, pc, pt = info2
                hasil.append({
                    "x1": x + aa, "y1": y,
                    "x2": x + bb, "y2": y + h,
                    "cnn": c, "tm": t, "pc": pc, "pt": pt,
                    "status": "KANDIDAT",
                })

    hasil = sorted(
        hasil,
        key=lambda z: (
            z["x1"],
            z["y1"]
        )
    )

    # Perbaikan akhir: gabung kembali karakter yang terpotong dua dan
    # pecah gumpalan dua karakter yang lolos, berbasis keyakinan CNN.
    try:
        hasil = _rapikan_gabung_pecah(
            hasil, biner, tinggi_acuan,
            model, labels, templates, label_tm
        )
    except Exception:
        pass

    # Buang sisa tanda baca label (mis. ':') di awal/akhir field.
    try:
        tinggi_acuan_final = float(
            np.median([max(1, s["y2"] - s["y1"]) for s in hasil])
        ) if hasil else float(tinggi_acuan)
        hasil = rapikan_segmen(hasil, tinggi_acuan_final)
    except Exception:
        pass

    return biner, hasil


# =========================================
# MENYUSUN HASIL MENJADI TEKS
# =========================================

def rapikan_segmen(hasil, tinggi_acuan):
    """
    Merapikan hasil segmentasi sebelum disusun menjadi teks.

    Membuang segmen pengganggu di awal/akhir yang bukan bagian isi field,
    yaitu tanda baca berdiri sendiri seperti ':' atau '=' yang tersisa
    dari label nota (mis. "Proyek :").

    Tidak mengubah urutan maupun karakter sah di tengah.
    """
    if not hasil:
        return hasil

    hasil = sorted(hasil, key=lambda z: z["x1"])

    buang = {":", "=", "~", "_", "|"}

    # Tanda yang memang kecil secara wajar, jadi tidak boleh dibuang
    # walaupun ukurannya jauh di bawah tinggi huruf.
    kecil_sah = {".", ",", "'", "`", "-", '"'}

    def pengganggu(item):
        label = item.get("cnn", "")
        if label in buang:
            return True

        # Serpihan sangat pendek di tepi (mis. sisa garis tabel atau
        # potongan huruf tetangga) bukan bagian isi field.
        tinggi_item = item.get("y2", 0) - item.get("y1", 0)
        return (
            label not in kecil_sah
            and tinggi_item < 0.45 * tinggi_acuan
        )

    while len(hasil) > 2 and pengganggu(hasil[0]):
        hasil = hasil[1:]
    while len(hasil) > 2 and pengganggu(hasil[-1]):
        hasil = hasil[:-1]

    return hasil


def gabung_segmen_menempel(hasil, biner, tinggi_acuan,
                           model, labels, templates, label_tm):
    """
    Menggabungkan dua segmen bersebelahan yang tintanya saling menempel
    (celah sangat kecil), lalu memprediksi ulang potongan gabungan.

    Hanya dilakukan bila lebar gabungan masih wajar untuk satu karakter,
    sehingga karakter yang sah tidak ikut tergabung.
    """
    if not hasil or biner is None:
        return hasil

    ambang_celah = max(1, int(0.06 * tinggi_acuan))
    batas_lebar = 1.15 * tinggi_acuan

    hasil = sorted(hasil, key=lambda z: z["x1"])
    keluaran = []
    i = 0

    while i < len(hasil):
        if i == len(hasil) - 1:
            keluaran.append(hasil[i])
            break

        kiri = hasil[i]
        kanan = hasil[i + 1]
        celah = kanan["x1"] - kiri["x2"]
        lebar = kanan["x2"] - kiri["x1"]

        if celah <= ambang_celah and lebar <= batas_lebar:
            y1 = min(kiri["y1"], kanan["y1"])
            y2 = max(kiri["y2"], kanan["y2"])
            x1, x2 = kiri["x1"], kanan["x2"]

            potong = biner[
                max(0, y1):min(biner.shape[0], y2),
                max(0, x1):min(biner.shape[1], x2),
            ]

            pred = _prediksi_potongan(
                potong, model, labels, templates, label_tm
            )
            if pred is not None:
                c, t, pc, pt = pred
                keluaran.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cnn": c, "tm": t, "pc": pc, "pt": pt,
                    "status": "KANDIDAT",
                })
                i += 2
                continue

        keluaran.append(kiri)
        i += 1

    return keluaran


def susun_teks(
    hasil,
    metode,
    tinggi_acuan
):
    teks = ""
    akhir = None

    # Hitung ulang tinggi acuan dari data segmen agar konsisten, dan ukur
    # celah antarkarakter untuk mendeteksi spasi antarkata.
    if hasil:
        tinggi_seg = np.median([max(1, s["y2"] - s["y1"]) for s in hasil])
        tinggi_acuan = float(tinggi_seg)
    if tinggi_acuan <= 0:
        tinggi_acuan = 12.0

    celah = []
    for i in range(1, len(hasil)):
        celah.append(hasil[i]["x1"] - hasil[i - 1]["x2"])

    # Ambang spasi antarkata.
    #
    # Dua ukuran dipakai, lalu diambil yang LEBIH KECIL agar spasi yang
    # wajar tidak hilang (keluhan nyata: "RS Abdul" menjadi "RSAbdul"):
    #   1. berbasis celah antarkarakter (aturan lama);
    #   2. berbasis lebar karakter pada baris ini.
    # Mengambil nilai terkecil berarti spasi lebih mudah dikenali, tanpa
    # pernah lebih sulit daripada aturan lama.
    lebar_kar = [
        max(1, s["x2"] - s["x1"])
        for s in hasil
        if s.get("cnn") not in {".", ",", "'", "`", "-", ":", ";"}
    ]
    kar_med = float(np.median(lebar_kar)) if lebar_kar else 0.55 * tinggi_acuan

    ambang_celah = (
        max(0.22 * tinggi_acuan,
            float(np.median(celah)) + 0.25 * tinggi_acuan)
        if celah else 0.30 * tinggi_acuan
    )
    ambang_lebar = max(0.28 * tinggi_acuan, 0.45 * kar_med)

    ambang_spasi = min(ambang_celah, ambang_lebar)

    for i, item in enumerate(hasil):
        if i > 0:
            gap = item["x1"] - hasil[i - 1]["x2"]
            if gap > ambang_spasi:
                teks += " "

        teks += item[metode]
        akhir = max(item["x2"], akhir or 0)

    return teks


# =========================================
# PENGUJIAN DAN PENYIMPANAN
# =========================================

def proses(path_gambar):
    path = Path(path_gambar)

    gambar = baca_gambar(path)

    if gambar is None:
        raise ValueError(
            f"Gambar tidak dapat dibaca: {path}"
        )

    output = path.parent / (
        "segmentasi_" + path.stem
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Memuat CNN dan Template Matching..."
    )

    model, labels = muat_cnn()
    templates, label_tm = muat_templates()

    biner, hasil = segmentasi_karakter(
        gambar,
        model,
        labels,
        templates,
        label_tm
    )

    visual = cv2.resize(
        gambar,
        (
            biner.shape[1],
            biner.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )

    tinggi = [
        s["y2"] - s["y1"]
        for s in hasil
    ]

    tinggi_acuan = (
        float(np.median(tinggi))
        if tinggi else 12.0
    )

    daftar = []

    for no, item in enumerate(hasil, 1):

        x1, y1, x2, y2 = (
            item[k] for k in (
                "x1", "y1", "x2", "y2"
            )
        )

        if item["status"] == "KANDIDAT":
            warna = (0, 160, 0)
        else:
            warna = (0, 0, 255)

        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            warna,
            1
        )

        crop = cv2.bitwise_not(
            biner[y1:y2, x1:x2]
        )

        simpan_gambar(
            output / "potongan"
            / f"segmen_{no:03d}.png",
            crop
        )

        daftar.append({
            "nomor": no,
            **item
        })

    simpan_gambar(
        output / "biner.png",
        biner
    )

    simpan_gambar(
        output / "hasil_segmentasi.png",
        visual
    )

    with open(
        output / "hasil_prediksi.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        kolom = [
            "nomor",
            "x1", "y1",
            "x2", "y2",
            "cnn", "tm",
            "pc", "pt",
            "status"
        ]

        penulis = csv.DictWriter(
            f,
            fieldnames=kolom
        )

        penulis.writeheader()
        penulis.writerows(daftar)

    teks_cnn = susun_teks(
        hasil,
        "cnn",
        tinggi_acuan
    )

    teks_tm = susun_teks(
        hasil,
        "tm",
        tinggi_acuan
    )

    with open(
        output / "hasil_teks.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"CNN: {teks_cnn}\n"
            f"Template Matching: {teks_tm}\n"
        )

    print("\n=== HASIL OCR EKSPERIMEN ===")
    print("Jumlah kandidat:", len(hasil))
    print("CNN:", teks_cnn)
    print("Template Matching:", teks_tm)
    print("Output:", output)

    print(
        "\nPERINGATAN: Hasil masih perlu "
        "dibandingkan dengan ground truth."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            'Gunakan: python -m '
            'processing.segmentasi_karakter '
            '"path_crop.png"'
        )

    proses(sys.argv[1])
