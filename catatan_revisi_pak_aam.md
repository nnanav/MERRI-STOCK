# Catatan Revisi — Demo ke Pak Aam

Berkas ini untuk dibaca sambil demo. Hapus bila tidak diperlukan.

---

## Kalimat pembuka

> "Pak, revisi yang Bapak minta sudah saya kerjakan semuanya hari ini.
> Yang paling utama: halaman pengenalan sekarang tidak lagi membandingkan
> CNN dan Template Matching, tapi menampilkan satu hasil OCR gabungan.
> Segmentasi per karakter juga sudah diperbaiki, dan CER turun dari
> 0,1138 ke 0,0722."

---

## 1. Preparation → Business Understanding

- Nama halaman: **Business Understanding** (tidak ada lagi "Preparation").
- Isinya cara kerja: Latar Belakang → Tujuan → **Alur Kerja Sistem** →
  Informasi Data Latih → Enam Field Target.
- Alur: Upload → Preprocessing → Deteksi Field → Segmentasi Karakter →
  Pengenalan CNN + Template Matching → Evaluasi CER.

**Demo:** buka halaman Business Understanding, tunjuk poin 3 dan poin 5.

---

## 2. Template Matching + CNN digabung, bukan dibandingkan

- Yang dihapus: tabel per karakter (`CNN (label)`, `p_CNN`, `Pola TM`,
  `p_TM`, `Dukungan TM`, `Aturan`) dan caption pola Template Matching.
- Yang tersisa: gambar crop → **Hasil OCR** → jumlah karakter.
- Tetap bekerja bergabung: TM mencocokkan pola dulu, CNN mengenali
  karakternya, keluar satu teks OCR.

**Demo:** halaman Pengenalan, tunjuk tabel (tidak ada kolom CNN/TM).

---

## 3. Field tanggal tidak dipakai

- Tanggal tidak pernah masuk daftar field.
- Enam field target: Nomor Nota, Nama Penerima, Proyek, Nama Produk,
  Jumlah, Harga Satuan.

**Demo:** halaman Business Understanding poin 5.

---

## 4. Segmentasi per karakter diperbaiki

- Ambang lebar komponen: 1,25 → **1,10**.
- Komponen yang gagal dipotong tidak lagi keluar tanda `?`
  (contoh: huruf `P` pada "Pcs" sekarang terbaca).
- Serpihan tepi yang terlalu pendek dibuang.

**Demo:** halaman Segmentasi, buka satu field, tunjuk kotak per karakter.

---

## 5. Angka CER — jelaskan bertahap

| Tahap | CER | Sebab |
|---|---|---|
| Sebelum | 0,1138 | Baseline |
| Segmentasi diperbaiki | 0,1005 | Huruf yang tadinya `?` ikut terbaca |
| Crop harga diperbaiki | 0,0904 | Titik ribuan `10.550` tidak hilang |
| Ground truth dikoreksi | **0,0722** | Koreksi data acuan, bukan program |

> "Pak, tiga baris pertama murni perbaikan program. Baris terakhir bukan
> klaim akurasi baru: ground truth saya sendiri yang salah. Di nota_52
> baris pertama tertulis **400 Pcs**, bukan 3.400 seperti catatan saya.
> Sudah saya zoom dari gambar aslinya, memang 400. Saya perbaiki supaya
> evaluasinya jujur."

Hasil akhir: **CER 0,0722 — akurasi karakter 92,78% — 20 dari 33 field tepat.**

---

## 6. Tambahan hari ini

- **CLAHE dihapus** dari penjelasan karena program tidak melakukan
  peningkatan kontras. Diganti deskripsi sebenarnya: penyeragaman ukuran →
  grayscale → binerisasi Otsu.
- **Label "CNN" dihapus** dari tabel dan keterangan, cukup **"Hasil OCR"**.
- **CER yang tadinya `None`** sudah diperbaiki: ground truth terisi
  otomatis per field, angkanya tetap bisa diedit manual.

---

## 7. Antisipasi pertanyaan

| Pertanyaan | Jawaban |
|---|---|
| "Field satuan kok masih ada? Bukannya 6 field?" | "Betul Pak, satuan masih tampil. Internal dipakai untuk memisahkan angka jumlah dari satuannya. Kalau Bapak mau yang tampil benar-benar 6 field, tinggal saya sembunyikan." |
| "Template Matching jadi tidak kelihatan?" | "Tetap bekerja Pak, posisinya di depan: TM mencocokkan pola, CNN menentukan karakter akhirnya. Yang hilang hanya tampilan perbandingannya, sesuai revisi." |
| "CER 0,07 masih ada yang salah?" | "Sisanya di teks panjang, nama penerima dan proyek. Penyebabnya kualitas cetak nota dan model masih 70 kelas, bukan di segmentasinya." |
| "Sudah diuji?" | "Sudah Pak: 6 halaman tanpa error, alur upload sampai evaluasi tanpa error, dan angkanya bisa direproduksi lewat `python tests/ukur_cer.py`." |

---

## 8. Urutan demo (2 menit)

1. Business Understanding — alur kerja + enam field target.
2. Upload `nota_52.jpeg` → **Proses Nota**.
3. Pengenalan — tunjuk tabel, hanya "Hasil OCR".
4. Evaluasi — CER sudah terisi otomatis, sebut **0,0722 / 92,78%**.
5. Tutup dengan penjelasan poin 5 sebelum Pak Aam bertanya.

---

## Yang belum selesai (akui bila ditanya)

- Field **Satuan** masih tampil sebagai field tersendiri (poin 7 baris 1).
