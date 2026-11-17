# MERRI-STOCK

Sistem Pengenalan Teks Nota Transaksi CV Merriena Jaya.

Aplikasi membaca gambar nota transaksi, mengenali karakter pada setiap field,
lalu menampilkan hasilnya sebagai teks.

## Cara Menjalankan

```powershell
cd "D:\CAPSTONE MARRIENA"
pip install -r requirements.txt
streamlit run app.py
```

Buka browser pada alamat yang ditampilkan (biasanya http://localhost:8501).

## Alur Penggunaan

1. **Business Understanding** - penjelasan sistem, data, dan metode
2. **Upload Nota** - unggah gambar nota, lalu klik **Proses Nota**
3. **Preprocessing** - crop kertas nota, peningkatan kualitas, dan binerisasi
4. **Segmentasi** - bounding box per field dan pemisahan per karakter
5. **Pengenalan** - hasil OCR semua field
6. **Hasil Ekstraksi** - periksa dan koreksi hasil OCR, lalu simpan/unduh
7. **Evaluasi** - tabel per field (Field, Ground Truth, Hasil OCR, Status)

## Struktur Folder

```
app.py                  Dashboard Streamlit (7 halaman)
PROMPT_AI.md            Prompt siap pakai untuk meminta AI menjelaskan program
requirements.txt        Daftar dependensi
processing/             Modul inti pengolahan citra dan OCR
models/                 Model CNN dan daftar label
dataset/                Data karakter (template TM dan data latih CNN)
data_nota_asli/         Gambar nota untuk pengujian
data_uji/               Gambar nota uji tambahan
scripts/                Skrip pelatihan dan validasi dataset
tools/                  Alat bantu pelabelan template
tests/                  Skrip pengujian
output/                 Hasil pengujian dan gambar data latih (dibuat ulang)
database/               Basis data SQLite transaksi (dibuat saat Simpan Data)
```

### Modul pada `processing/`

| File | Fungsi |
|---|---|
| `dataset_loader.py` | Memuat gambar, normalisasi karakter 32x32 |
| `deteksi_baris.py` | Mendeteksi baris teks pada nota |
| `deteksi_field.py` | Menemukan field: nomor, penerima, proyek, produk, jumlah, satuan, harga |
| `pemrosesan_field.py` | Isolasi nomor nota, pembersihan minus, pemisahan jumlah/satuan |

Catatan: field **satuan** tetap dideteksi karena dipakai untuk memisahkan
jumlah dari satuannya, tetapi tidak ditampilkan di halaman Segmentasi,
Pengenalan, maupun Evaluasi karena bukan bagian data transaksi yang disimpan.
| `segmentasi_karakter.py` | Projection profile + Dynamic Programming + CNN |
| `pengenalan_karakter.py` | Memuat CNN dan Template Matching, penggabungan keputusan |
| `lokalisasi_template.py` | Pencocokan pola karakter dengan template (eksperimen) |
| `data_transaksi.py` | Validasi angka nota, simpan SQLite, dan unduh CSV |

## Metode

| Tahap | Metode |
|---|---|
| Deteksi baris | Adaptive threshold, morfologi, connected components |
| Segmentasi karakter | Projection profile + Dynamic Programming |
| Pengenalan karaker | CNN (prediksi utama) + Template Matching (pencocok pola) |
| Evaluasi | Status Benar/Salah per field (exact match); CER tersedia lewat skrip uji |

## Prompt AI untuk Menjelaskan Program

`PROMPT_AI.md` berisi prompt siap pakai (untuk ChatGPT/Claude/Gemini) yang
menjelaskan sistem ini secara lengkap: identitas, alur 5 tahap, metode tiap
tahap beserta parameter, dataset, arsitektur CNN, cara evaluasi, hasil
pengukuran, dan keterbatasan yang diketahui. Salin isi antara penanda
`=== MULAI PROMPT ===` dan `=== AKHIR PROMPT ===`, lalu tempel ke AI.

## Pelatihan Ulang CNN

```powershell
python scripts/latih_cnn.py
```

Hasil pelatihan disimpan ke `models/cnn_karakter.keras` dan `models/labels.json`.
**Backup model lama terlebih dahulu** jika ingin membandingkan hasil.

## Gambar Data Latih Karakter

Membuat gambar grid seluruh kelas karakter dari dataset asli (untuk laporan):

```powershell
python scripts/buat_gambar_data_latih.py                      # karakter gelap di latar putih
python scripts/buat_gambar_data_latih.py --gaya latih          # persis input CNN (putih di hitam)
python scripts/buat_gambar_data_latih.py --kolom 10            # atur jumlah kolom
```

Hasil disimpan ke `output/data_latih_karakter.png` (atau `output/data_latih_input_cnn.png`).
Setiap kotak berisi satu contoh asli hasil normalisasi 32x32, dengan nomor kelas
dan kode (`A.U` huruf besar, `a.L` huruf kecil, `, S` simbol).

## Hasil Ekstraksi dan Penyimpanan

Halaman **Hasil Ekstraksi** memakai hasil OCR dari halaman Pengenalan Karakter
(tidak menjalankan OCR ulang). Nilai OCR ditampilkan sebagai isian yang dapat
dikoreksi; hasil OCR asli tidak diubah dan halaman Evaluasi tetap memakai hasil
OCR tersebut.

- **Pratinjau.** Gambar nota yang ditampilkan adalah hasil crop kertas nota
  (koreksi perspektif + peningkatan kualitas dari tahap Preprocessing), bukan
  foto unggahan asli, agar tulisan mudah dibaca saat memeriksa isian.
- **Validasi.** Nomor Nota, Nama Penerima, dan Proyek wajib diisi. Jumlah harus
  angka positif dan Harga Satuan tidak boleh negatif. Angka format Indonesia
  (`9.000`, `10.550`, `9.000,00`) dibaca sesuai nilainya, sedangkan angka rusak
  seperti `9.000P00` atau `200a000` ditolak dan harus diperbaiki dulu.
- **Simpan.** Data disimpan ke `database/transaksi.db` (SQLite) dalam dua tabel:
  `transaksi` (informasi utama nota) dan `transaksi_produk` (satu baris per
  produk). Sidik isi data mencegah penyimpanan ganda akibat klik berulang.
- **Unduh.** Tombol **Unduh Hasil** menghasilkan CSV dengan satu baris per
  produk. Tombol aktif setelah data tersimpan, sehingga yang diunduh hanya data
  yang sudah dikoreksi dan dikonfirmasi.

## Pengujian

```powershell
python tests/uji_app_streamlit.py     # cek 7 halaman dashboard
python tests/uji_info_dataset.py      # cek konsistensi angka data latih
python tests/uji_integrasi_segmen.py  # cek alur upload sampai evaluasi
python tests/ukur_cer.py              # evaluasi CER per field vs ground truth
python tests/ukur_segmen_gt.py        # jumlah kotak vs jumlah karakter GT
python tests/uji_upload.py            # cek jalur upload (decode gambar + OCR)
python tests/uji_halaman_tidak_blank.py  # pastikan tiap halaman ada isinya
python tests/uji_ui_segmentasi.py     # cek UI Segmentasi (6 tab field)
python tests/uji_ui_pengenalan.py     # cek UI Pengenalan Karakter (6 tab field)
python tests/uji_ui_evaluasi.py       # cek UI Evaluasi (tanpa CER)
python tests/uji_hasil_ekstraksi.py   # cek koreksi, validasi, simpan SQLite, CSV
python tests/ukur_kecepatan.py        # ukur waktu muat halaman
python tests/audit_segmentasi.py      # cek karakter tergabung/terpotong (16 nota)
python tests/audit_generalisasi.py    # cek kewajaran hasil pada nota baru
python tests/sapu_ambang.py           # bandingkan ambang lebar komponen
python tests/diag_segmen.py           # rincian segmen per field
```

## Catatan

- Model CNN menggunakan 70 kelas karakter (huruf besar, huruf kecil, angka, simbol).
- Template Matching memakai seluruh gambar pada `dataset/huruf`, `dataset/angka`,
  dan `dataset/simbol`.
- Folder `output/` berisi hasil pengujian dan gambar data latih; isinya dibuat
  ulang saat program dijalankan sehingga aman dihapus.
- `database/transaksi.db` berisi data transaksi yang sudah dikonfirmasi pengguna.
  Berkas ini **bukan** hasil sementara: jangan dihapus saat membersihkan
  `output/`. Folder `database/` diabaikan Git agar data nota tidak ikut terunggah.
- **Memori.** TensorFlow + model CNN membutuhkan sekitar 0,5 GB RAM. Pada mesin
  ber-RAM kecil (mis. 4 GB), jalankan hanya SATU server Streamlit dan tutup
  aplikasi lain. Bila muncul pesan `Insufficient memory` dari OpenCV, artinya
  memori sistem habis (bukan kesalahan kode): hentikan server lain lebih dulu
  (`Get-Process python,streamlit | Stop-Process`), lalu jalankan ulang.
