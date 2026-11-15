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
6. **Evaluasi** - CER per field utuh terhadap ground truth (status exact match)

## Struktur Folder

```
app.py                  Dashboard Streamlit (6 halaman)
requirements.txt        Daftar dependensi
processing/             Modul inti pengolahan citra dan OCR
models/                 Model CNN dan daftar label
dataset/                Data karakter (template TM dan data latih CNN)
data_nota_asli/         Gambar nota untuk pengujian
data_uji/               Gambar nota uji tambahan
scripts/                Skrip pelatihan dan validasi dataset
tools/                  Alat bantu pelabelan template
tests/                  Skrip pengujian
archive/                Backup kode dan model versi sebelumnya
```

### Modul pada `processing/`

| File | Fungsi |
|---|---|
| `dataset_loader.py` | Memuat gambar, normalisasi karakter 32x32 |
| `deteksi_baris.py` | Mendeteksi baris teks pada nota |
| `deteksi_field.py` | Menemukan field: nomor, penerima, proyek, produk, jumlah, satuan, harga |
| `pemrosesan_field.py` | Isolasi nomor nota, pembersihan minus, pemisahan jumlah/satuan |
| `segmentasi_karakter.py` | Projection profile + Dynamic Programming + CNN |
| `pengenalan_karakter.py` | Memuat CNN dan Template Matching, penggabungan keputusan |
| `lokalisasi_template.py` | Pencocokan pola karakter dengan template (eksperimen) |

## Metode

| Tahap | Metode |
|---|---|
| Deteksi baris | Adaptive threshold, morfologi, connected components |
| Segmentasi karakter | Projection profile + Dynamic Programming |
| Pengenalan karaker | CNN (prediksi utama) + Template Matching (pencocok pola) |
| Evaluasi | Character Error Rate (CER) |

## Pelatihan Ulang CNN

```powershell
python scripts/latih_cnn.py
```

Hasil pelatihan disimpan ke `models/cnn_karakter.keras` dan `models/labels.json`.
**Backup model lama terlebih dahulu** jika ingin membandingkan hasil.

## Pengujian

```powershell
python tests/uji_app_streamlit.py     # cek 6 halaman dashboard
python tests/uji_info_dataset.py      # cek konsistensi angka data latih
python tests/uji_integrasi_segmen.py  # cek alur upload sampai evaluasi
python tests/ukur_cer.py              # evaluasi CER per field vs ground truth
python tests/ukur_segmen_gt.py        # jumlah kotak vs jumlah karakter GT
python tests/uji_upload.py            # cek jalur upload (decode gambar + OCR)
python tests/uji_halaman_berisi.py    # render halaman dengan nota nyata
python tests/audit_segmentasi.py      # cek karakter tergabung/terpotong (16 nota)
python tests/audit_generalisasi.py    # cek kewajaran hasil pada nota baru
python tests/sapu_ambang.py           # bandingkan ambang lebar komponen
python tests/diag_segmen.py           # rincian segmen per field
```

## Catatan

- Model CNN menggunakan 70 kelas karakter (huruf besar, huruf kecil, angka, simbol).
- Template Matching memakai seluruh gambar pada `dataset/huruf`, `dataset/angka`,
  dan `dataset/simbol`.
- Folder `archive/` berisi backup kode dan model versi sebelumnya; tidak dipakai
  saat aplikasi berjalan dan aman untuk diabaikan.
- **Memori.** TensorFlow + model CNN membutuhkan sekitar 0,5 GB RAM. Pada mesin
  ber-RAM kecil (mis. 4 GB), jalankan hanya SATU server Streamlit dan tutup
  aplikasi lain. Bila muncul pesan `Insufficient memory` dari OpenCV, artinya
  memori sistem habis (bukan kesalahan kode): hentikan server lain lebih dulu
  (`Get-Process python,streamlit | Stop-Process`), lalu jalankan ulang.
