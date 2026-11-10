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
3. **Preprocessing** - lihat mask merah nomor nota dan tahapan pengolahan
4. **Segmentasi** - bounding box per field dan pemisahan per karakter
5. **Pengenalan** - hasil OCR semua field (label dari CNN)
6. **Evaluasi** - isi ground truth, CER dihitung otomatis

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
python tests/uji_integrasi_segmen.py  # cek alur upload sampai evaluasi
python tests/ukur_cer.py              # ukur CER terhadap ground truth
```

## Catatan

- Model CNN menggunakan 70 kelas karakter (huruf besar, huruf kecil, angka, simbol).
- Template Matching memakai seluruh gambar pada `dataset/huruf`, `dataset/angka`,
  dan `dataset/simbol`.
- Folder `archive/` berisi backup kode dan model versi sebelumnya; tidak dipakai
  saat aplikasi berjalan dan aman untuk diabaikan.
