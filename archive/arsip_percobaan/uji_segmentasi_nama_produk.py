"""Uji segmentasi awal dua baris produk dari nota_52.jpeg.
Jalankan: python uji_segmentasi_nama_produk.py "path_ke_nota_52.jpeg"
Tidak mengubah file foto asli atau modul OCR yang sudah ada.
"""
from pathlib import Path
import sys
import csv
import cv2
import numpy as np

# Koordinat dalam citra asli 1600 x 1227 piksel (x1, y1, x2, y2).
# Ini khusus nota_52.jpeg, bukan koordinat universal semua nota.
ROI = {
    "produk_1": (594, 293, 835, 313),
    "produk_2": (594, 318, 835, 338),
}


def baca(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def segmen_kolom(biner, min_lebar=2):
    """Calon segmen berdasarkan celah kolom. Huruf bersambung bisa tetap menyatu."""
    proyeksi = np.any(biner > 0, axis=0)
    batas = np.diff(np.r_[False, proyeksi, False].astype(np.int8))
    awal = np.where(batas == 1)[0]
    akhir = np.where(batas == -1)[0]
    return [(int(a), int(b)) for a, b in zip(awal, akhir)
            if b - a >= min_lebar]


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Contoh: python uji_segmentasi_nama_produk.py "data_uji\\nota_52.jpeg"')
    path = Path(sys.argv[1])
    if not path.is_file():
        raise FileNotFoundError(path)
    img = baca(path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    if (img.shape[1], img.shape[0]) != (1600, 1227):
        raise ValueError(f"Ukuran nota {img.shape[1]}x{img.shape[0]}; ROI hanya berlaku untuk foto 1600x1227.")

    out = Path(__file__).resolve().parent / 'hasil_segmentasi_nota52'
    out.mkdir(exist_ok=True)
    rows = []

    for nama, (x1, y1, x2, y2) in ROI.items():
        potongan = img[y1:y2, x1:x2]
        abu = cv2.cvtColor(potongan, cv2.COLOR_BGR2GRAY)
        # Memperbesar agar lebih mudah diperiksa secara visual; bukan menambah informasi asli.
        besar = cv2.resize(abu, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, biner = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        segmen = segmen_kolom(biner)
        overlay = cv2.cvtColor(abu, cv2.COLOR_GRAY2BGR)
        for i, (a,b) in enumerate(segmen, start=1):
            cv2.rectangle(overlay, (a, 0), (b-1, abu.shape[0]-1), (0, 0, 255), 1)
            rows.append((nama, i, x1+a, x1+b, y1, y2, b-a))
        overlay = cv2.resize(overlay, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / f'{nama}_crop.png'), besar)
        cv2.imwrite(str(out / f'{nama}_biner.png'), biner)
        cv2.imwrite(str(out / f'{nama}_batas.png'), overlay)
        print(f'{nama}: {len(segmen)} calon segmen (belum tentu sama dengan jumlah karakter)')

    with (out / 'calon_segmen.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['baris', 'nomor', 'x_awal', 'x_akhir_eksklusif', 'y_awal', 'y_akhir_eksklusif', 'lebar'])
        writer.writerows(rows)
    print('Hasil disimpan di:', out)
    print('Periksa *_batas.png. Kotak yang memuat beberapa huruf perlu diperbaiki pada tahap berikutnya.')


if __name__ == '__main__':
    main()
