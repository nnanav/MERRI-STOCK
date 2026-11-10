
from pathlib import Path
import sys
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "hasil_crop_produk"
OUTPUT.mkdir(exist_ok=True)


def baca_gambar(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan(nama, gambar):
    path = OUTPUT / nama
    sukses, data = cv2.imencode(".png", gambar)
    if sukses:
        data.tofile(str(path))


def pilih_area(gambar, nomor):
    tinggi, lebar = gambar.shape[:2]

    skala = min(
        1200 / lebar,
        850 / tinggi,
        1.0
    )

    tampil = cv2.resize(
        gambar,
        None,
        fx=skala,
        fy=skala
    )

    print(f"\nPILIH NAMA PRODUK {nomor}")
    print("Tarik kotak mengelilingi SATU baris produk.")
    print("Tekan ENTER setelah memilih.")
    print("Tekan C jika ingin membatalkan pilihan.")

    nama_window = f"Pilih Produk {nomor}"

    x, y, w, h = cv2.selectROI(
        nama_window,
        tampil,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(nama_window)

    if w == 0 or h == 0:
        return None

    x1 = max(0, round(x / skala))
    y1 = max(0, round(y / skala))
    x2 = min(lebar, round((x + w) / skala))
    y2 = min(tinggi, round((y + h) / skala))

    return gambar[y1:y2, x1:x2].copy()


def proses_crop(crop, nomor):
    # Perbesar crop agar mudah diperiksa
    besar = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    abu = cv2.cvtColor(
        besar,
        cv2.COLOR_BGR2GRAY
    )

    abu = cv2.GaussianBlur(
        abu,
        (3, 3),
        0
    )

    _, biner = cv2.threshold(
        abu,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Mendeteksi komponen tinta,
    # bukan memaksakan jumlah karakter.
    jumlah, label, statistik, pusat = (
        cv2.connectedComponentsWithStats(
            biner,
            connectivity=8
        )
    )

    visual = besar.copy()
    kotak = []

    for i in range(1, jumlah):
        x, y, w, h, luas = statistik[i]

        # Hilangkan noise yang sangat kecil.
        if luas < 8 or h < 4 or w < 2:
            continue

        kotak.append((x, y, w, h))

    kotak.sort(key=lambda item: item[0])

    for x, y, w, h in kotak:
        cv2.rectangle(
            visual,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1
        )

    simpan(f"produk_{nomor}_crop.png", besar)
    simpan(f"produk_{nomor}_biner.png", biner)
    simpan(f"produk_{nomor}_komponen.png", visual)

    print(
        f"Produk {nomor}: "
        f"{len(kotak)} komponen tinta terdeteksi"
    )


def main():
    if len(sys.argv) < 2:
        print(
            'Penggunaan: python uji_crop_produk.py '
            '"lokasi_gambar.jpeg"'
        )
        return

    path = Path(sys.argv[1])

    gambar = baca_gambar(path)

    if gambar is None:
        print("Gagal membuka gambar:", path)
        return

    print("Ukuran gambar:", gambar.shape[1], "x", gambar.shape[0])

    for nomor in (1, 2):
        crop = pilih_area(gambar, nomor)

        if crop is None:
            print("Pemilihan dibatalkan.")
            return

        proses_crop(crop, nomor)

    cv2.destroyAllWindows()

    print("\nSelesai.")
    print("Hasil tersimpan di:", OUTPUT)


if __name__ == "__main__":
    main()
