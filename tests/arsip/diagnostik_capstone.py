from pathlib import Path
import cv2
import numpy as np

from uji_deteksi_baris import deteksi_baris
from processing.deteksi_field import (
    deteksi_produk,
    deteksi_field_lain,
    crop_field,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "diagnostik"
OUT.mkdir(exist_ok=True)


def baca(path):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan(path, gambar):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", gambar)
    if ok:
        data.tofile(str(path))


def mask_merah(gambar):
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 65, 45), (12, 255, 255)),
        cv2.inRange(hsv, (150, 65, 45), (179, 255, 255)),
    )


def kandidat_nomor_merah(kecil):
    """Cari kandidat tulisan merah 'NO : xxxx' di area atas."""
    H, W = kecil.shape[:2]
    merah = mask_merah(kecil)
    merah = cv2.morphologyEx(
        merah, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(merah, connectivity=8)
    kandidat = []
    for i in range(1, n):
        x, y, w, h, luas = stats[i]
        if luas < 120 or h < 6:
            continue
        if y > 0.30 * H:
            continue
        if x < 0.45 * W:
            continue
        kandidat.append((x, y, w, h, luas))
    kandidat.sort(key=lambda z: z[0])
    return kandidat, merah


def pisah_digit_merah(crop):
    """Isolasi digit dari 'NO : xxxx' berdasarkan jarak komponen."""
    merah = mask_merah(crop)
    proyeksi = (np.count_nonzero(merah, axis=0) > 0).astype(np.int8)
    w = crop.shape[1]
    # cari celah terlebar antar kelompok tinta merah
    gap_terbaik, posisi = 0, None
    run = 0
    for x in range(w):
        if proyeksi[x] == 0:
            run += 1
        else:
            if run > gap_terbaik:
                gap_terbaik, posisi = run, x
            run = 0
    return merah, proyeksi, gap_terbaik, posisi


def main():
    daftar = [
        ("nota_01", BASE / "data_nota_asli" / "nota_01.jpeg"),
        ("nota_26", BASE / "data_uji" / "nota_26.jpeg"),
        ("nota_52", BASE / "data_uji" / "nota_52.jpeg"),
        ("nota_54", Path(r"D:\PROJECT MARRIENA\data_uji\nota_54.jpeg")),
    ]

    for nama, path in daftar:
        print("\n================", nama, "================")
        if not path.exists():
            print("  FILE TIDAK ADA:", path)
            continue

        gambar = baca(path)
        if gambar is None:
            print("  GAGAL BACA")
            continue

        kecil, biner, visual, kotak = deteksi_baris(gambar)
        H, W = kecil.shape[:2]

        kandidat = [{
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "w": x2 - x1, "h": y2 - y1,
        } for (x1, y1, x2, y2) in kotak]

        produk = deteksi_produk(kandidat, W, H)
        info = deteksi_field_lain(kandidat, W, H, produk)

        print(f"  Ukuran proses: {W}x{H} | kandidat baris: {len(kandidat)}")
        print(f"  Produk terdeteksi: {len(produk)}")
        for i, p in enumerate(produk, 1):
            print(f"    - produk {i}: x={p['x1']}-{p['x2']} y={p['y1']}-{p['y2']}")

        nn = info["nomor_nota"]
        tg = info["tanggal"]
        print(f"  [persentase] nomor_nota box: {nn['x1']},{nn['y1']},{nn['x2']},{nn['y2']}")
        print(f"  [persentase] tanggal    box: {tg['x1']},{tg['y1']},{tg['x2']},{tg['y2']}")

        # kandidat nomor merah (metode yang 'pernah berhasil')
        kmerah, mmerah = kandidat_nomor_merah(kecil)
        print(f"  Red-mask kandidat nomor (atas-kanan): {len(kmerah)}")
        for (x, y, w, h, luas) in kmerah:
            print(f"    merah: x={x} y={y} w={w} h={h} luas={luas}")

        folder = OUT / nama
        simpan(folder / "01_deteksi_baris.png", visual)
        simpan(folder / "02_biner.png", biner)
        simpan(folder / "03_mask_merah.png", mmerah)

        crop_nn = crop_field(kecil, nn, margin=4)
        if crop_nn is not None:
            simpan(folder / "10_nomor_persentase.png", crop_nn)

        crop_tg = crop_field(kecil, tg, margin=4)
        if crop_tg is not None:
            simpan(folder / "11_tanggal_persentase.png", crop_tg)

        # crop nomor dari kandidat merah + isolasi digit
        if kmerah:
            x, y, w, h, _ = max(kmerah, key=lambda z: z[2] * z[3])
            x1 = max(0, x - 4); y1 = max(0, y - 4)
            x2 = min(W, x + w + 4); y2 = min(H, y + h + 4)
            crop_merah = kecil[y1:y2, x1:x2].copy()
            simpan(folder / "20_nomor_merah.png", crop_merah)
            _, proy, gap, pos = pisah_digit_merah(crop_merah)
            print(f"    isolasi digit: lebar={crop_merah.shape[1]} gap_terlebar={gap} di x={pos}")
            if pos is not None:
                digit = crop_merah[:, max(0, pos - 2):]
                simpan(folder / "21_digit_kandidat.png", digit)

        print("  Output:", folder)


if __name__ == "__main__":
    main()