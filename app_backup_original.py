
def ambil_angka_nomor_nota(crop):
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    merah = cv2.bitwise_or(
        cv2.inRange(
            hsv, (0, 65, 45), (12, 255, 255)
        ),
        cv2.inRange(
            hsv, (150, 65, 45), (179, 255, 255)
        )
    )

    # Ambil batas tulisan merah yang sebenarnya.
    ys, xs = np.where(merah > 0)

    if len(xs) == 0:
        return None

    kiri = max(0, int(xs.min()) - 2)
    kanan = min(w, int(xs.max()) + 3)

    atas = max(0, int(ys.min()) - 2)
    bawah = min(h, int(ys.max()) + 3)

    lebar_teks = kanan - kiri

    if lebar_teks <= 0 or bawah <= atas:
        return None

    # Ambil wilayah kanan tempat digit nomor berada.
    # Ini fallback berbasis tata letak "NO : 0299".
    awal_digit = kiri + int(0.39 * lebar_teks)

    hasil = crop[
        atas:bawah,
        awal_digit:kanan
    ].copy()

    return hasil if hasil.size > 0 else None
