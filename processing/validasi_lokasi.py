"""Validasi geometri pada kanvas kerja Pipeline A."""

import cv2
import numpy as np

from processing.deteksi_field import (
    _batas_harga_sebelum_sama_dengan,
    _kotak_gabung,
)


def _mask_tinta(kanvas):
    abu = cv2.cvtColor(kanvas, cv2.COLOR_BGR2GRAY)
    biner = cv2.adaptiveThreshold(
        abu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13,
    )
    return abu, cv2.connectedComponentsWithStats(biner)[2][1:]


def _pasangan_titik(titik, tinggi, toleransi_x, jarak_min):
    return sorted(
        (x, (y + v + h) / 2)
        for x, y, _, h in titik for u, v, _, _ in titik
        if abs(x - u) <= toleransi_x and jarak_min(h) <= v - y <= tinggi
    )


def _pilih_header(solusi, toleransi, offset_kiri=4):
    if not solusi:
        return None
    y, x, nilai = min(solusi, key=lambda s: (s[0], s[1]))
    if any(abs(u - x) > toleransi or abs(v - y) > toleransi for v, u, _ in solusi):
        return None
    return tuple(_kotak_gabung([{
        "x1": max(k["x1"], x + offset_kiri), "x2": k["x2"],
        "y1": k["y1"], "y2": k["y2"],
    }]) for k in nilai[:2])


def _header_dari_garis(abu, stats, kandidat):
    lebar = abu.shape[1]
    tepi = cv2.Canny(abu, 45, 130)
    terdeteksi = cv2.HoughLinesP(
        tepi, 1, np.pi / 720, 90,
        minLineLength=round(0.35 * lebar), maxLineGap=round(0.04 * lebar),
    )
    if terdeteksi is None:
        return None
    garis = sorted(
        ((x, y, u, v) for x, y, u, v in terdeteksi[:, 0]
         if u > x and abs(v - y) < 0.08 * (u - x)),
        key=lambda g: g[1] + g[3],
    )
    solusi = []
    for lx, ly, rx, ry in garis:
        panjang = rx - lx
        miring = (ry - ly) / panjang

        def posisi_y(x):
            return ly + (x - lx) * miring

        if not any(
            min(u, rx) - max(x, lx) > 0.67 * panjang
            and 0.022 * panjang <= posisi_y(x) - y <= 0.075 * panjang
            and abs((v - y) / (u - x) - miring) < 0.025
            for x, y, u, v in garis
        ):
            continue
        titik = [
            (int(x), int(y), int(w), int(h)) for x, y, w, h, luas in stats
            if lx + 0.04 * panjang <= x <= lx + 0.28 * panjang
            and 0.012 * panjang <= y - posisi_y(x) <= 0.15 * panjang
            and 1 <= w <= 0.008 * panjang and 1 <= h <= 0.008 * panjang
            and luas >= 2
        ]
        dua = _pasangan_titik(titik, max(4, 0.011 * panjang), 2, lambda _: 2)
        for x, y in dua:
            deret = sorted(v - y for u, v in dua if abs(u - x) <= 3 and v - y > 7)
            if len(deret) < 2:
                continue
            jarak = deret[0]
            if not 0.012 * panjang <= jarak <= 0.036 * panjang:
                continue
            if not any(abs(d - 2 * jarak) <= 0.23 * jarak for d in deret[1:]):
                continue
            nilai = []
            for pusat in (y, y + jarak):
                calon = [
                    k for k in kandidat
                    if x - 0.80 * jarak <= k["x1"] <= x + 1.9 * jarak
                    and k["x2"] > x + 1.15 * jarak and k["w"] >= 1.15 * jarak
                    and abs(k["cy"] - pusat - miring * (k["cx"] - x)) <= 0.46 * jarak
                    and k["h"] <= 1.55 * jarak
                ]
                if not calon:
                    break
                nilai.append(min(calon, key=lambda k: (abs(k["x1"] - x), -k["w"])))
            if len(nilai) == 2 and nilai[0] is not nilai[1]:
                solusi.append((y, x, nilai))
    return _pilih_header(solusi, 8)


def _header_dari_label(stats, kandidat, lebar, nomor):
    if not nomor:
        return None
    _, y0, _, y1 = nomor
    tinggi = y1 - y0
    titik = [
        (int(x), int(y), int(w), int(h)) for x, y, w, h, luas in stats
        if y1 + 2 * tinggi < y < y1 + 11 * tinggi
        and 0.12 * lebar < x < 0.54 * lebar
        and 1 <= w <= 0.36 * tinggi and 1 <= h <= 0.30 * tinggi and luas >= 2
    ]
    dua = _pasangan_titik(titik, 0.52 * tinggi, 2, lambda h: max(2, h))
    solusi = []
    for x, y in dua:
        for u, v in dua:
            jarak = v - y
            if not 0.58 * tinggi <= jarak <= 1.65 * tinggi or abs(u - x) > 3:
                continue
            if not any(abs(a - x) <= 4 and abs(b - y - 2 * jarak) < 0.22 * jarak
                       for a, b in dua):
                continue
            nilai = []
            for pusat in (y, v, y + 2 * jarak):
                sebaris = [k for k in kandidat if abs(k["cy"] - pusat) < 0.52 * jarak]
                calon = [
                    k for k in sebaris
                    if x - 1.15 * jarak <= k["x1"] <= x + 2 * jarak
                    and k["x2"] > x + 1.35 * jarak and k["h"] <= 1.55 * jarak
                ]
                label = any(
                    k["x1"] < x - 1.5 * jarak and k["x2"] <= x + 0.25 * jarak
                    for k in sebaris
                )
                if not calon or not label:
                    break
                nilai.append(min(calon, key=lambda k: abs(k["cy"] - pusat) + 0.2 * abs(k["x1"] - x)))
            if len(nilai) == 3 and nilai[2]["w"] > nilai[1]["w"]:
                solusi.append((y, x, nilai))
    return _pilih_header(solusi, 6)


def validasi_header(kanvas, kandidat, baseline, nomor):
    """Kembalikan header aktif dan jangkar jika ada bukti penggantian."""
    abu, stats = _mask_tinta(kanvas)
    anchor = _header_dari_garis(abu, stats, kandidat)
    if anchor is None:
        anchor = _header_dari_label(stats, kandidat, abu.shape[1], nomor)
    if anchor is None:
        return baseline, None

    jarak = anchor[1]["cy"] - anchor[0]["cy"]
    if jarak <= 0:
        return baseline, None
    gagal = [
        k is None or abs(k["cy"] - a["cy"]) > 0.65 * jarak or k["h"] > 1.8 * jarak
        for k, a in zip(baseline, anchor)
    ]
    melintasi = any(
        k and k["y1"] <= anchor[0]["cy"] and k["y2"] >= anchor[1]["cy"]
        for k in baseline
    )
    if all(gagal) or (any(gagal) and melintasi):
        return anchor, anchor
    return baseline, None


def _ada_pemisah_harga(kanvas, kotak):
    crop = kanvas[kotak["y1"]:kotak["y2"], kotak["x1"]:kotak["x2"]]
    return min(crop.shape[:2]) >= 4 and (
        _batas_harga_sebelum_sama_dengan(crop, lindungi_digit=True) is not None
    )


def _pulihkan_harga(kanvas, baris, nama, jumlah):
    pasangan = [(j, p) for _, j, p in baris if j is not None and p is not None]
    if len(pasangan) < 2:
        return None
    tinggi = float(np.median([p["h"] for _, p in pasangan]))
    kiri = [p["x1"] for _, p in pasangan]
    geser = [p["cy"] - j["cy"] for j, p in pasangan]
    if np.ptp(kiri) > 0.9 * tinggi or np.ptp(geser) > 0.42 * tinggi:
        return None
    x = round(np.median(kiri))
    y = jumlah["cy"] + np.median(geser)
    if x <= jumlah["x2"] + 4 or abs(y - nama["cy"]) > 0.65 * nama["h"]:
        return None
    if any(abs(p["cy"] - y) < 0.8 * tinggi for _, p in pasangan):
        return None
    kotak = _kotak_gabung([{
        "x1": max(jumlah["x2"] + 4, round(x - 0.25 * tinggi)),
        "y1": max(0, round(y - 0.48 * tinggi)),
        "x2": min(kanvas.shape[1], round(x + 5 * tinggi)),
        "y2": min(kanvas.shape[0], round(y + 0.48 * tinggi)),
    }])
    return kotak if _ada_pemisah_harga(kanvas, kotak) else None


def validasi_baris(kanvas, kandidat, baris, anchor):
    """Koreksi pergeseran kolom dan cari harga kosong di baris yang sama."""
    hasil = []
    for nama, jumlah, harga in baris:
        h = nama["h"]
        if anchor and nama["x1"] - anchor[0]["x1"] > 6 * h:
            awal = anchor[0]["x1"]
            nama_kiri = [
                k for k in kandidat if abs(k["x1"] - awal) < 1.6 * h
                and k["w"] > 3 * h and abs(k["cy"] - nama["cy"]) < 0.48 * h
            ]
            jumlah_kiri = [
                k for k in kandidat if k["x2"] < nama["x1"] and k["x1"] > awal + 5 * h
                and abs(k["cy"] - nama["cy"]) < 0.45 * h and k["w"] > 1.8 * h
            ]
            if len(nama_kiri) == 1 and len(jumlah_kiri) == 1 and jumlah is not None:
                nama, jumlah, harga = nama_kiri[0], jumlah_kiri[0], jumlah
        kosong = harga is None and len(baris) >= 2
        if jumlah is not None and (kosong or (harga is not None and anchor)):
            batas = kanvas.shape[1] if kosong else harga["x1"]
            calon = [
                k for k in kandidat
                if jumlah["x2"] + 4 < k["x1"] < batas
                and k["x1"] >= 0.66 * kanvas.shape[1]
                and k["h"] >= 0.68 * h and k["w"] >= 1.5 * h
                and abs(k["cy"] - jumlah["cy"]) <= 0.38 * min(h, jumlah["h"])
                and not any(k is p for _, _, p in baris)
                and not any(abs(k["cy"] - p["cy"]) < abs(k["cy"] - jumlah["cy"])
                            for p, _, _ in baris if p is not nama)
                and _ada_pemisah_harga(kanvas, k)
            ]
            if len(calon) == 1:
                harga = calon[0]
        if harga is None and jumlah is not None:
            harga = _pulihkan_harga(kanvas, baris, nama, jumlah)
        hasil.append((nama, jumlah, harga))
    return hasil
