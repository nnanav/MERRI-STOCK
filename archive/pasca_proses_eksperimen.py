"""
Prototipe pasca-proses per-field (BELUM diintegrasikan ke app.py).

Tujuan: mengoreksi hasil OCR berdasarkan alfabet yang sah untuk tiap
field. Tidak memakai ground truth, tidak mengubah model CNN.
"""

# Huruf yang sering tertukar dengan digit (bentuk mirip).
KE_DIGIT = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
    "/": ".", ",": ".", ":": ".", ";": ".",
}

# Huruf yang sering tertukar dengan huruf lain (untuk field satuan).
KE_HURUF = {
    "?": "P", "$": "S", "@": "a", "0": "O", "1": "I", "5": "S",
    "8": "B", "6": "G", "2": "Z", "4": "A", "3": "E", "7": "T",
}

# Satuan baku pada nota (huruf kecil).
SATUAN_BAKU = ["pcs", "m2", "m3", "rit", "lsn", "bh", "kg", "set", "unit"]


def _rapikan_digit(teks, perbaiki_pemisah=True):
    hasil = []
    for ch in teks:
        if ch.isdigit():
            hasil.append(ch)
        elif ch in KE_DIGIT:
            hasil.append(KE_DIGIT[ch])
    teks = "".join(hasil)

    if not perbaiki_pemisah:
        # Harga: pemisah sudah benar dari OCR, jangan diubah.
        return teks

    # Jumlah: pemisah '/' atau ',' hasil salah baca koma -> jadikan koma.
    if "/" in teks:
        teks = teks.replace("/", ",")
    return teks


def _rapikan_satuan(teks):
    # Buang spasi dan tanda baca; sisakan huruf.
    hasil = []
    for ch in teks:
        if ch.isalpha():
            hasil.append(ch)
        elif ch in KE_HURUF:
            hasil.append(KE_HURUF[ch])
    teks = "".join(hasil)
    if not teks:
        return teks

    rendah = teks.lower()

    # Cocokkan ke satuan baku (awalan cukup, mis. 'csx' -> 'pcs').
    for baku in SATUAN_BAKU:
        if rendah.startswith(baku) or baku in rendah:
            if baku == "pcs":
                return "Pcs"
            if baku in ("m2", "m3"):
                return baku.upper()
            return baku.capitalize()

    return teks


def rapikan_field(teks, jenis):
    """
    Menerapkan pasca-proses sesuai jenis field.
    jenis: 'nomor' | 'jumlah' | 'harga' | 'satuan' | lainnya (tidak diubah)
    """
    if not teks:
        return teks

    if jenis == "nomor":
        # Nomor nota: hanya digit, tanpa pemisah.
        return "".join(
            ch if ch.isdigit() else KE_DIGIT.get(ch, "")
            for ch in teks
        )
    if jenis == "jumlah":
        return _rapikan_digit(teks, perbaiki_pemisah=True)
    if jenis == "harga":
        return _rapikan_digit(teks, perbaiki_pemisah=False)
    if jenis == "satuan":
        return _rapikan_satuan(teks)
    return teks