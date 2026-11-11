"""
Ground truth field nota untuk pengukuran CER.

Dibaca manual dari nota asli. Hanya dipakai untuk EVALUASI,
tidak pernah dipakai di dalam pipeline pengenalan.
"""

GT = {
    # nota_01 (NO: 0086)
    ("nota_01", "nomor"): "0086",
    ("nota_01", "penerima"): "Bapak Ir. Haryanto/CV Perkasa Konstruksi",
    ("nota_01", "proyek"): "Proyek PAMA L4",
    ("nota_01", "jumlah_1"): "65,00",
    ("nota_01", "jumlah_2"): "2.701,00",
    ("nota_01", "jumlah_3"): "2,00",
    ("nota_01", "harga_1"): "39.000",
    ("nota_01", "harga_2"): "139.500",
    ("nota_01", "harga_3"): "200.000",
    ("nota_01", "satuan_1"): "Pcs",
    ("nota_01", "satuan_2"): "Pcs",
    ("nota_01", "satuan_3"): "Rit",
    ("nota_01", "produk_1"): "Kanstein PU 50x30x18x13 Cm",
    ("nota_01", "produk_2"): "Paving Diagonal 8 Cm (K)",
    ("nota_01", "produk_3"): "Ongkos Buruh Muat",

    # nota_26 (NO: 0099)
    ("nota_26", "nomor"): "0099",
    ("nota_26", "penerima"): "PT. SOLUSI INDUSTRI ENERGI",
    ("nota_26", "proyek"): "GKM Gree Towwer Jln. TB simatupang Kav. 89 G Lantal 19 - 08",
    ("nota_26", "jumlah_1"): "9.000,00",
    ("nota_26", "harga_1"): "10.550",
    ("nota_26", "satuan_1"): "Pcs",
    ("nota_26", "produk_1"): "BATA BETON 40x20x10 Cm",

    # nota_52 (NO: 0299)
    ("nota_52", "nomor"): "0299",
    ("nota_52", "penerima"): "Bpk Indra Hadiyanto",
    ("nota_52", "proyek"): "KSO Adhi PG Traning Center PSSI IKN",
    # Nota_52 menulis jumlah tanpa pemisah ribuan pada baris 1 ("400")
    # dan memakai koma pada baris 2 ("3,800"); dikonfirmasi lewat
    # pembacaan ulang gambar asli 1600 px.
    ("nota_52", "jumlah_1"): "400",
    ("nota_52", "jumlah_2"): "3,800",
    ("nota_52", "harga_1"): "35,696",
    ("nota_52", "harga_2"): "20,240",
    ("nota_52", "satuan_1"): "Pcs",
    ("nota_52", "satuan_2"): "Pcs",
    ("nota_52", "produk_1"): "Konstein S 40x30x15x8 K-300",
    ("nota_52", "produk_2"): "Konstein Jepit 40x20x10 K-300",
}