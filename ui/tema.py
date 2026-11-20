"""
Tema visual terpusat aplikasi MERRI-STOCK.

Berisi satu sumber CSS dan beberapa helper tampilan (header halaman,
sidebar, chip status). Semua fungsi interaktif tetap memakai komponen
native Streamlit; CSS hanya memperbaiki tampilannya.

Palet: Navy #142D4E, Teal #0D9488, Background #F4F7FB,
Putih #FFFFFF, Border #DCE5EF, Teks sekunder #64748B.
"""

import html

import streamlit as st


NAVY = "#142D4E"
TEAL = "#0D9488"
BG = "#F4F7FB"
WHITE = "#FFFFFF"
BORDER = "#DCE5EF"
MUTED = "#64748B"

# Ikon garis 24x24. Dipakai pada eyebrow halaman dan menu sidebar.
IKON = {
    "bu": (
        "<path d='M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.3.2.5.6.5 1v.1h6"
        "v-.1c0-.4.2-.8.5-1A6 6 0 0 0 12 3z'/>"
    ),
    "upload": (
        "<path d='M12 16V4M7 9l5-5 5 5'/>"
        "<path d='M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2'/>"
    ),
    "preproses": (
        "<path d='M4 6h9M18 6h2M4 12h4M12 12h8M4 18h9M18 18h2'/>"
        "<circle cx='15.5' cy='6' r='2'/><circle cx='9.5' cy='12' r='2'/>"
        "<circle cx='15.5' cy='18' r='2'/>"
    ),
    "segmen": (
        "<rect x='4' y='4' width='6' height='6' rx='1.4'/>"
        "<rect x='14' y='4' width='6' height='6' rx='1.4'/>"
        "<rect x='4' y='14' width='6' height='6' rx='1.4'/>"
        "<rect x='14' y='14' width='6' height='6' rx='1.4'/>"
    ),
    "kenal": (
        "<path d='M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8"
        "M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16'/>"
        "<path d='M9 9h6M9 12h6M9 15h3'/>"
    ),
    "ekstraksi": (
        "<path d='M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7z'/>"
        "<path d='M14 3v4h4M9 13h6M9 17h4'/>"
    ),
    "evaluasi": (
        "<circle cx='12' cy='12' r='8'/><path d='M8.8 12.2l2.2 2.2 4.2-4.4'/>"
    ),
}


def svg(nama, ukuran=15, tebal=1.8):
    """Kembalikan string SVG inline untuk ikon bernama."""
    isi = IKON.get(nama, IKON["evaluasi"])
    return (
        f"<svg viewBox='0 0 24 24' width='{ukuran}' height='{ukuran}' "
        "fill='none' stroke='currentColor' stroke-width='{t}' "
        "stroke-linecap='round' stroke-linejoin='round'>"
    ).format(t=tebal) + isi + "</svg>"



CSS_DASAR = """
/* ---------- Dasar & layout ---------- */
:root {
  --ms-navy: #142D4E;
  --ms-teal: #0D9488;
  --ms-bg: #F4F7FB;
  --ms-white: #FFFFFF;
  --ms-border: #DCE5EF;
  --ms-muted: #64748B;
  --ms-teal-soft: #E6F4F1;
  --ms-radius: 14px;
  --ms-radius-sm: 10px;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.stApp,
[data-testid="stAppViewContainer"] { background: var(--ms-bg); }

header[data-testid="stHeader"] { background: transparent; }

[data-testid="stMainBlockContainer"] {
  padding: 1.9rem 2.4rem 4rem 2.4rem;
  max-width: 1280px;
}

[data-testid="stVerticalBlock"] { gap: 0.85rem; }

/* ---------- Tipografi ---------- */
[data-testid="stMainBlockContainer"] h1 {
  font-size: 1.78rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--ms-navy);
  margin: 0 0 0.15rem 0;
  line-height: 1.25;
}

[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 {
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: var(--ms-navy);
  margin: 0.35rem 0 0.1rem 0;
}

[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] li {
  font-size: 0.9rem;
  line-height: 1.65;
  color: #334155;
}

[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {
  font-size: 0.84rem;
  line-height: 1.6;
  color: var(--ms-muted);
}

/* ---------- Eyebrow / label langkah halaman ---------- */
.ms-eyebrow {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.71rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ms-teal);
  margin: 0 0 0.5rem 0;
}

.ms-eyebrow svg { flex: none; }

/* ---------- Panel putih ---------- */
[data-testid="stMainBlockContainer"]
  [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--ms-white);
  border: 1px solid var(--ms-border);
  border-radius: var(--ms-radius);
  padding: 1rem 1.1rem;
  box-shadow: 0 1px 2px rgba(20, 45, 78, 0.04);
}

[data-testid="stMainBlockContainer"]
  [data-testid="stVerticalBlockBorderWrapper"] h3 {
  margin-top: 0;
}

/* ---------- Gambar ---------- */
[data-testid="stMainBlockContainer"] [data-testid="stImage"] img {
  border-radius: var(--ms-radius-sm);
  border: 1px solid var(--ms-border);
  background: var(--ms-white);
}

[data-testid="stMainBlockContainer"] [data-testid="stImage"] figcaption p {
  font-size: 0.78rem;
  color: var(--ms-muted);
}
"""


def _svg_mask(nama):
    """Data URI SVG untuk mask CSS (warna mengikuti currentColor)."""
    isi = IKON.get(nama, IKON["evaluasi"])
    svg_teks = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        "fill='none' stroke='black' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'>" + isi + "</svg>"
    )
    terenkode = (
        svg_teks.replace("<", "%3C").replace(">", "%3E")
        .replace("#", "%23").replace('"', "'")
    )
    return 'url("data:image/svg+xml,' + terenkode + '")'


_CSS_SIDEBAR = """
/* ---------- Sidebar navy ---------- */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
  background: #142D4E;
}

[data-testid="stSidebar"] {
  border-right: 1px solid rgba(255, 255, 255, 0.07);
}

[data-testid="stSidebarUserContent"] {
  padding: 0.5rem 0.85rem 1.2rem 0.85rem;
}

/* Identitas aplikasi */
.ms-brand {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.3rem 0.2rem 1rem 0.2rem;
}

.ms-brand-mark {
  width: 38px;
  height: 38px;
  flex: none;
  border-radius: 11px;
  background: linear-gradient(135deg, #0D9488, #0B7F76);
  color: #FFFFFF;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.03em;
}

.ms-brand-name {
  color: #FFFFFF !important;
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  line-height: 1.2;
}

.ms-brand-sub {
  color: #B9CBDE !important;
  font-size: 0.73rem;
  line-height: 1.45;
  margin-top: 0.12rem;
}

/* Label grup menu */
.ms-nav-label {
  color: #7C93AC;
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 0.1rem 0 0.4rem 0.35rem;
}

/* Radio diubah menjadi daftar navigasi.
   Struktur DOM (baseweb): label > div(teks) > input > div(mark radio).
   Yang disembunyikan adalah MARK (div terakhir), teks tetap tampil. */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 2px; }

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label {
  width: 100%;
  padding: 9px 12px 9px 40px !important;
  margin: 0 0 2px 0 !important;
  border-radius: 9px;
  position: relative;
  cursor: pointer;
  transition: background 0.15s ease;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"]
  > label > div:last-of-type {
  display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label p,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label span,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label div {
  color: #E8EFF7 !important;
  font-size: 0.87rem;
  font-weight: 500;
  margin: 0;
  line-height: 1.35;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:hover {
  background: rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:hover p,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:hover span,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:hover div {
  color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label::before {
  content: "";
  position: absolute;
  left: 13px;
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  background-color: #8FA6BF;
  -webkit-mask-repeat: no-repeat;
  mask-repeat: no-repeat;
  -webkit-mask-position: center;
  mask-position: center;
  -webkit-mask-size: contain;
  mask-size: contain;
  transition: background-color 0.15s ease;
}

/* Menu aktif */
[data-testid="stSidebar"] [data-testid="stRadioGroup"]
  > label:has(input:checked) {
  background: rgba(13, 148, 136, 0.22);
  box-shadow: inset 3px 0 0 0 #0D9488;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"]
  > label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadioGroup"]
  > label:has(input:checked) span,
[data-testid="stSidebar"] [data-testid="stRadioGroup"]
  > label:has(input:checked) div {
  color: #FFFFFF !important;
  font-weight: 600;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"]
  > label:has(input:checked)::before {
  background-color: #2DD4BF;
}

/* Nota aktif */
.ms-note {
  margin-top: 1rem;
  padding: 0.6rem 0.75rem;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid rgba(255, 255, 255, 0.09);
}

.ms-note span {
  display: block;
  color: #8FA6BF;
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.18rem;
}

.ms-note strong {
  color: #FFFFFF;
  font-size: 0.82rem;
  font-weight: 600;
  word-break: break-word;
}

.ms-note-kosong {
  color: #7C93AC;
  font-size: 0.76rem;
  margin-top: 1rem;
  line-height: 1.5;
}

.ms-side-foot {
  margin-top: 1rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  color: #7C93AC;
  font-size: 0.69rem;
  line-height: 1.5;
}
"""


def _aturan_ikon_nav():
    """Satu aturan CSS per menu, berurutan sesuai daftar HALAMAN."""
    urutan = [
        "bu", "upload", "preproses", "segmen",
        "kenal", "ekstraksi", "evaluasi",
    ]
    baris = []
    for posisi, nama in enumerate(urutan, 1):
        mask = _svg_mask(nama)
        baris.append(
            '[data-testid="stSidebar"] [data-testid="stRadioGroup"] '
            f'> label:nth-of-type({posisi})::before {{'
            f" -webkit-mask-image: {mask}; mask-image: {mask}; }}"
        )
    return "\n".join(baris)



CSS_KOMPONEN = """
/* ---------- Tombol ---------- */
.stButton > button,
.stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid #DCE5EF;
  background: #FFFFFF;
  color: #142D4E;
  font-size: 0.87rem;
  font-weight: 600;
  padding: 0.5rem 1.05rem;
  transition: border-color 0.15s ease, color 0.15s ease,
              background 0.15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: #0D9488;
  color: #0D9488;
  background: #FFFFFF;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background: #0D9488;
  border-color: #0D9488;
  color: #FFFFFF;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background: #0B7F76;
  border-color: #0B7F76;
  color: #FFFFFF;
}

.stButton > button:disabled,
.stDownloadButton > button:disabled {
  background: #F1F5F9;
  border-color: #E2E8F0;
  color: #94A3B8;
}

/* ---------- Input ---------- */
.stTextInput label p,
.stTextArea label p,
.stFileUploader label p {
  font-size: 0.8rem;
  font-weight: 600;
  color: #334155;
}

[data-baseweb="input"],
[data-baseweb="textarea"] {
  border-radius: 10px !important;
  border-color: #DCE5EF !important;
  background: #FFFFFF !important;
}

[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within {
  border-color: #0D9488 !important;
  box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12) !important;
}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
  font-size: 0.88rem;
  color: #142D4E;
}

/* ---------- Upload ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: #FFFFFF;
  border: 1.5px dashed #C9D6E4;
  border-radius: 12px;
  padding: 1.5rem 1.2rem;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #0D9488;
  background: #FAFEFD;
}

/* ---------- Tab ---------- */
[data-baseweb="tab-list"] {
  gap: 0.3rem;
  border-bottom: 1px solid #DCE5EF;
}

[data-baseweb="tab"] {
  height: auto;
  padding: 0.45rem 0.9rem;
  border-radius: 9px 9px 0 0;
  background: transparent;
  font-size: 0.85rem;
  font-weight: 600;
  color: #64748B;
}

[data-baseweb="tab"]:hover { color: #0D9488; }

[data-baseweb="tab"][aria-selected="true"] {
  color: #0D9488;
  background: #E6F4F1;
}

[data-baseweb="tab-highlight"] { background: #0D9488; }
[data-baseweb="tab-border"] { display: none; }

/* ---------- Alert ---------- */
[data-testid="stAlert"] {
  border-radius: 10px;
  border: 1px solid #DCE5EF;
  font-size: 0.85rem;
}

[data-testid="stAlert"] p {
  font-size: 0.85rem;
  line-height: 1.55;
}

/* ---------- Metric ---------- */
[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid #DCE5EF;
  border-radius: 12px;
  padding: 0.85rem 1rem;
}

[data-testid="stMetricLabel"] p {
  font-size: 0.7rem !important;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #64748B !important;
}

[data-testid="stMetricValue"] {
  font-size: 1.5rem;
  font-weight: 700;
  color: #142D4E;
}

/* ---------- Tabel ---------- */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
  border: 1px solid #DCE5EF;
  border-radius: 12px;
  overflow: hidden;
  background: #FFFFFF;
}

/* ---------- Blok kode hasil OCR ---------- */
[data-testid="stCode"] {
  border: 1px solid #DCE5EF;
  border-radius: 10px;
  background: #F8FAFC;
}

[data-testid="stCode"] pre {
  background: transparent;
  font-size: 0.85rem;
}

/* ---------- Expander & divider ---------- */
[data-testid="stExpander"] {
  background: #FFFFFF;
  border: 1px solid #DCE5EF;
  border-radius: 12px;
}

[data-testid="stExpander"] summary p {
  font-size: 0.87rem;
  font-weight: 600;
  color: #142D4E;
}

[data-testid="stMainBlockContainer"] hr {
  margin: 0.9rem 0;
  border-color: #E2E8F0;
  opacity: 1;
}

/* ---------- Diagram alur ---------- */
.ms-flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
  padding: 1rem 1.1rem;
  background: #FFFFFF;
  border: 1px solid #DCE5EF;
  border-radius: 14px;
}

.ms-flow-step {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.4rem 0.7rem;
  border-radius: 9px;
  background: #F4F7FB;
  border: 1px solid #E4EBF3;
  font-size: 0.78rem;
  font-weight: 600;
  color: #142D4E;
}

.ms-flow-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #0D9488;
  color: #FFFFFF;
  font-size: 0.66rem;
  font-weight: 700;
}

.ms-flow-arrow { color: #94A3B8; font-size: 0.9rem; }

/* ---------- Chip status ---------- */
.ms-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.1rem 0 0.2rem 0;
}

.ms-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.22rem 0.6rem;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  border: 1px solid #DCE5EF;
  background: #FFFFFF;
  color: #334155;
}

.ms-chip i {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  display: inline-block;
}

.ms-chip-benar i { background: #0D9488; }
.ms-chip-salah i { background: #DC2626; }
.ms-chip-belum i { background: #94A3B8; }
"""



CSS = CSS_DASAR + _CSS_SIDEBAR + _aturan_ikon_nav() + CSS_KOMPONEN


def pasang_tema():
    """Sisipkan CSS tema. Dipanggil sekali di awal setiap render."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def kepala_halaman(judul, deskripsi):
    """
    Header halaman: judul lalu deskripsi singkat.

    Memakai st.header dan st.caption native agar struktur halaman tetap
    terbaca (termasuk oleh pengujian), dengan CSS yang merapikan tampilan.
    """
    st.header(judul)
    if deskripsi:
        st.caption(deskripsi)


def seksi(judul, deskripsi=None):
    """Sub-judul bagian beserta keterangan singkat opsional."""
    st.subheader(judul)
    if deskripsi:
        st.caption(deskripsi)


def sidebar_brand():
    """Identitas aplikasi pada bagian atas sidebar."""
    st.markdown(
        '<div class="ms-brand">'
        '<div class="ms-brand-mark">MS</div>'
        "<div>"
        '<div class="ms-brand-name">MERRI-STOCK</div>'
        '<div class="ms-brand-sub">CV Merriena Jaya<br>'
        "Pengenalan Teks Nota Transaksi</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def sidebar_label_menu(teks="Menu"):
    """Label kecil di atas daftar menu."""
    st.markdown(
        f'<div class="ms-nav-label">{html.escape(teks)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_status(nama_file=None):
    """Keterangan nota yang sedang aktif pada sidebar."""
    if nama_file:
        st.markdown(
            '<div class="ms-note"><span>Nota aktif</span>'
            f"<strong>{html.escape(str(nama_file))}</strong></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ms-note-kosong">Belum ada nota yang diunggah.</div>',
            unsafe_allow_html=True,
        )


def alur(tahapan):
    """Diagram alur horizontal dari daftar nama tahap (statis, dekoratif)."""
    potong = []
    for posisi, nama in enumerate(tahapan, 1):
        if posisi > 1:
            potong.append('<span class="ms-flow-arrow">&rarr;</span>')
        potong.append(
            '<span class="ms-flow-step">'
            f'<span class="ms-flow-num">{posisi}</span>'
            f"{html.escape(str(nama))}</span>"
        )
    st.markdown(
        '<div class="ms-flow">' + "".join(potong) + "</div>",
        unsafe_allow_html=True,
    )


def legenda_status():
    """Legenda warna kolom Status pada halaman Evaluasi."""
    st.markdown(
        '<div class="ms-chip-row">'
        '<span class="ms-chip ms-chip-benar"><i></i>Benar</span>'
        '<span class="ms-chip ms-chip-salah"><i></i>Salah</span>'
        '<span class="ms-chip ms-chip-belum"><i></i>Belum diisi</span>'
        "</div>",
        unsafe_allow_html=True,
    )
