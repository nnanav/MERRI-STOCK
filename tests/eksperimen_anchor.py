"""Eksperimen offline header geometris pada kanvas produksi A."""
import json
import sys
from pathlib import Path
from types import FunctionType

import cv2
import numpy as np

from eksperimen_crop_kertas import ROOT, app, simpan, ukur
from processing.deteksi_field import _kotak_gabung
from processing.preprocessing_nota import crop_dan_luruskan

OUT = ROOT / "output" / "eksperimen_anchor"
REFERENSI = Path(__file__).with_name("acuan_anchor.json")


def header_struktur(kanvas, kandidat):
    abu = cv2.cvtColor(kanvas, cv2.COLOR_BGR2GRAY)
    h, w = abu.shape
    tepi = cv2.Canny(abu, 45, 130)
    garis = cv2.HoughLinesP(tepi, 1, np.pi / 720, 90,
                            minLineLength=round(.35*w), maxLineGap=round(.04*w))
    if garis is None:
        return None, "garis kurang"
    garis = sorted(
        ((x, y, u, v) for x, y, u, v in garis[:, 0]
         if u > x and abs(v-y) < .08*(u-x)),
        key=lambda g: g[1]+g[3],
    )
    biner = cv2.adaptiveThreshold(abu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 13)
    stats = cv2.connectedComponentsWithStats(biner)[2][1:]
    solusi = []
    for lx, ly, rx, ry in garis:
        panjang = rx-lx
        kemiringan = (ry-ly)/panjang
        def posisi_y(x):
            return ly + (x-lx)*kemiringan

        kop = any(
            min(u,rx)-max(x,lx) > .67*panjang
            and .022*panjang <= posisi_y(x)-y <= .075*panjang
            and abs((v-y)/(u-x)-kemiringan) < .025
            for x,y,u,v in garis
        )
        if not kop:
            continue
        kecil = [
            (int(x), int(y), int(ww), int(hh)) for x,y,ww,hh,luas in stats
            if lx+.04*panjang <= x <= lx+.28*panjang
            and .012*panjang <= y-posisi_y(x) <= .15*panjang
            and 1 <= ww <= .008*panjang and 1 <= hh <= .008*panjang
            and luas >= 2
        ]
        titik_dua = sorted(
            (x, (y+v+hh)/2) for x,y,ww,hh in kecil for u,v,_,_ in kecil
            if abs(x-u) <= 2 and 2 <= v-y <= max(4,.011*panjang)
        )
        for x,y in titik_dua:
            deret = sorted(v-y for u,v in titik_dua if abs(u-x)<=3 and v-y>7)
            if len(deret)<2:
                continue
            jarak = deret[0]
            if not .012*panjang <= jarak <= .036*panjang:
                continue
            if not any(abs(d-2*jarak) <= .23*jarak for d in deret[1:]):
                continue
            nilai = []
            for pusat in (y,y+jarak):
                calon = [
                    k for k in kandidat
                    if x-.80*jarak <= k["x1"] <= x+1.9*jarak
                    and k["x2"] > x+1.15*jarak
                    and k["w"] >= 1.15*jarak
                    and abs(k["cy"]-pusat-kemiringan*(k["cx"]-x)) <= .46*jarak
                    and k["h"] <= 1.55*jarak
                ]
                if not calon:
                    break
                kotak = min(calon,key=lambda k: (abs(k["x1"]-x), -k["w"]))
                nilai.append(kotak)
            if len(nilai)==2 and nilai[0] is not nilai[1]:
                solusi.append((y,x,nilai))
    if not solusi:
        return None, "pola titik dua atau nilai kurang"
    y,x,nilai = min(solusi,key=lambda s: (s[0],s[1]))
    if any(abs(u-x)>8 or abs(v-y)>8 for v,u,_ in solusi):
        return None, "anchor ganda"
    hasil = tuple(_kotak_gabung([{
        "x1": max(k["x1"], x+4), "x2": k["x2"],
        "y1": k["y1"], "y2": k["y2"],
    }]) for k in nilai)
    return hasil, "anchor"


def deteksi_baru(foto, aktif):
    alasan = None

    def header(kandidat, lebar, tinggi):
        nonlocal alasan
        nilai, alasan = header_struktur(aktif["kecil"], kandidat)
        return nilai if nilai is not None else app.deteksi_penerima_proyek(
            kandidat, lebar, tinggi)

    deteksi = FunctionType(app.jalankan_deteksi.__code__, {
        **app.jalankan_deteksi.__globals__,
        "preprocess_nota": lambda _: aktif["preproses"],
        "deteksi_penerima_proyek": header,
    })
    return deteksi(foto), alasan


def transformasi_ke_kanvas(foto, det):
    dasar = crop_dan_luruskan(foto)
    matriks = np.eye(3, dtype=np.float64)
    if dasar["status"] == "koreksi_perspektif":
        x,y,u,v = dasar["bbox"]
        tujuan = np.float32([[x,y],[u-1,y],[u-1,v-1],[x,v-1]])
        matriks = cv2.getPerspectiveTransform(dasar["kotak"], tujuan)
    h,w = foto.shape[:2]
    hh,ww = det["kecil"].shape[:2]
    return np.diag([ww/w,hh/h,1]) @ matriks


def variasi(foto):
    h,w = foto.shape[:2]
    ukuran = (round(.78*w), round(.78*h))
    kecil = cv2.resize(foto, ukuran, interpolation=cv2.INTER_AREA)
    warna = (191,187,182)
    a,b = w//7,h//15
    yield "geser_skala", cv2.copyMakeBorder(kecil,b,h//8,a,w//35,
        cv2.BORDER_CONSTANT,value=warna), np.array([[ukuran[0]/w,0,a],[0,ukuran[1]/h,b],[0,0,1]])
    rotasi = cv2.getRotationMatrix2D((w/2,h/2),4,.9)
    yield "miring", cv2.warpAffine(foto,rotasi,(w,h),borderValue=warna), np.vstack([rotasi,[0,0,1]])
    yield "kontras_rendah", cv2.convertScaleAbs(foto,alpha=.56,beta=90), np.eye(3)


def proyeksi(bb, matriks):
    x,y,u,v = bb
    titik = cv2.perspectiveTransform(np.float32([[[x,y],[u,y],[u,v],[x,v]]]), matriks)[0]
    mn = titik.min(axis=0)
    mx = titik.max(axis=0)
    return [float(k) for k in (*mn,*mx)]


def iou(a,b):
    if b is None:
        return 0.0
    irisan = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    luas = lambda x: (x[2]-x[0])*(x[3]-x[1])
    return irisan/(luas(a)+luas(b)-irisan)


def main():
    geometri = "--geometri" in sys.argv[1:]
    tujuan = OUT / ("geometri_v2.json" if geometri else "hasil_v2.json")
    cv2.setNumThreads(2)
    catatan = json.loads(REFERENSI.read_text(encoding="utf-8-sig"))
    hasil = []
    semua = sorted(p for folder in ("data_nota_asli","data_uji")
                  for p in (ROOT/folder).glob("*.jp*g"))
    for p in semua:
        foto = app.baca_gambar(p)
        asli = app.jalankan_deteksi(foto)
        gt = catatan.get(p.stem)
        sampel = [("asli",foto,np.eye(3))]
        if gt:
            sampel.extend(variasi(foto))
            dari_kanvas_asli = np.linalg.inv(transformasi_ke_kanvas(foto,asli))
        for nama,citra,warp in sampel:
            a = asli if nama=="asli" else app.jalankan_deteksi(citra)
            b, alasan = deteksi_baru(citra,a)
            assert np.array_equal(a["kecil"],b["kecil"])
            assert [(f["id"],f["bbox"]) for f in a["fields"] if f["jenis"] not in ("proyek","penerima")] == [
                (f["id"],f["bbox"]) for f in b["fields"] if f["jenis"] not in ("proyek","penerima")]
            baris = dict(nota=p.stem,kondisi=nama,alasan=alasan)
            for key,hasil_det in (("A",a),("B",b)):
                fields = {f["id"]: f["bbox"] for f in hasil_det["fields"]}
                stat = dict(produk=len(hasil_det["produk"]),bbox=fields)
                if gt:
                    if nama=="asli":
                        matriks = np.eye(3)
                    else:
                        matriks = transformasi_ke_kanvas(citra,a) @ warp @ dari_kanvas_asli
                    acuan = {k:proyeksi(v[0],matriks) for k,v in gt.items()}
                    stat["iou"] = {k:iou(bb,fields.get(k)) for k,bb in acuan.items()}
                    if not geometri:
                        pred = {k:v["tanpa"].get("teks","") for k,v in
                                app.jalankan_ocr_semua(hasil_det,False).items()}
                        stat["ocr"] = ukur({k:v[1] for k,v in gt.items()},pred)
                        stat["prediksi"] = pred
                    folder = OUT/"visual"/p.stem/nama
                    simpan(folder,"kanvas_aktual",hasil_det["kecil"])
                    overlay = app.gambar_kotak_field(hasil_det["kecil"],hasil_det["fields"])
                    for bb in acuan.values():
                        x,y,u,v = map(round,bb)
                        cv2.rectangle(overlay,(x,y),(u,v),(200,65,200),1)
                    simpan(folder,f"{key}_overlay_ilustrasi",overlay)
                baris[key] = stat
            hasil.append(baris)
        tujuan.write_text(json.dumps(hasil,indent=2,default=lambda v:v.item()),encoding="utf-8")
        print(p.stem,flush=True)


if __name__ == "__main__":
    main()
