# Import Library
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import io
import os
import string
import unicodedata
import nltk
import gdown
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ---- Bagian 1 : Konfigurasi Halaman ----
st.set_page_config(
    page_title="SentiAI - Analisis Sentimen AI Apps",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Bagian 2 : Membuat tampilan CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:#0a0e1a; --surface:#111827; --surface2:#1c2436;
    --accent:#6366f1; --accent2:#818cf8;
    --text:#e2e8f0; --muted:#64748b; --border:#1e293b;
}
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg) !important; font-family: 'Inter', sans-serif; color: var(--text); }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.main-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 0.25rem; }
.subtitle { color: var(--muted); font-size: 1rem; margin-bottom: 2rem; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.25rem 1.5rem; text-align: center; transition: transform 0.2s, box-shadow 0.2s; }
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(99,102,241,0.15); }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: var(--accent2); }
.metric-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }
.sentiment-badge { display: inline-block; padding: 0.4rem 1.1rem; border-radius: 999px; font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.03em; }
.badge-positif { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b98144; }
.badge-negatif { background: rgba(244,63,94,0.15); color: #f43f5e; border: 1px solid #f43f5e44; }
.badge-netral  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b44; }
.result-card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 1.75rem; margin: 1rem 0; }
.conf-bar-bg { background: var(--surface2); border-radius: 999px; height: 10px; overflow: hidden; margin: 4px 0 12px; }
.conf-bar-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }
.section-header { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: var(--text); border-left: 4px solid var(--accent); padding-left: 0.8rem; margin: 1.5rem 0 1rem; }
.stTextArea textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--text) !important; font-family: 'Inter', sans-serif !important; }
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important; }
.stButton > button { background: linear-gradient(135deg, #6366f1, #818cf8) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; padding: 0.65rem 2rem !important; transition: all 0.2s !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important; }
.stSelectbox > div > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }
div[data-testid="stTabs"] button { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: var(--muted) !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent2) !important; border-bottom-color: var(--accent) !important; }
.stFileUploader { background: var(--surface2) !important; border: 2px dashed var(--border) !important; border-radius: 16px !important; }
.upload-hint { background: var(--surface2); border: 1px dashed var(--border); border-radius: 12px; padding: 0.8rem 1rem; font-size: 0.82rem; color: var(--muted); margin-top: 0.5rem; }
.history-item { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.88rem; }
.info-box { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25); border-radius: 12px; padding: 0.9rem 1.1rem; font-size: 0.88rem; color: #a5b4fc; margin: 0.75rem 0; }
div[data-testid="stMarkdownContainer"] p { color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ---- Bagian 3 : Konfigurasi konstanta aplikasi ----
LABEL_MAP  = {0: "Negatif", 1: "Netral", 2: "Positif"}
COLORS     = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#f43f5e"}
EMOJI_MAP  = {"Positif": "😊", "Netral": "😐", "Negatif": "😠"}
BADGE_CLASS = {"Positif": "badge-positif", "Netral": "badge-netral", "Negatif": "badge-negatif"}

AI_APPS = ["ChatGPT", "Google Gemini", "Grok", "Microsoft Copilot", "Meta AI"]

FILE_CSV = {
    "ChatGPT":           "review_chatgpt5k.csv",
    "Google Gemini":     "review_gemini5k.csv",
    "Grok":              "review_grok5k.csv",
    "Microsoft Copilot": "review_copilot5k.csv",
    "Meta AI":           "review_meta5k.csv",
}

# Kamus Kata Kunci Aspek
ASPECT_KEYWORDS = {
    "Performa": [
        "cepat", "lemot", "lambat", "loading", "delay", "lag", "ngelag",
        "lelet", "respons", "respon", "buffering", "hang", "freeze",
        "crash", "error", "bug", "stuck", "berat", "ringan",
        "tidak responsif", "loading lama", "sering error",
    ],
    "Akurasi": [
        "akurat", "tepat", "benar", "sesuai", "relevan",
        "tidak akurat", "kurang tepat", "tidak sesuai", "ngawur",
        "salah", "melenceng", "tidak relevan", "jawaban benar",
        "jawaban salah", "hasil tepat", "hasil tidak sesuai",
    ],
    "UI/UX": [
        "tampilan", "interface", "ui", "ux", "desain",
        "mudah", "gampang", "user friendly", "intuitif",
        "sulit", "ribet", "membingungkan", "tidak nyaman",
        "navigasi", "menu", "layout", "fitur mudah digunakan",
        "kurang user friendly",
    ],
    "Fitur": [
        "fitur", "fungsi", "tool", "tools", "fitur baru",
        "gambar", "image", "foto", "ilustrasi",
        "upload", "scan", "kamera", "camera",
        "voice", "suara", "translate", "terjemahan",
        "coding", "kode", "program", "programming",
        "pdf", "dokumen", "file",
    ],
    "Harga": [
        "mahal", "murah", "harga", "biaya",
        "langganan", "berlangganan", "subscription",
        "premium", "pro", "plus",
        "berbayar", "bayar", "gratis", "trial",
    ],
}

# Kamus Kata Kunci Use Case
USECASE_KEYWORDS = {
    "Belajar": [
        "belajar", "materi", "pelajaran", "penjelasan",
        "contoh soal", "latihan", "tugas",
        "ringkasan", "resume", "skripsi",
        "makalah", "penelitian", "kuliah",
    ],
    "Coding": [
        "coding", "program", "ngoding", "debug",
        "python", "java", "html", "sql",
        "koding", "algoritma", "machine learning",
        "code", "developer",
    ],
    "Konten": [
        "konten", "caption", "instagram", "tiktok",
        "youtube", "script", "story",
        "copywriting", "marketing",
        "iklan", "promosi",
        "artikel", "blog",
    ],
    "Pekerjaan": [
        "kerja", "pekerjaan", "kantor",
        "laporan", "proposal", "email",
        "dokumen", "presentasi",
        "produktivitas", "administrasi",
    ],
}

# Kata yang dihapus sebelum membuat wordcloud
STOP_WORDS = set([
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "ada",
    "tidak", "saya", "aku", "kamu", "dia", "mereka", "kami", "kita", "akan",
    "sudah", "bisa", "juga", "lebih", "sangat", "banget", "paling", "bagi",
    "pada", "atau", "jika", "kalau", "karena", "tapi", "tetapi", "namun",
    "app", "aplikasi", "nya", "ya", "yg", "yah", "deh", "sih", "nih", "lah",
    "ai", "jadi", "buat", "pengguna", "google",
])


# ---- Bagian 4 : Fungsi Model IndoBERT ----
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    return {
        "text": text,
        "label": LABEL_MAP[pred_class],
        "confidence": {
            "Negatif": float(probs[0][0]),
            "Netral":  float(probs[0][1]),
            "Positif": float(probs[0][2]),
        }
    }

def predict_banyak_teks(daftar_teks, tokenizer, model, progress_bar=None):
    hasil_semua = []

    for i, teks in enumerate(daftar_teks):
        hasil = predict_sentiment(str(teks), tokenizer, model)
        label = hasil["label"]

        hasil_semua.append({
            "text": teks,
            "label": label,
            "confidence": hasil["confidence"][label],
        })

        if progress_bar is not None:
            progress_bar.progress((i + 1) / len(daftar_teks))

    return hasil_semua


# ---- Bagian 5 : Pipeline Preprocessing ---- 

@st.cache_resource(show_spinner=False)
def unduh_nltk():
    nltk.download("punkt",      quiet=True)
    nltk.download("punkt_tab",  quiet=True)
    nltk.download("stopwords",  quiet=True)

unduh_nltk()

# Kamus Slangwords
SLANGWORDS = {
    "gk":"tidak","ga":"tidak","gak":"tidak","nggak":"tidak","tdk":"tidak",
    "ngga":"tidak","enggak":"tidak","kagak":"tidak",
    "bgt":"banget","bngt":"banget","bgtt":"banget",
    "parahh":"sangat","parahhh":"sangat","bangettt":"banget","super":"sangat",
    "wkwk":"haha","wkwwk":"haha","wk":"haha","wkwkwk":"haha","wkwkwkwk":"haha",
    "haha":"haha","hehe":"haha","hihi":"haha","awokwok":"haha","kwkw":"haha",
    "lol":"haha","lmao":"haha",
    "makasi":"terima kasih","mksh":"terima kasih","thanks":"terima kasih",
    "thx":"terima kasih","ty":"terima kasih","bener":"benar","bner":"benar",
    "salahh":"salah","aja":"saja","doang":"saja","aj":"saja",
    "yg":"yang","dgn":"dengan","krn":"karena","karna":"karena",
    "udh":"sudah","udah":"sudah","blm":"belum","yng":"yang",
    "trs":"terus","trus":"terus","sm":"sama","dr":"dari","kece":"bagus",
    "anjay":"kagum","anjir":"kagum","anjirr":"kagum","anjayyy":"kagum","anjy":"kagum",
    "buset":"kagum","gila":"sangat","gilaa":"sangat","gokil":"sangat bagus",
    "sumpah":"sangat","demi":"sangat","asli":"sangat","real":"nyata",
    "wih":"kagum","wah":"kagum","woah":"kagum","ih":"tidak suka","ihh":"tidak suka",
    "yah":"kecewa","yahh":"kecewa","fix":"pasti","auto":"langsung",
    "best":"terbaik","receh":"tidak penting","random":"acak","ad":"ada","ak":"aku",
    "overrated":"terlalu dilebihkan","underrated":"kurang dihargai",
    "baikk":"baik","baikkk":"baik","baguus":"bagus","baguuus":"bagus","bagusss":"bagus",
    "kerenn":"keren","kerennn":"keren","mantapp":"mantap","mantappp":"mantap",
    "jelekk":"jelek","jelekkk":"jelek","burukk":"buruk","burukkk":"buruk",
    "lemott":"lemot","lemottt":"lemot","lambatt":"lambat","lambattt":"lambat",
    "cepet":"cepat","cepettt":"cepat","cepattt":"cepat",
    "gtw":"tidak tahu","gatau":"tidak tahu","ga tau":"tidak tahu",
    "gk tau":"tidak tahu","gak tau":"tidak tahu","nggak tau":"tidak tahu","tau":"tahu",
    "ai":"artificial intelligence","gpt":"chatgpt","chatgpt":"chatgpt",
    "copilot":"copilot","gemini":"gemini","grok":"grok",
    "metaai":"meta ai","openai":"openai",
    "lemot":"lambat","lelet":"lambat","lag":"lambat","ngelag":"lambat",
    "delay":"lambat","error":"error","eror":"error","err":"error",
    "bug":"bug","buggy":"bug","crash":"crash","forceclose":"crash","fc":"crash",
    "ngaco":"tidak akurat","ngawur":"tidak akurat","ngasal":"tidak akurat","asal":"tidak akurat",
    "halu":"tidak masuk akal","kurang":"kurang baik",
    "akurat":"akurat","detail":"detail","jelas":"jelas","lengkap":"lengkap","informatif":"informatif",
    "mantul":"mantap","mantap":"bagus","keren":"bagus","bagus":"bagus",
    "good":"bagus","great":"bagus","nice":"bagus",
    "recommended":"direkomendasikan","rekomended":"direkomendasikan",
    "worthit":"bernilai","worth":"bernilai",
    "jelek":"buruk","jelekk":"buruk","burik":"buruk","ampas":"buruk","parah":"buruk",
    "zonk":"buruk","fail":"gagal","gagal":"gagal","payah":"buruk",
    "download":"unduh","install":"pasang","update":"perbarui","upgrade":"perbarui",
    "login":"masuk","logout":"keluar","signin":"masuk","signup":"daftar",
    "jawabannya":"jawaban","respon":"jawaban","reply":"jawaban","jawab":"jawaban",
    "prompt":"pertanyaan","ngoding":"coding","coding":"coding","kode":"coding",
    "slow":"lambat","fast":"cepat","wrong":"salah","right":"benar",
    "helpful":"membantu","useless":"tidak berguna","really":"sangat","image":"gambar",
    "realy":"sangat","gud":"bagus","god":"bagus",
}

# Daftar tambahan stopwords 
@st.cache_resource(show_spinner=False)
def buat_stopwords():
    list_stop = set(stopwords.words("indonesian"))
    list_stop.update(set(stopwords.words("english")))
    list_stop.update([
        "iya","yaa","ya","yak","yap","aja","doang","dong","nih","deh","kok",
        "loh","lah","sih","kah","tuh","nya","pun","kan","wkwk","wkwwk","wk",
        "haha","hehe","hihi","huhu","lol","lmao","bro","sis","gan","min",
        "dll","dsb","etc","and","or","the","apk","aplikasi","app","ku","di",
        "wow","woww","wowww","nihh","dongg","dehh","sihh","yakkk","yaaah",
    ])
    return list_stop

STOPWORDS_SET = buat_stopwords()

# Kamus kata-kata yang dipertahankan
KATA_PENTING = {
    "baik","bagus","buruk","jelek","keren","mantap",
    "puas","kecewa","recommended","recommend","best",
    "parah","lumayan","biasa","suka","tidak_suka",
}


# ---- Fungsi Preprocessing ----
# Fungsi Cleaning
def pp_bersihkan(tekss):
    tekss = unicodedata.normalize("NFKD", str(tekss))
    tekss = tekss.encode("ascii", "ignore").decode("utf-8")
    tekss = re.sub(r"http\S+", "", tekss)          
    tekss = re.sub(r"[0-9]+", "", tekss)             
    tekss = re.sub(r"[^\w\s]", "", tekss)         
    tekss = tekss.replace("\n", "")                
    tekss = tekss.translate(str.maketrans("", "", string.punctuation))
    tekss = tekss.strip()
    return tekss

# Fungsi Lowercase
def pp_kecilkan(tekss):
    return tekss.lower()

# Fungsi Normalisasi Teks
def pp_normal_karakter(tekss):
    return re.sub(r"(.)\1{1,}", r"\1", tekss)

# Fungsi Normalisasi slang
def pp_slang(tekss):
    kata_kata = tekss.split()
    hasil = []
    for kata in kata_kata:
        if kata.lower() in SLANGWORDS:
            hasil.append(SLANGWORDS[kata.lower()])
        else:
            hasil.append(kata)
    return " ".join(hasil)

# Fungsi Tokenisasi 
def pp_tokenisasi(tekss):
    return word_tokenize(tekss)

# Fungsi Stopword
def pp_stopword(daftar_kata):
    hasil = []
    for kata in daftar_kata:
        if (kata not in STOPWORDS_SET) or (kata in KATA_PENTING):
            hasil.append(kata)
    return hasil

# Fungsi Menggabungkan Token
def pp_gabung(daftar_kata):
    return " ".join(daftar_kata)

# Fungsi Preprocessing Final
def jalankan_preprocessing(text):
    teks = pp_bersihkan(text)
    teks = pp_kecilkan(teks)
    teks = pp_normal_karakter(teks)
    teks = pp_slang(teks)
    token = pp_tokenisasi(teks)
    token = pp_stopword(token)
    teks_fix = pp_gabung(token)
    return teks_fix


# ---- Analisis Aspek dan Use Case ----
# Fungsi Deteksi Aspek
def deteksi_aspek(teks_fix):
    hasil = []
    teks_lower = str(teks_fix).lower()
    for nama_aspek, daftar_keyword in ASPECT_KEYWORDS.items():
        for keyword in daftar_keyword:
            if keyword in teks_lower:
                hasil.append(nama_aspek)
                break  
    return hasil if hasil else ["lainnya"]

# FUngsi Deteksi Use Case
def deteksi_usecase(teks_fix):
    hasil = []
    teks_lower = str(teks_fix).lower()
    for nama_uc, daftar_keyword in USECASE_KEYWORDS.items():
        for keyword in daftar_keyword:
            if keyword in teks_lower:
                hasil.append(nama_uc)
                break 
    return hasil if hasil else ["lainnya"]

# Fungsi Cleaning Wordclooud
def bersihkan_teks(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    kata_bersih = [k for k in text.split() if k not in STOP_WORDS and len(k) > 2]
    return " ".join(kata_bersih)

# Fungsi Labeling
def petakan_score_ke_sentimen(score):
    if score <= 2:
        return "Negatif"
    elif score == 3:
        return "Netral"
    else:
        return "Positif"

# Fungsi Membaca Data
@st.cache_data(show_spinner=False)
def muat_semua_dataset():
    data_semua_app = {}

    for nama_app, nama_file in FILE_CSV.items():
        path_file = nama_file

        if os.path.exists(path_file):
            df = pd.read_csv(path_file)

            if "content" in df.columns and "score" in df.columns:
                df = df.dropna(subset=["content"])
                df["teks_fix"] = df["content"].apply(jalankan_preprocessing)
                df["label"] = df["score"].apply(petakan_score_ke_sentimen)

                # Memilih Kolom yang Diperlukan
                kolom_yang_dipakai = ["content", "teks_fix", "score", "at", "label"]
                kolom_tersedia = [k for k in kolom_yang_dipakai if k in df.columns]
                df = df[kolom_tersedia]

                data_semua_app[nama_app] = df
            else:
                st.warning(f"⚠️ File {nama_file} tidak memiliki kolom 'content' atau 'score'.")
        else:
            st.warning(f"⚠️ File tidak ditemukan: {nama_file}")

    return data_semua_app


# ---- Bagian 6 : Fungsi Pembuat Grafik ----
# Fungsi mengatur tema
def atur_tema_grafik(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

# Fungsi membuat kartu metrik
def kartu_metrik(kolom, nilai, label, warna):
    with kolom:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:{warna};'>{nilai}</div>
          <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

# Fungsi Grafik Donut  - Distribusi sentimen per dataset
def grafik_donut(jumlah_per_label, judul):
    nama_label = list(jumlah_per_label.keys())
    jumlah    = list(jumlah_per_label.values())
    total     = sum(jumlah)

    fig = px.pie(
        names=nama_label, values=jumlah, hole=0.6,
        color=nama_label, color_discrete_map=COLORS, title=judul,
    )
    fig.add_annotation(text=f"Total<br><b>{total:,}</b>", x=0.5, y=0.5, showarrow=False, font_size=14)
    fig = atur_tema_grafik(fig)
    fig.update_layout(height=320)
    return fig

# Fungsi menghitung persentase sentimen
def hitung_persen_sentimen(data_semua_app):
    baris_data = []

    for nama_app, df in data_semua_app.items():
        jumlah_per_label = df["label"].value_counts()
        total = len(df)

        for sentimen in ["Positif", "Netral", "Negatif"]:
            jumlah = jumlah_per_label.get(sentimen, 0)
            persen = round(jumlah / total * 100, 1)
            baris_data.append({"App": nama_app, "Sentimen": sentimen, "Persen": persen})

    return pd.DataFrame(baris_data)

# Fungsi Bar Chart - Perbandingan distribusi sentimen 5 aplikasi AI
def grafik_perbandingan_app(data_semua_app):
    df_plot = hitung_persen_sentimen(data_semua_app)

    fig = px.bar(
        df_plot, x="App", y="Persen", color="Sentimen",
        color_discrete_map=COLORS, barmode="group", text="Persen",
        title="Perbandingan Distribusi Sentimen - 5 Aplikasi AI",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig = atur_tema_grafik(fig)
    fig.update_layout(height=420, yaxis_title="Persentase (%)", xaxis_title="")
    return fig

# Fungsi Heatmap
def grafik_heatmap_app(data_semua_app):
    df_plot  = hitung_persen_sentimen(data_semua_app)
    df_pivot = df_plot.pivot(index="App", columns="Sentimen", values="Persen")
    df_pivot = df_pivot[["Positif", "Netral", "Negatif"]]

    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns.tolist(),
        y=df_pivot.index.tolist(),
        colorscale="Viridis",
        text=df_pivot.values,
        texttemplate="%{text}%",
    ))
    fig.update_layout(title="Heatmap Sentimen Antar Aplikasi AI")
    fig = atur_tema_grafik(fig)
    fig.update_layout(height=400)
    return fig

# Fungsi Bar Chart - Distribusi dataset per dataset
def grafik_distribusi_rating(df):
    jumlah_per_rating = df["score"].value_counts().sort_index()

    fig = px.bar(
        x=[f"★{i}" for i in jumlah_per_rating.index],
        y=jumlah_per_rating.values,
        title="Distribusi Rating Bintang",
        labels={"x": "Rating", "y": "Jumlah Ulasan"},
    )
    fig.update_traces(
        marker_color=["#f43f5e", "#fb923c", "#f59e0b", "#34d399", "#10b981"],
        text=jumlah_per_rating.values,
        textposition="outside",
    )
    fig = atur_tema_grafik(fig)
    fig.update_layout(height=320)
    return fig

# Fungsi Line Chart - Tren waktu jumlah sentimen
def grafik_tren_waktu(df):
    if "at" not in df.columns:
        return None
    try:
        df = df.copy()
        df["bulan"] = pd.to_datetime(df["at"]).dt.to_period("M").astype(str)
        df_hitung = df.groupby(["bulan", "label"]).size().reset_index(name="jumlah")

        fig = px.line(
            df_hitung, x="bulan", y="jumlah", color="label",
            color_discrete_map=COLORS, markers=True,
            title="Tren Sentimen dari Waktu ke Waktu",
            labels={"bulan": "Bulan", "jumlah": "Jumlah Ulasan", "label": "Sentimen"},
        )
        fig = atur_tema_grafik(fig)
        fig.update_layout(height=360, xaxis_title="")
        return fig
    except Exception:
        return None

# Fungsi persiapan data kategori
def siapkan_data_exploded(df, kolom_kategori, fungsi_deteksi):
    df = df.copy()
    kolom_teks = "teks_fix" if "teks_fix" in df.columns else "content"

    df[kolom_kategori] = df[kolom_teks].apply(fungsi_deteksi)
    
    df_exploded = df.explode(kolom_kategori)
    df_exploded = df_exploded[df_exploded[kolom_kategori] != "lainnya"]

    return df_exploded

# Fungsi menghitung persentase kategori
def hitung_persen_kategori(df_exploded, kolom_kategori):
    df_exploded = df_exploded.reset_index(drop=True)
    tabel_persen = pd.crosstab(
        df_exploded[kolom_kategori],
        df_exploded["label"],
        normalize="index",
    ) * 100

    urutan_label = [l for l in ["Negatif", "Netral", "Positif"] if l in tabel_persen.columns]
    tabel_persen = tabel_persen[urutan_label]

    return tabel_persen

# Fungsi Bar Chart - Persentase aspek dan use case
def grafik_persen_stacked(tabel_persen, urutan_kategori, judul, label_x):
    tabel_persen = tabel_persen.reindex(
        [k for k in urutan_kategori if k in tabel_persen.index]
    )

    fig = go.Figure()
    for sentimen in ["Negatif", "Netral", "Positif"]:
        if sentimen not in tabel_persen.columns:
            continue
        fig.add_trace(go.Bar(
            name=sentimen,
            x=tabel_persen.index.tolist(),
            y=tabel_persen[sentimen].round(1).tolist(),
            marker_color=COLORS[sentimen],
            text=tabel_persen[sentimen].round(1).astype(str) + "%",
            textposition="inside",
        ))

    fig.update_layout(
        title=judul, barmode="stack",
        xaxis_title=label_x, yaxis_title="Persentase Sentimen (%)",
        yaxis_range=[0, 100],
    )
    fig = atur_tema_grafik(fig)
    fig.update_layout(height=400)
    return fig

# Fungsi Bar Horizontal - Aspek & Use Case
def grafik_jumlah_horizontal(seri_jumlah, urutan_kategori, judul, label_y):
    seri_jumlah = seri_jumlah.reindex(
        [k for k in urutan_kategori if k in seri_jumlah.index]
    ).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=seri_jumlah.values,
        y=seri_jumlah.index.tolist(),
        orientation="h",
        marker_color="#6366f1",
        text=seri_jumlah.values,
        textposition="outside",
    ))
    fig.update_layout(
        title=judul,
        xaxis_title="Jumlah Ulasan", yaxis_title=label_y,
    )
    fig = atur_tema_grafik(fig)
    fig.update_layout(height=320)
    return fig

# Tabel ringkasan persentase sentimen per kategori
def tabel_ringkasan_persen(tabel_persen, urutan_kategori, nama_kolom_kategori):
    tabel = tabel_persen.reindex(
        [k for k in urutan_kategori if k in tabel_persen.index]
    ).round(1)

    tabel = tabel.reset_index()
    tabel = tabel.rename(columns={
        tabel.columns[0]: nama_kolom_kategori,
        "Negatif": "Negatif (%)",
        "Netral":  "Netral (%)",
        "Positif": "Positif (%)",
    })
    return tabel


# ---- ASPEK : 2 grafik + 1 tabel ----
# Fungsi distribusi sentimen aspek
def grafik_persen_sentimen_aspek(df):
    df_exploded = siapkan_data_exploded(df, "aspek", deteksi_aspek)

    if df_exploded.empty:
        fig = go.Figure()
        fig.update_layout(title="Tidak ada data aspek yang terdeteksi")
        return atur_tema_grafik(fig)

    tabel_persen = hitung_persen_kategori(df_exploded, "aspek")
    urutan_aspek = list(ASPECT_KEYWORDS.keys()) 

    return grafik_persen_stacked(
        tabel_persen, urutan_aspek,
        judul="Distribusi Sentimen per Aspek (%)",
        label_x="Aspek",
    )

# Fungsi jumlah ulasan per aspek
def grafik_jumlah_aspek(df):
    df_exploded = siapkan_data_exploded(df, "aspek", deteksi_aspek)

    if df_exploded.empty:
        fig = go.Figure()
        fig.update_layout(title="Tidak ada data aspek yang terdeteksi")
        return atur_tema_grafik(fig)

    jumlah_aspek = df_exploded["aspek"].value_counts()
    urutan_aspek = list(ASPECT_KEYWORDS.keys())

    return grafik_jumlah_horizontal(
        jumlah_aspek, urutan_aspek,
        judul="Jumlah Data per Aspek",
        label_y="Aspek",
    )

# Fungsi tabel ringkasan aspek
def tabel_ringkasan_aspek(df):
    df_exploded = siapkan_data_exploded(df, "aspek", deteksi_aspek)

    if df_exploded.empty:
        return pd.DataFrame(columns=["Aspek", "Negatif (%)", "Netral (%)", "Positif (%)"])

    tabel_persen = hitung_persen_kategori(df_exploded, "aspek")
    urutan_aspek = list(ASPECT_KEYWORDS.keys())
    return tabel_ringkasan_persen(tabel_persen, urutan_aspek, "Aspek")


# ---- USE CASE : 2 grafik + 1 tabel ----
# Fungsi distribusi sentimen use case
def grafik_persen_sentimen_usecase(df):
    df_exploded = siapkan_data_exploded(df, "usecase", deteksi_usecase)

    if df_exploded.empty:
        fig = go.Figure()
        fig.update_layout(title="Tidak ada data use case yang terdeteksi")
        return atur_tema_grafik(fig)

    tabel_persen = hitung_persen_kategori(df_exploded, "usecase")
    urutan_uc = list(USECASE_KEYWORDS.keys())  

    return grafik_persen_stacked(
        tabel_persen, urutan_uc,
        judul="Distribusi Sentimen per Use Case (%)",
        label_x="Use Case",
    )

# Fungsi jumlah ulasan per use case
def grafik_jumlah_usecase(df):
    df_exploded = siapkan_data_exploded(df, "usecase", deteksi_usecase)

    if df_exploded.empty:
        fig = go.Figure()
        fig.update_layout(title="Tidak ada data use case yang terdeteksi")
        return atur_tema_grafik(fig)

    jumlah_uc = df_exploded["usecase"].value_counts()
    urutan_uc = list(USECASE_KEYWORDS.keys())

    return grafik_jumlah_horizontal(
        jumlah_uc, urutan_uc,
        judul="Jumlah Data per Use Case",
        label_y="Use Case",
    )

# Fungsi tabel ringkasan use case
def tabel_ringkasan_usecase(df):
    df_exploded = siapkan_data_exploded(df, "usecase", deteksi_usecase)

    if df_exploded.empty:
        return pd.DataFrame(columns=["Use Case", "Negatif (%)", "Netral (%)", "Positif (%)"])

    tabel_persen = hitung_persen_kategori(df_exploded, "usecase")
    urutan_uc = list(USECASE_KEYWORDS.keys())
    return tabel_ringkasan_persen(tabel_persen, urutan_uc, "Use Case")

# Fungsi Wordcloud
def buat_wordcloud(daftar_teks, jenis_sentimen="all"):
    semua_teks = " ".join([bersihkan_teks(t) for t in daftar_teks])

    if semua_teks.strip() == "":
        return None

    pilihan_warna = {"Positif": "Greens", "Negatif": "Reds", "Netral": "Oranges", "all": "cool"}
    warna = pilihan_warna.get(jenis_sentimen, "viridis")

    wc = WordCloud(width=900, height=400, background_color="#111827", colormap=warna, max_words=100).generate(semua_teks)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#111827")
    ax.imshow(wc)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return buf

# Fungsi 1 tab wordcloud
def tampilkan_wordcloud_tab(tab, daftar_teks, jenis_sentimen, pesan_kosong=None):
    with tab:
        if len(daftar_teks) == 0:
            if pesan_kosong:
                st.info(pesan_kosong)
            return
        gambar = buat_wordcloud(daftar_teks, jenis_sentimen)
        if gambar:
            st.image(gambar, use_container_width=True)


# ---- Bagian 7 : Sidebar ----
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem; text-align:center;'>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                  background:linear-gradient(135deg,#6366f1,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🧠 SentiAI
      </div>
      <div style='font-size:0.75rem;color:#475569;margin-top:2px;'>IndoBERT · Sentiment Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Konfigurasi Model**")
    model_path = st.text_input(
        "Path Model IndoBERT", value="indobert_copilot_model",
        help="Folder berisi: config.json, model.safetensors, tokenizer.json"
    )

    st.markdown("---")
    st.markdown("**📊 Pilih Dataset**")
    selected_app = st.selectbox(
        "Dataset yang ditampilkan di Visualisasi",
        options=AI_APPS, index=3,
        help="Pilih salah satu dari 5 dataset untuk dilihat detailnya."
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#475569;'>
    <b style='color:#6366f1'>Model:</b> IndoBERT Fine-tuned<br>
    <b style='color:#6366f1'>Label:</b> Positif · Netral · Negatif<br>
    <b style='color:#6366f1'>Dataset Latih:</b> Copilot Reviews<br>
    <b style='color:#6366f1'>Bahasa:</b> Indonesia
    </div>
    """, unsafe_allow_html=True)


# ---- Bagian 8 : Memuat Model dan Dataset ----
# ---- Download Model dari Google Drive ----

MODEL_DIR = "indobert_copilot_model"
ZIP_FILE = "indobert_copilot_model.zip"

FILE_ID = "1HghcauQXhfARgHeUl95fPbB6qL6culh9"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_DIR):
    with st.spinner("Downloading IndoBERT model..."):
        gdown.download(URL, ZIP_FILE, quiet=False)

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")

        os.remove(ZIP_FILE)

tokenizer   = None
model       = None
model_loaded = False

if os.path.exists(MODEL_DIR):
    with st.spinner("Memuat model IndoBERT..."):
        try:
            tokenizer, model = load_model(MODEL_DIR)
            model_loaded = True
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
else:
    st.sidebar.warning(f"⚠️ Folder model tidak ditemukan: '{model_path}'")

# --- Memuat semua dataset CSV ---
with st.spinner("Memuat dataset..."):
    data_semua_app = muat_semua_dataset()


# ---- Bagian 9 : Header ----
col_title, col_status = st.columns([4, 1])

with col_title:
    st.markdown("""
    <div class='main-title'>SentiAI Dashboard</div>
    <div class='subtitle'>Analisis Sentimen Pengguna Indonesia terhadap Aplikasi AI · IndoBERT Fine-tuned</div>
    """, unsafe_allow_html=True)

with col_status:
    if model_loaded:
        st.markdown("""<div style='background:rgba(16,185,129,0.12);border:1px solid #10b98155;border-radius:10px;
        padding:0.5rem 0.8rem;text-align:center;margin-top:1rem;'>
        <span style='color:#10b981;font-size:0.8rem;font-weight:600;'>● Model Aktif</span></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:rgba(244,63,94,0.12);border:1px solid #f43f5e55;border-radius:10px;
        padding:0.5rem 0.8rem;text-align:center;margin-top:1rem;'>
        <span style='color:#f43f5e;font-size:0.8rem;font-weight:600;'>● Demo Mode</span></div>""", unsafe_allow_html=True)


# ---- Bagian 10 : Tab Utama ----
tab1, tab2, tab3 = st.tabs(["🔍 Inference", "📊 Visualisasi", "📁 Batch Analysis"])


# ==== TAB 1 : INFERENCE ====
with tab1:
    col_kiri, col_kanan = st.columns([3, 2], gap="large")

    with col_kiri:
        st.markdown("<div class='section-header'>Analisis Teks</div>", unsafe_allow_html=True)

        nilai_awal = st.session_state.get("contoh_teks", "")
        user_input = st.text_area(
            "Masukkan teks ulasan", value=nilai_awal,
            placeholder="Contoh: Aplikasinya sangat membantu dan responsif banget...",
            height=130, label_visibility="collapsed",
        )

        col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 3])
        with col_btn1:
            tombol_analisis = st.button("🔍 Analisis", use_container_width=True)
        with col_btn2:
            if st.button("🗑 Hapus", use_container_width=True):
                st.session_state["contoh_teks"] = ""
                st.rerun()

        st.markdown("<div class='upload-hint'>💡 Gunakan ulasan Bahasa Indonesia untuk hasil optimal. Model dilatih pada data ulasan aplikasi Copilot dari Google Play Store.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='font-size:1rem;'>Contoh Teks</div>", unsafe_allow_html=True)

        contoh_teks_map = {
            "😊 Positif": "Fiturnya sangat canggih dan membantu pekerjaan sehari-hari, respon cepat!",
            "😐 Netral":  "Aplikasinya biasa saja, tidak terlalu spesial tapi cukup untuk kebutuhan dasar.",
            "😠 Negatif": "Aplikasi ini sangat lemot dan sering crash, sangat mengecewakan!",
        }

        kolom_contoh = st.columns(3)
        for i, (label_tombol, isi_teks) in enumerate(contoh_teks_map.items()):
            with kolom_contoh[i]:
                if st.button(label_tombol):
                    st.session_state["contoh_teks"] = isi_teks
                    st.rerun()

    with col_kanan:
        st.markdown("<div class='section-header'>Hasil Prediksi</div>", unsafe_allow_html=True)

        if tombol_analisis and user_input.strip() != "":
            if model_loaded:
                with st.spinner("Menganalisis..."):
                    hasil = predict_sentiment(user_input.strip(), tokenizer, model)
            else:
                text_lower = user_input.lower()
                ada_positif = any(k in text_lower for k in ["bagus", "keren", "mantap", "canggih", "suka", "membantu"])
                ada_negatif = any(k in text_lower for k in ["lemot", "buruk", "jelek", "kecewa", "crash", "kesal"])

                if ada_positif:
                    lbl, cfd = "Positif", {"Negatif": 0.08, "Netral": 0.10, "Positif": 0.82}
                elif ada_negatif:
                    lbl, cfd = "Negatif", {"Negatif": 0.78, "Netral": 0.12, "Positif": 0.10}
                else:
                    lbl, cfd = "Netral",  {"Negatif": 0.18, "Netral": 0.64, "Positif": 0.18}

                hasil = {"text": user_input, "label": lbl, "confidence": cfd}

            label_prediksi       = hasil["label"]
            confidence_tertinggi = hasil["confidence"][label_prediksi]
            warna_label          = COLORS[label_prediksi]

            html = "<div class='result-card'>"
            html += f"<div style='margin-bottom:1rem;'><span class='sentiment-badge {BADGE_CLASS[label_prediksi]}'>{EMOJI_MAP[label_prediksi]} {label_prediksi}</span></div>"
            html += "<div style='font-size:0.82rem;color:#64748b;margin-bottom:0.5rem;'>CONFIDENCE SCORE</div>"
            html += f"<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:700;color:{warna_label};margin-bottom:1.2rem;'>{confidence_tertinggi:.1%}</div>"

            for sentimen in ["Positif", "Netral", "Negatif"]:
                nilai = hasil["confidence"][sentimen]
                warna = COLORS[sentimen]
                html += "<div style='margin-bottom:0.3rem;'>"
                html += f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:3px;'><span style='color:#94a3b8;'>{sentimen}</span><span style='color:{warna};font-weight:600;'>{nilai:.1%}</span></div>"
                html += f"<div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{nilai*100:.1f}%;background:{warna};'></div></div>"
                html += "</div>"
            html += "</div>"

            st.markdown(html, unsafe_allow_html=True)

            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].insert(0, hasil)
            st.session_state["history"] = st.session_state["history"][:6]
        else:
            st.markdown("""
            <div class='result-card' style='text-align:center;padding:3rem 1rem;'>
              <div style='font-size:3rem;margin-bottom:0.75rem;'>🔍</div>
              <div style='color:#475569;font-size:0.9rem;'>Masukkan teks dan klik <b>Analisis</b></div>
            </div>""", unsafe_allow_html=True)

    if "history" in st.session_state and st.session_state["history"]:
        st.markdown("<div class='section-header'>Riwayat Analisis</div>", unsafe_allow_html=True)
        kolom_history = st.columns(2)
        for i, item in enumerate(st.session_state["history"]):
            label = item["label"]
            teks  = item["text"][:80] + "..." if len(item["text"]) > 80 else item["text"]
            with kolom_history[i % 2]:
                st.markdown(f"""
                <div class='history-item'>
                  <span style='color:{COLORS[label]};font-weight:600;font-size:0.82rem;'>{EMOJI_MAP[label]} {label}</span>
                  <span style='color:#475569;font-size:0.78rem;float:right;'>{item["confidence"][label]:.1%}</span>
                  <div style='color:#94a3b8;font-size:0.82rem;margin-top:0.3rem;'>{teks}</div>
                </div>""", unsafe_allow_html=True)


# ==== TAB 2 : VISUALISASI DATASET ====
with tab2:
    if len(data_semua_app) == 0:
        st.error("❌ Tidak ada dataset yang berhasil dimuat. Pastikan 5 file CSV ada di folder yang sama dengan app.py.")
    else:
        if selected_app in data_semua_app:
            df_terpilih = data_semua_app[selected_app]
        else:
            df_terpilih = list(data_semua_app.values())[0]

        jumlah_per_label = df_terpilih["label"].value_counts().to_dict()
        total_data = len(df_terpilih)

        jp = jumlah_per_label.get("Positif", 0)
        jn = jumlah_per_label.get("Netral", 0)
        jg = jumlah_per_label.get("Negatif", 0)

        # -------- Kartu statistik --------
        st.markdown(f"<div class='section-header'>Statistik Dataset · {selected_app}</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        kartu_metrik(m1, f"{total_data:,}", "Total Ulasan", "#6366f1")
        kartu_metrik(m2, f"{jp:,} ({jp/total_data*100:.0f}%)", "Positif", "#10b981")
        kartu_metrik(m3, f"{jn:,} ({jn/total_data*100:.0f}%)", "Netral", "#f59e0b")
        kartu_metrik(m4, f"{jg:,} ({jg/total_data*100:.0f}%)", "Negatif", "#f43f5e")
        st.markdown("")

        # -------- Row 1: Donut + Perbandingan 5 app --------
        r1k, r1d = st.columns([1, 2], gap="medium")
        with r1k:
            st.plotly_chart(grafik_donut(jumlah_per_label, f"Distribusi Sentimen · {selected_app}"),
                            use_container_width=True, config={"displayModeBar": False})
        with r1d:
            st.plotly_chart(grafik_perbandingan_app(data_semua_app),
                            use_container_width=True, config={"displayModeBar": False})

        # -------- Row 2: Heatmap --------
        st.plotly_chart(grafik_heatmap_app(data_semua_app),
                        use_container_width=True, config={"displayModeBar": False})
        
        # -------- Row 3: Rating bintang + Tren waktu --------
        r3k, r3d = st.columns(2, gap="medium")
        with r3k:
            st.plotly_chart(grafik_distribusi_rating(df_terpilih),
                            use_container_width=True, config={"displayModeBar": False})
        with r3d:
            fig_tren = grafik_tren_waktu(df_terpilih)
            if fig_tren:
                st.plotly_chart(fig_tren, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Kolom tanggal ('at') tidak tersedia untuk grafik tren.")

        # -------- WordCloud --------
        st.markdown("<div class='section-header'>WordCloud Sentimen</div>", unsafe_allow_html=True)

        # Filter teks berdasarkan label sentimen
        teks_positif = df_terpilih[df_terpilih["label"] == "Positif"]["content"].tolist()
        teks_netral  = df_terpilih[df_terpilih["label"] == "Netral"]["content"].tolist()
        teks_negatif = df_terpilih[df_terpilih["label"] == "Negatif"]["content"].tolist()
        teks_semua   = df_terpilih["content"].tolist()

        wc1, wc2, wc3, wc4 = st.tabs(["🌈 Semua", "😊 Positif", "😐 Netral", "😠 Negatif"])
        tampilkan_wordcloud_tab(wc1, teks_semua,   "all")
        tampilkan_wordcloud_tab(wc2, teks_positif, "Positif", "Tidak ada ulasan Positif")
        tampilkan_wordcloud_tab(wc3, teks_netral,  "Netral",  "Tidak ada ulasan Netral")
        tampilkan_wordcloud_tab(wc4, teks_negatif, "Negatif", "Tidak ada ulasan Negatif")

        # -------- Aspek --------
        st.markdown("<div class='section-header'>Analisis Berdasarkan Aspek</div>", unsafe_allow_html=True)

        a1, a2 = st.columns(2, gap="medium")
        with a1:
            st.plotly_chart(grafik_persen_sentimen_aspek(df_terpilih),
                            use_container_width=True, config={"displayModeBar": False})
        with a2:
            st.plotly_chart(grafik_jumlah_aspek(df_terpilih),
                            use_container_width=True, config={"displayModeBar": False})

        st.dataframe(tabel_ringkasan_aspek(df_terpilih),
                        use_container_width=True)
        
        # --------Use Case --------
        st.markdown("<div class='section-header'>Analisis Berdasarkan Use Case</div>", unsafe_allow_html=True)

        a1, a2 = st.columns(2, gap="medium")
        with a1:
            st.plotly_chart(grafik_persen_sentimen_usecase(df_terpilih),
                            use_container_width=True, config={"displayModeBar": False})
        with a2:
            st.plotly_chart(grafik_jumlah_usecase(df_terpilih),
                            use_container_width=True, config={"displayModeBar": False})

        st.dataframe(tabel_ringkasan_usecase(df_terpilih),
                        use_container_width=True)


# ==== TAB 3 : BATCH ANALYSIS ====
with tab3:
    st.markdown("<div class='section-header'>Upload Dataset untuk Analisis Batch</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    📋 Upload file CSV baru dengan kolom <code>content</code> (teks ulasan).
    Kolom <code>score</code> (rating 1-5) dan <code>at</code> (tanggal) bersifat opsional.
    Model IndoBERT akan memprediksi sentimen setiap baris.
    </div>""", unsafe_allow_html=True)

    col_upload, col_format = st.columns([2, 1])
    with col_upload:
        uploaded_file      = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")
        selected_app_batch = st.selectbox("Tandai sebagai dataset dari AI App:", options=AI_APPS, index=3)
    with col_format:
        st.markdown("""
        <div class='metric-card' style='margin-top:0.5rem;'>
          <div style='font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;'>Format CSV</div>
          <div style='font-family:monospace;font-size:0.78rem;color:#818cf8;background:#0a0e1a;padding:0.6rem;border-radius:8px;'>
            content,score,at<br>"Teks ulasan...",5,2024-01-15
          </div>
        </div>""", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df_upload     = pd.read_csv(uploaded_file)
            jumlah_baris  = len(df_upload)
            st.success(f"✅ Dataset dimuat: {jumlah_baris:,} baris · Kolom: {', '.join(df_upload.columns.tolist())}")

            if "content" not in df_upload.columns:
                st.error("❌ Kolom 'content' tidak ditemukan dalam file CSV.")
            else:
                col_run, col_sample = st.columns([1, 3])
                with col_run:
                    tombol_jalankan = st.button("🚀 Jalankan Analisis", use_container_width=True)
                with col_sample:
                    n_sample = st.slider("Jumlah sample dianalisis (0 = semua)", 0, min(jumlah_baris, 500), 100)

                if tombol_jalankan:
                    daftar_teks = df_upload["content"].fillna("").tolist()
                    if n_sample > 0:
                        daftar_teks = daftar_teks[:n_sample]

                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    if model_loaded:
                        status_text.text("Menganalisis dengan model IndoBERT...")
                        hasil_batch = predict_banyak_teks(daftar_teks, tokenizer, model, progress_bar)
                    else:
                        hasil_batch  = []
                        kata_positif = ["bagus", "keren", "mantap", "suka", "membantu", "cepat"]
                        kata_negatif = ["lemot", "buruk", "jelek", "kecewa", "crash", "kesal"]
                        for i, teks in enumerate(daftar_teks):
                            tl = str(teks).lower()
                            if any(k in tl for k in kata_positif):
                                lbl, cfd = "Positif", np.random.uniform(0.7, 0.95)
                            elif any(k in tl for k in kata_negatif):
                                lbl, cfd = "Negatif", np.random.uniform(0.65, 0.92)
                            else:
                                lbl = np.random.choice(["Positif", "Netral", "Negatif"], p=[0.45, 0.30, 0.25])
                                cfd = np.random.uniform(0.55, 0.85)
                            hasil_batch.append({"text": teks, "label": lbl, "confidence": cfd})
                            progress_bar.progress((i + 1) / len(daftar_teks))

                    progress_bar.empty()
                    status_text.empty()

                    df_hasil = pd.DataFrame(hasil_batch)
                    if "score" in df_upload.columns:
                        df_hasil["score"] = df_upload["score"].values[:len(df_hasil)]
                    if "at" in df_upload.columns:
                        df_hasil["at"] = df_upload["at"].values[:len(df_hasil)]

                    st.session_state["df_hasil_batch"]  = df_hasil
                    st.session_state["nama_app_batch"]  = selected_app_batch

        except Exception as e:
            st.error(f"Error membaca file: {e}")

    if "df_hasil_batch" in st.session_state:
        df_hasil = st.session_state["df_hasil_batch"]
        nama_app = st.session_state.get("nama_app_batch", "Dataset")

        st.markdown(f"<div class='section-header'>Hasil Analisis · {nama_app}</div>", unsafe_allow_html=True)

        jpl  = df_hasil["label"].value_counts().to_dict()
        tot  = len(df_hasil)
        jp   = jpl.get("Positif", 0)
        jn   = jpl.get("Netral", 0)
        jg   = jpl.get("Negatif", 0)

        mc1, mc2, mc3, mc4 = st.columns(4)
        kartu_metrik(mc1, f"{tot:,}",                     "Total Dianalisis", "#6366f1")
        kartu_metrik(mc2, f"{jp:,} ({jp/tot*100:.0f}%)",  "Positif",          "#10b981")
        kartu_metrik(mc3, f"{jn:,} ({jn/tot*100:.0f}%)",  "Netral",           "#f59e0b")
        kartu_metrik(mc4, f"{jg:,} ({jg/tot*100:.0f}%)",  "Negatif",          "#f43f5e")
        st.markdown("")

        df_visual = df_hasil.rename(columns={"text": "content"})
        bc1, bc2 = st.columns(2, gap="medium")
        with bc1:
            st.plotly_chart(grafik_persen_sentimen_aspek(df_visual),
                            use_container_width=True, config={"displayModeBar": False})
        with bc2:
            st.plotly_chart(grafik_jumlah_aspek(df_visual),
                            use_container_width=True, config={"displayModeBar": False})
        st.dataframe(
            tabel_ringkasan_aspek(df_visual),
            use_container_width=True)

        uc1, uc2 = st.columns(2, gap="medium")
        with uc1:
            st.plotly_chart(grafik_persen_sentimen_usecase(df_visual),
                            use_container_width=True, config={"displayModeBar": False})
        with uc2:
            st.plotly_chart(
                grafik_jumlah_usecase(df_visual),
                use_container_width=True, config={"displayModeBar": False})
        st.dataframe(
            tabel_ringkasan_usecase(df_visual),
            use_container_width=True)
            
        st.markdown("<div class='section-header'>WordCloud Hasil Batch</div>", unsafe_allow_html=True)
        wbt1, wbt2, wbt3, wbt4 = st.tabs(["🌈 Semua", "😊 Positif", "😐 Netral", "😠 Negatif"])
        tampilkan_wordcloud_tab(wbt1, df_hasil["text"].tolist(), "all")
        tampilkan_wordcloud_tab(wbt2, df_hasil[df_hasil["label"] == "Positif"]["text"].tolist(), "Positif", "Tidak ada data Positif")
        tampilkan_wordcloud_tab(wbt3, df_hasil[df_hasil["label"] == "Netral"]["text"].tolist(),  "Netral",  "Tidak ada data Netral")
        tampilkan_wordcloud_tab(wbt4, df_hasil[df_hasil["label"] == "Negatif"]["text"].tolist(), "Negatif", "Tidak ada data Negatif")

        st.markdown("<div class='section-header'>Ekspor Hasil</div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Hasil CSV",
            data=df_hasil.to_csv(index=False),
            file_name=f"sentimen_{nama_app.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

        st.markdown("<div class='section-header'>Preview Data</div>", unsafe_allow_html=True)
        st.dataframe(df_hasil[["text", "label", "confidence"]].head(50), use_container_width=True, height=300)