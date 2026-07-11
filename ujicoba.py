import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="Analisis Sentimen AI",
    layout="wide"
)

st.title("🧠 Analisis Sentimen Aplikasi AI")
st.write("Menggunakan Model IndoBERT")

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():

    model_path = "indobert_grok_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ==================================================
# LABEL
# ==================================================
LABEL_MAP = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# ==================================================
# INFERENCE
# ==================================================
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            "Negatif": float(probs[0][0]),
            "Netral": float(probs[0][1]),
            "Positif": float(probs[0][2])
        }
    }

# ==================================================
# TABS
# ==================================================
tab1, tab2 = st.tabs([
    "🔍 Prediksi",
    "📊 Dashboard"
])

# ==================================================
# TAB 1
# ==================================================
with tab1:

    st.subheader("Prediksi Sentimen")

    text = st.text_area(
        "Masukkan ulasan"
    )

    if st.button("Analisis"):

        result = predict_sentiment(text)

        st.success(
            f"Sentimen : {result['label']}"
        )

        prob_df = pd.DataFrame({
            "Sentimen": list(result["confidence"].keys()),
            "Probabilitas": list(result["confidence"].values())
        })

        fig = px.bar(
            prob_df,
            x="Sentimen",
            y="Probabilitas",
            color="Sentimen"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# TAB 2
# ==================================================
with tab2:

    st.subheader("Dashboard Analisis Sentimen")

    uploaded_file = st.file_uploader(
        "Upload hasil inference",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        # wajib ada kolom sentiment
        # content | sentiment

        # ==========================================
        # KPI
        # ==========================================

        total = len(df)

        positif = len(
            df[df["sentiment"]=="Positif"]
        )

        netral = len(
            df[df["sentiment"]=="Netral"]
        )

        negatif = len(
            df[df["sentiment"]=="Negatif"]
        )

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Total", total)
        c2.metric("Positif", positif)
        c3.metric("Netral", netral)
        c4.metric("Negatif", negatif)

        # ==========================================
        # DISTRIBUSI SENTIMEN
        # ==========================================

        st.subheader(
            "Distribusi Sentimen"
        )

        fig = px.histogram(
            df,
            x="sentiment",
            color="sentiment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # WORDCLOUD
        # ==========================================

        st.subheader(
            "WordCloud"
        )

        text_all = " ".join(
            df["content"].astype(str)
        )

        wc = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(text_all)

        fig_wc, ax = plt.subplots()

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig_wc)

        # ==========================================
        # ASPEK
        # ==========================================

        st.subheader(
            "Analisis Aspek"
        )

        aspek_dict = {

            "Performa":[
                "cepat",
                "lemot",
                "loading",
                "error"
            ],

            "Fitur":[
                "fitur",
                "fungsi",
                "tools"
            ],

            "Akurasi":[
                "akurat",
                "jawaban",
                "relevan"
            ],

            "UI/UX":[
                "tampilan",
                "desain",
                "interface"
            ]
        }

        def get_aspek(text):

            text = str(text).lower()

            for aspek, keywords in aspek_dict.items():

                for word in keywords:

                    if word in text:

                        return aspek

            return "Lainnya"

        df["Aspek"] = df["content"].apply(
            get_aspek
        )

        aspek_df = (
            df.groupby(
                ["Aspek","sentiment"]
            )
            .size()
            .reset_index(name="Jumlah")
        )

        fig_aspek = px.bar(
            aspek_df,
            x="Aspek",
            y="Jumlah",
            color="sentiment",
            barmode="group"
        )

        st.plotly_chart(
            fig_aspek,
            use_container_width=True
        )

        # ==========================================
        # USE CASE
        # ==========================================

        st.subheader(
            "Analisis Use Case"
        )

        usecase_dict = {

            "Coding":[
                "python",
                "coding",
                "program"
            ],

            "Writing":[
                "menulis",
                "artikel",
                "essay"
            ],

            "Research":[
                "riset",
                "penelitian",
                "skripsi"
            ],

            "Education":[
                "belajar",
                "tugas",
                "kuliah"
            ]
        }

        def get_usecase(text):

            text = str(text).lower()

            for uc, keywords in usecase_dict.items():

                for word in keywords:

                    if word in text:

                        return uc

            return "Lainnya"

        df["UseCase"] = df["content"].apply(
            get_usecase
        )

        uc_df = (
            df.groupby(
                ["UseCase","sentiment"]
            )
            .size()
            .reset_index(name="Jumlah")
        )

        fig_uc = px.bar(
            uc_df,
            x="UseCase",
            y="Jumlah",
            color="sentiment"
        )

        st.plotly_chart(
            fig_uc,
            use_container_width=True
        )