import streamlit as st
import numpy as np
import librosa
import pickle
import whisper
import re
import spacy
import tempfile
import os
import base64

from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
from googletrans import Translator

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Accent-Aware Speech Intelligence System",
    page_icon="🎧",
    layout="centered"
)

# ================= BACKGROUND =================
def set_bg_image(img_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_path}");
            background-size: cover;
            background-attachment: fixed;
        }}

        .glass {{
            background: rgba(0, 123, 255, 0.12);
            border: 1px solid rgba(0, 200, 255, 0.5);
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            margin-bottom: 20px;
            transition: 0.3s ease;
        }}

        .glass:hover {{
            transform: scale(1.02);
        }}

        .action-card {{
            background: linear-gradient(135deg,#0f2027,#2c5364);
            border: 1px solid rgba(0, 200, 255, 0.4);
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(0, 200, 255, 0.2);
            transition: 0.3s;
        }}

        .action-card:hover {{
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(0, 200, 255, 0.5);
        }}

        /* ===== PREMIUM MODE BUTTONS ===== */
        div[data-testid="stButton"] > button {{
            background: linear-gradient(135deg,#0f2027,#2c5364);
            border: 1px solid rgba(0, 200, 255, 0.6);
            color: white;
            padding: 15px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 15px;
            transition: 0.3s ease;
            box-shadow: 0 0 15px rgba(0, 200, 255, 0.3);
        }}

        div[data-testid="stButton"] > button:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 25px rgba(0, 200, 255, 0.8);
            border: 1px solid rgba(0, 200, 255, 1);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

with open("background.jpg", "rb") as img_file:
    encoded = base64.b64encode(img_file.read()).decode()
set_bg_image(encoded)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    accent_model = pickle.load(open("models/accent_classifier.pkl", "rb"))
    whisper_model = whisper.load_model("small")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summary_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    nlp = spacy.load("en_core_web_sm")
    return accent_model, whisper_model, tokenizer, summary_model, nlp

accent_model, whisper_model, tokenizer, summary_model, nlp = load_models()
translator = Translator()

LABEL_MAP = {0: "American", 1: "British", 2: "Indian"}
ACCENT_EMOJI = {"American": "US", "British": "UK", "Indian": "IN"}

ACTION_KEYWORDS = ["call","bring","buy","need","meet","schedule","send","do","give","take","solve"]

LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Spanish": "es"
}

# ================= NLP FUNCTIONS =================
def detect_tone(text):
    t = text.lower()
    if any(w in t for w in ["urgent","asap","immediately"]):
        return "Urgent"
    if any(w in t for w in ["inform","update","regarding"]):
        return "Informational"
    return "Neutral"

def detect_intent(text):
    t = text.lower()
    if "please" in t:
        return "Request"
    if any(k in t for k in ACTION_KEYWORDS):
        return "Task-Oriented"
    return "General"

def extract_entities(text):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def categorize_action(sentence):
    s = sentence.lower()
    if any(k in s for k in ["call","email","message"]):
        return "Communication"
    if any(k in s for k in ["buy","bring"]):
        return "Purchase"
    if any(k in s for k in ["meet","schedule"]):
        return "Scheduling"
    return "Task"

def extract_categorized_actions(text):
    actions = []
    for sent in re.split(r"[.!?]", text):
        sent = sent.strip()
        if sent and any(k in sent.lower() for k in ACTION_KEYWORDS):
            actions.append({
                "action": sent,
                "category": categorize_action(sent)
            })
    return actions

def generate_summary(text):
    if len(text.split()) <= 15:
        return text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    summary_ids = summary_model.generate(
        inputs["input_ids"],
        max_length=50,
        min_length=8,
        num_beams=1,
        no_repeat_ngram_size=4,
        early_stopping=True,
        do_sample=False
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# ================= PROCESS AUDIO FUNCTION (ADDED BACK) =================
def process_audio(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = np.mean(mfcc, axis=1).reshape(1,-1)

    prediction = accent_model.predict(features)[0]
    probabilities = accent_model.predict_proba(features)

    accent = LABEL_MAP[int(prediction)]
    confidence = np.max(probabilities)*100

    result = whisper_model.transcribe(audio, language="en")
    transcription = result["text"]

    tone=detect_tone(transcription)
    intent=detect_intent(transcription)
    entities=extract_entities(transcription)
    actions=extract_categorized_actions(transcription)
    summary=generate_summary(transcription)

    return accent,confidence,transcription,tone,intent,entities,actions,summary

# ================= PDF FUNCTION =================
def generate_pdf(accent,intent,tone,actions,entities,summary):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(15, 32, 39)
    pdf.rect(0, 0, 210, 30, 'F')

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 18)
    pdf.set_xy(10, 10)
    pdf.cell(0, 10, "Accent-Aware Speech Analysis Report", ln=True)

    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Speech Insights", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Accent: {accent}", ln=True)
    pdf.cell(0, 8, f"Intent: {intent}", ln=True)
    pdf.cell(0, 8, f"Tone: {tone}", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detected Action Categories", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "", 12)
    action_text = ", ".join(actions) if actions else "None"
    pdf.multi_cell(0, 8, action_text)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Named Entities", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "", 12)

    if entities:
        for ent in entities:
            pdf.multi_cell(0, 8, f"{ent['label']} : {ent['text']}")
    else:
        pdf.multi_cell(0, 8, "None")

    pdf.ln(8)

    pdf.set_fill_color(240, 248, 255)
    pdf.set_draw_color(0, 123, 255)
    pdf.rect(10, pdf.get_y(), 190, 40, 'DF')

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Summary", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, summary)

    pdf_path = "Speech_Analysis_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ================= UI =================
st.title("🎧 Accent-Aware Speech Intelligence System")
st.caption("Advanced AI-powered speech intelligence platform")

st.markdown("## 🎛 Select Input Mode")
st.markdown("### Choose Audio Source")

col1, col2 = st.columns(2)

with col1:
    upload_btn = st.button("📂 Upload Audio File", use_container_width=True)

with col2:
    record_btn = st.button("🎙️ Record Live Audio", use_container_width=True)

if "mode" not in st.session_state:
    st.session_state.mode = None

if upload_btn:
    st.session_state.mode = "Upload Audio File"

if record_btn:
    st.session_state.mode = "Record Live Audio"

mode = st.session_state.mode

audio_data = None
sr = 16000

if mode == "Upload Audio File":
    uploaded = st.file_uploader("Upload English Audio (.wav / .mp3)", type=["wav","mp3"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name
        audio_data, sr = librosa.load(audio_path, sr=16000)
        os.remove(audio_path)

if mode == "Record Live Audio":
    audio_bytes = st.audio_input("🎙 Record your voice")
    if audio_bytes is not None:
        st.audio(audio_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            audio_path = tmp.name
        audio_data, sr = librosa.load(audio_path, sr=16000)
        os.remove(audio_path)

if audio_data is not None:

    accent,confidence,transcription,tone,intent,entities,actions,summary = process_audio(audio_data, sr)

    st.snow()

    st.subheader("🎯 Quick Insights")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Accent",f"{ACCENT_EMOJI[accent]} {accent}")
    c2.metric("Confidence",f"{confidence:.2f}%")
    c3.metric("Intent",intent)
    c4.metric("Tone",tone)

    st.markdown("### 🔎 Accent Confidence Level")
    st.progress(int(confidence))

    with st.expander("📝 Transcription"):
        st.write(transcription)

    st.subheader("🔖 Named Entities")
    if entities:
        cols = st.columns(3)
        for i, ent in enumerate(entities):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="action-card" style="text-align:center;">
                    <h4>{ent['text']}</h4>
                    <p style="color:#6fd3ff;"><b>{ent['label']}</b></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No named entities detected.")

    st.subheader("✅ Action Items")
    if actions:
        for a in actions:
            st.markdown(f"""
            <div class="action-card">
                <h4>{a['action']}</h4>
                <p>Category: <b>{a['category']}</b></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No action items detected.")

    st.subheader("📌 Summary")
    st.write(summary)

    st.subheader("🌍 Spoken Summary (Multi-Language)")
    selected_lang=st.selectbox("Select Language", list(LANGUAGES.keys()))

    translated = translator.translate(summary, dest=LANGUAGES[selected_lang]).text
    tts=gTTS(translated, lang=LANGUAGES[selected_lang])
    tts_path="spoken_summary.mp3"
    tts.save(tts_path)
    st.audio(tts_path)

    pdf_file=generate_pdf(
        accent,
        intent,
        tone,
        [a["category"] for a in actions],
        entities,
        summary
    )

    with open(pdf_file,"rb") as f:
        st.download_button(
            "⬇ Download PDF Report",
            f,
            file_name="Speech_Analysis_Report.pdf"
        )

st.markdown("""
---
### 🚀 System Capabilities
- Accent Detection with Confidence Score
- Speech-to-Text (Whisper)
- Intent & Tone Detection
- Named Entity Recognition
- Action Extraction
- Transformer-based Summarization
- Multilingual Spoken Summary
- Exportable PDF Report
""")