# app.py — NeuroTap Parkinson's Voice Screening (FFmpeg-free version)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import tempfile
import matplotlib.pyplot as plt
import librosa.display
import plotly.graph_objects as go
import io
import datetime
import json
import os
import hashlib
import traceback
import soundfile as sf

# recorder (streamlit-webrtc)
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# PDF generation (optional)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# -------------------------------------------------
# Config
# -------------------------------------------------
st.set_page_config(page_title="NeuroTap – Parkinson's Voice Screening", page_icon="🧠", layout="wide")

USERS_FILE = "users.json"
HISTORY_FILE = "user_history.json"
MODEL_PATHS = ["voice_model.pkl", os.path.join("model_reports", "voice_model.pkl")]
DEFAULT_N_MFCC = 22

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump({}, f)

# -------------------------------------------------
# Utility - user management & history
# -------------------------------------------------
def load_users():
    if os.path.exists(USERS_FILE):
        try:
            return json.load(open(USERS_FILE, "r"))
        except Exception:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_history():
    try:
        return json.load(open(HISTORY_FILE, "r"))
    except:
        return {}

def save_history(data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def add_history_entry(username, entry):
    h = load_history()
    if username not in h:
        h[username] = []
    h[username].append(entry)
    save_history(h)

# -------------------------------------------------
# Load ML model
# -------------------------------------------------
model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            model = joblib.load(p)
            break
        except Exception:
            model = None

if model is None:
    st.error("❌ Model file `voice_model.pkl` not found. Place it in root or model_reports/")
    st.stop()

# -------------------------------------------------
# Audio helpers
# -------------------------------------------------
def save_bytes_to_tmp(data_bytes, suffix=".wav"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

@st.cache_data(show_spinner=False)
def extract_voice_features(audio_path, n_mfcc=DEFAULT_N_MFCC):
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        raise RuntimeError(f"librosa.load failed: {e}")
    y, _ = librosa.effects.trim(y)
    if y.size == 0:
        raise RuntimeError("Audio empty after trimming")
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean, y, sr, mfcc

# -------------------------------------------------
# Recorder (no ffmpeg)
# -------------------------------------------------
def record_audio(label="🎤 Recorder"):
    """
    Returns recorded audio as WAV bytes or None
    """
    ctx = webrtc_streamer(
        key=label,
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.state.playing and ctx.audio_receiver:
        frames = ctx.audio_receiver.get_frames(timeout=1)
        if frames:
            # concatenate audio frames
            audio = np.concatenate([f.to_ndarray().flatten() for f in frames])
            buf = io.BytesIO()
            sf.write(buf, audio, 48000, format="WAV")
            buf.seek(0)
            return buf.read()
    return None

# -------------------------------------------------
# Prediction helpers
# -------------------------------------------------
def _get_positive_proba(model, vec):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([vec])[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                pos_idx = list(classes).index(1)
            except ValueError:
                pos_idx = 1 if len(probs) > 1 else 0
        else:
            pos_idx = 1 if len(probs) > 1 else 0
        return float(probs[pos_idx])
    else:
        pred = model.predict([vec])[0]
        try:
            return float(pred)
        except:
            return 0.0

def interpret_score(score):
    if score < 35:
        return "Likely Healthy", "😃", "Low risk"
    elif score < 60:
        return "Borderline", "😐", "Monitor"
    elif score < 80:
        return "Moderate Risk", "😟", "Consider evaluation"
    else:
        return "High Risk", "🚨", "Seek medical advice"

def predict_vector(vec):
    proba = _get_positive_proba(model, vec)
    percent = int(round(proba * 100))
    label, emoji, desc = interpret_score(percent)
    return {"label": label, "emoji": emoji, "desc": desc, "score": percent, "prob": proba}

# -------------------------------------------------
# Plots and PDF helpers
# -------------------------------------------------
def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(6,2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title=title)
    return fig

def plot_spectrogram(y, sr, title="Spectrogram"):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(6,3))
    im = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    ax.set(title=title)
    return fig

def plot_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Confidence (%)"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range':[0,40], 'color':"lightgreen"},
                {'range':[40,70], 'color':"yellow"},
                {'range':[70,100], 'color':"red"},
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def generate_pdf(user, entry, figs, mode="single", entry_b=None):
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("🧠 NeuroTap – Parkinson's Voice Report", styles["Title"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(f"<b>User:</b> {user}", styles["Normal"]))
    flow.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    if mode=="single":
        data = [["Result", entry["label"]],
                ["Confidence", f"{entry['score']}%"],
                ["Remarks", entry["desc"]]]
    else:
        data = [["", "Sample A", "Sample B"],
                ["Result", entry["label"], entry_b["label"]],
                ["Confidence", f"{entry['score']}%", f"{entry_b['score']}%"],
                ["Remarks", entry["desc"], entry_b["desc"]]]
    table = Table(data)
    table.setStyle(TableStyle([("BOX", (0,0), (-1,-1), 1, colors.black),
                               ("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
    flow.append(table)
    flow.append(Spacer(1,12))

    for f in figs:
        try:
            img = fig_to_img(f)
            flow.append(Image(img, width=400, height=200))
            flow.append(Spacer(1,8))
        except:
            pass

    doc.build(flow)
    buf.seek(0)
    return buf

# -------------------------------------------------
# Authentication UI
# -------------------------------------------------
st.sidebar.title("🔐 Authentication")
users = load_users()
mode = st.sidebar.radio("Choose:", ("Login","Signup"))

if mode=="Signup":
    u = st.sidebar.text_input("New Username")
    p = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Create account"):
        if not u or not p:
            st.sidebar.error("Enter both fields")
        elif u in users:
            st.sidebar.error("User exists")
        else:
            users[u] = hash_password(p)
            save_users(users)
            st.sidebar.success("Account created, login now.")
            st.rerun()

if mode=="Login":
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if u in users and users[u]==hash_password(p):
            st.session_state["auth_user"]=u
            st.sidebar.success(f"Welcome {u}")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

if "auth_user" not in st.session_state:
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.pop("auth_user", None)
    st.rerun()

# -------------------------------------------------
# Main app UI
# -------------------------------------------------
st.title("🧠 NeuroTap – Parkinson's Voice Screening")

tab1, tab2 = st.tabs(["🔍 Single Test","⚖️ Compare Voices"])

# --- Single Test ---
with tab1:
    st.subheader("Single Test")
    method = st.radio("Input:", ["Upload","Record"])
    file = None
    rec = None

    if method == "Upload":
        file = st.file_uploader("Upload .wav/.mp3", type=["wav","mp3"])
    else:
        rec = record_audio("Single Test Recorder")

    if st.button("Analyze"):
        if (not file) and (not rec):
            st.warning("Provide audio via upload or record")
        else:
            try:
                if file:
                    tmp_path = save_bytes_to_tmp(file.read(), suffix=os.path.splitext(file.name)[1] or ".wav")
                else:
                    tmp_path = save_bytes_to_tmp(rec, suffix=".wav")

                vec, y, sr, _ = extract_voice_features(tmp_path)
                res = predict_vector(vec)
                st.markdown(f"### {res['emoji']} {res['label']}")
                st.metric("Confidence", f"{res['score']}%")
                st.info(res["desc"])
                st.plotly_chart(plot_gauge(res["score"]))
                st.pyplot(plot_waveform(y, sr))
                st.pyplot(plot_spectrogram(y, sr))
                with open(tmp_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")

                entry = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "label": res["label"], "score": res["score"], "desc": res["desc"]}
                add_history_entry(st.session_state["auth_user"], entry)

                if HAS_REPORTLAB:
                    pdf = generate_pdf(st.session_state["auth_user"], entry, [plot_waveform(y,sr), plot_spectrogram(y,sr)])
                    st.download_button("📥 Download PDF", data=pdf, file_name="report.pdf", mime="application/pdf")
                else:
                    st.info("Install reportlab to enable PDF reports.")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.error(traceback.format_exc())

# --- Compare Voices ---
with tab2:
    st.subheader("Compare A vs B")
    col1, col2 = st.columns(2)

    up_a = col1.file_uploader("Sample A", type=["wav","mp3"], key="up_a")
    up_b = col2.file_uploader("Sample B", type=["wav","mp3"], key="up_b")

    rec_a = col1.empty()
    rec_b = col2.empty()

    with col1:
        rec_a_bytes = record_audio("Recorder A")
    with col2:
        rec_b_bytes = record_audio("Recorder B")

    if st.button("Compare"):
        if not (up_a or rec_a_bytes) or not (up_b or rec_b_bytes):
            st.warning("Provide both samples (upload or record)")
        else:
            try:
                def prepare_input(uploaded, recorded_bytes):
                    if uploaded:
                        return save_bytes_to_tmp(uploaded.read(), suffix=os.path.splitext(uploaded.name)[1] or ".wav")
                    else:
                        return save_bytes_to_tmp(recorded_bytes, suffix=".wav")

                pa = prepare_input(up_a, rec_a_bytes)
                pb = prepare_input(up_b, rec_b_bytes)

                va, ya, sra, _ = extract_voice_features(pa)
                vb, yb, srb, _ = extract_voice_features(pb)
                ra = predict_vector(va)
                rb = predict_vector(vb)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Sample A:** {ra['emoji']} {ra['label']}")
                    st.metric("Conf", f"{ra['score']}%")
                    st.pyplot(plot_waveform(ya, sra))
                    with open(pa, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                with c2:
                    st.markdown(f"**Sample B:** {rb['emoji']} {rb['label']}")
                    st.metric("Conf", f"{rb['score']}%")
                    st.pyplot(plot_waveform(yb, srb))
                    with open(pb, "rb") as f:
                        st.audio(f.read(), format="audio/wav")

                fig = go.Figure(data=[go.Bar(x=["A","B"], y=[ra['score'], rb['score']])])
                fig.update_layout(title="Comparison", yaxis=dict(range=[0,100]))
                st.plotly_chart(fig)

                entry_a = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "label": ra["label"], "score": ra["score"], "desc": ra["desc"]}
                entry_b = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "label": rb["label"], "score": rb["score"], "desc": rb["desc"]}
                add_history_entry(st.session_state["auth_user"], entry_a)
                add_history_entry(st.session_state["auth_user"], entry_b)

                if HAS_REPORTLAB:
                    pdf = generate_pdf(st.session_state["auth_user"], entry_a, [fig], mode="compare", entry_b=entry_b)
                    st.download_button("📥 Download Comparison PDF", data=pdf, file_name="comparison.pdf", mime="application/pdf")
                else:
                    st.info("Install reportlab to enable PDF reports.")
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                st.error(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("NeuroTap demo – not medical advice")
