import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
from st_audiorec import st_audiorec

# Page Config
st.set_page_config(page_title="NeuroTap - Parkinson's Detection", page_icon="🧠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #f0f8ff, #e6f2ff);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("ℹ️ About NeuroTap")
st.sidebar.info("""
**NeuroTap** uses AI to analyze voice recordings 
and assess the risk of **Parkinson’s Disease**.

You can either:
- 📂 Upload a `.wav` or `.mp3` file  
- 🎤 Record directly from your laptop/phone mic  
""")

# Load ML Model
with open("voice_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature Extraction
def extract_voice_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

# Prediction
def predict_voice(mfcc_vector):
    if mfcc_vector is None:
        return "Error", 0, "Could not process voice file."

    proba = model.predict_proba([mfcc_vector])[0][1]
    percent = round(proba * 100)

    if percent < 20:
        return "🟩 Very Low Risk", percent, "Your voice shows no sign of Parkinson’s."
    elif percent < 40:
        return "🟩 Low Risk", percent, "Minor voice variations found — healthy."
    elif percent < 60:
        return "🟨 Monitor", percent, "Some voice traits overlap. Recheck suggested."
    elif percent < 80:
        return "🟧 Moderate Risk", percent, "Speech patterns suggest possible early signs."
    else:
        return "🟥 High Risk", percent, "Strong vocal patterns linked to Parkinson’s. Please consult a doctor."


# Main UI
st.title("🧠 NeuroTap – Parkinson's Detection from Voice")
st.write("Upload a voice sample or record your voice live for instant analysis.")

option = st.radio("Choose Input Method:", ["📂 Upload File", "🎤 Record with Laptop/Phone Mic"])

# --- File Upload ---
if option == "📂 Upload File":
    voice_file = st.file_uploader("Upload your voice sample (.wav or .mp3)", type=["wav", "mp3"])
    if st.button("🔍 Predict from File"):
        if voice_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                data, samplerate = sf.read(voice_file)
                sf.write(temp.name, data, samplerate)
                features = extract_voice_features(temp.name)

            level, score, description = predict_voice(features)
            st.subheader(level)
            st.progress(score)
            st.markdown(f"### 📊 Confidence: **{score}%**")
            st.info(description)
        else:
            st.warning("Please upload a voice sample.")

# --- Record with Laptop/Phone Mic ---
elif option == "🎤 Record with Laptop/Phone Mic":
    st.write("Press the record button below to capture your voice:")

    audio_data = st_audiorec()

    if audio_data is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_data)
            temp.flush()
            features = extract_voice_features(temp.name)

        level, score, description = predict_voice(features)
        st.subheader(level)
        st.progress(score)
        st.markdown(f"### 📊 Confidence: **{score}%**")
        st.info(description)

        # Playback
        st.audio(audio_data, format="audio/wav")
