import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
import librosa.display
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io
import datetime
from st_audiorec import st_audiorec

# =========================================
# PAGE CONFIG & CSS
# =========================================
st.set_page_config(page_title="NeuroTap - Parkinson's Detection", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .reportview-container {background: linear-gradient(135deg, #f0f8ff, #e6f2ff);}
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

# Sidebar
st.sidebar.title("ℹ About NeuroTap")
st.sidebar.info("""
*NeuroTap* uses AI to analyze voice recordings 
and assess the risk of *Parkinson’s Disease*.

You can either:
- 📂 Upload a .wav or .mp3 file  
- 🎤 Record directly from mic  
- 🆚 Compare two voices  
""")

# Load model
with open("voice_model.pkl", "rb") as f:
    model = pickle.load(f)


# =========================================
# FEATURE EXTRACTION
# =========================================
def extract_voice_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean, y, sr, mfcc
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None, None, None, None


# =========================================
# PREDICTION
# =========================================
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


# =========================================
# PDF REPORT
# =========================================
def generate_pdf_report(result, plots, mode="single"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    flowables.append(Paragraph("<b>🧠 NeuroTap – Parkinson's Voice Analysis</b>", styles['Title']))
    flowables.append(Spacer(1, 12))
    flowables.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    flowables.append(Spacer(1, 12))

    if mode == "single":
        level, score, desc = result
        flowables.append(Paragraph(f"<b>Prediction Result:</b> {level}", styles['Heading2']))
        flowables.append(Paragraph(f"<b>Confidence:</b> {score}%", styles['Normal']))
        flowables.append(Paragraph(desc, styles['Normal']))
        flowables.append(Spacer(1, 12))
    elif mode == "compare":
        r1, r2 = result
        flowables.append(Paragraph("<b>Comparison Report</b>", styles['Heading2']))
        flowables.append(Paragraph(f"Voice 1 → {r1[0]} ({r1[1]}%)", styles['Normal']))
        flowables.append(Paragraph(f"Voice 2 → {r2[0]} ({r2[1]}%)", styles['Normal']))
        flowables.append(Spacer(1, 12))

    # Add plots
    for plot in plots:
        img_buffer = io.BytesIO()
        plot.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        flowables.append(Image(img_buffer, width=400, height=200))
        flowables.append(Spacer(1, 12))

    doc.build(flowables)
    buffer.seek(0)
    return buffer


# =========================================
# EXTRA VISUALIZATIONS
# =========================================
def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="steelblue")
    ax.set(title=title)
    return fig


def plot_spectrogram(y, sr, title="Spectrogram"):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap="magma", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title=title)
    return fig


def plot_radar(features1, features2=None):
    labels = list(features1.keys())
    values1 = list(features1.values())
    values1_norm = (np.array(values1) - np.min(values1)) / (np.max(values1) - np.min(values1) + 1e-6)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values1_norm.tolist() + [values1_norm[0]],
                                  theta=labels + [labels[0]],
                                  fill='toself', name='Voice 1', line=dict(color="royalblue")))

    if features2:
        values2 = list(features2.values())
        values2_norm = (np.array(values2) - np.min(values2)) / (np.max(values2) - np.min(values2) + 1e-6)
        fig.add_trace(go.Scatterpolar(r=values2_norm.tolist() + [values2_norm[0]],
                                      theta=labels + [labels[0]],
                                      fill='toself', name='Voice 2', line=dict(color="crimson")))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig


# =========================================
# MAIN APP
# =========================================
st.title("🧠 NeuroTap – Parkinson's Detection from Voice")
option = st.radio("Choose Input Method:", ["📂 Upload File", "🎤 Record with Laptop/Phone Mic", "🆚 Compare Two Voices"])

raw = None

# --- Upload File ---
if option == "📂 Upload File":
    voice_file = st.file_uploader("Upload your voice sample (.wav or .mp3)", type=["wav", "mp3"])
    if st.button("🔍 Predict from File"):
        if voice_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                data, samplerate = sf.read(voice_file)
                sf.write(temp.name, data, samplerate)
                features, y, sr, mfcc = extract_voice_features(temp.name)

            level, score, description = predict_voice(features)
            st.subheader(level)
            st.progress(score)
            st.markdown(f"### 📊 Confidence: *{score}%*")
            st.info(description)

            # Plots
            figs = [plot_waveform(y, sr), plot_spectrogram(y, sr)]
            st.pyplot(figs[0]); st.pyplot(figs[1])

            # Radar
            features_dict = {
                "Zero-Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "Spectral Rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                "MFCC1": np.mean(mfcc[0]),
                "MFCC2": np.mean(mfcc[1])
            }
            st.plotly_chart(plot_radar(features_dict))

            pdf_buffer = generate_pdf_report((level, score, description), figs, mode="single")
            st.download_button("📥 Download Report as PDF", data=pdf_buffer,
                               file_name="NeuroTap_Report.pdf", mime="application/pdf")
        else:
            st.warning("Please upload a voice sample.")

# --- Record Mic ---
elif option == "🎤 Record with Laptop/Phone Mic":
    st.write("Press the record button below:")
    audio_data = st_audiorec()
    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_data)
            temp.flush()
            features, y, sr, mfcc = extract_voice_features(temp.name)

        level, score, description = predict_voice(features)
        st.subheader(level)
        st.progress(score)
        st.markdown(f"### 📊 Confidence: *{score}%*")
        st.info(description)

        figs = [plot_waveform(y, sr), plot_spectrogram(y, sr)]
        st.pyplot(figs[0]); st.pyplot(figs[1])
        st.plotly_chart(plot_radar({"Zero-Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                                    "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                                    "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                                    "Spectral Rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                                    "MFCC1": np.mean(mfcc[0]), "MFCC2": np.mean(mfcc[1])}))

        st.audio(audio_data, format="audio/wav")
        pdf_buffer = generate_pdf_report((level, score, description), figs, mode="single")
        st.download_button("📥 Download Report as PDF", data=pdf_buffer,
                           file_name="NeuroTap_Report.pdf", mime="application/pdf")

# --- Compare Two Voices ---
elif option == "🆚 Compare Two Voices":
    file1 = st.file_uploader("Upload Voice Sample 1", type=["wav", "mp3"], key="f1")
    file2 = st.file_uploader("Upload Voice Sample 2", type=["wav", "mp3"], key="f2")
    if st.button("🔍 Compare Voices"):
        if file1 and file2:
            # Process file1
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp1:
                data1, sr1 = sf.read(file1)
                sf.write(temp1.name, data1, sr1)
                f1, y1, sr1, mfcc1 = extract_voice_features(temp1.name)
            r1 = predict_voice(f1)

            # Process file2
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp2:
                data2, sr2 = sf.read(file2)
                sf.write(temp2.name, data2, sr2)
                f2, y2, sr2, mfcc2 = extract_voice_features(temp2.name)
            r2 = predict_voice(f2)

            st.subheader("📊 Comparison Results")
            st.write(f"Voice 1 → {r1[0]} ({r1[1]}%)")
            st.write(f"Voice 2 → {r2[0]} ({r2[1]}%)")

            # Side-by-side plots
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_waveform(y1, sr1, "Voice 1 Waveform"))
                st.pyplot(plot_spectrogram(y1, sr1, "Voice 1 Spectrogram"))
            with col2:
                st.pyplot(plot_waveform(y2, sr2, "Voice 2 Waveform"))
                st.pyplot(plot_spectrogram(y2, sr2, "Voice 2 Spectrogram"))

            # Radar
            fdict1 = {"Zero-Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y1)),
                      "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y1, sr=sr1)),
                      "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y1, sr=sr1)),
                      "Spectral Rolloff": np.mean(librosa.feature.spectral_rolloff(y=y1, sr=sr1)),
                      "MFCC1": np.mean(mfcc1[0]), "MFCC2": np.mean(mfcc1[1])}
            fdict2 = {"Zero-Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y2)),
                      "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y2, sr=sr2)),
                      "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y2, sr=sr2)),
                      "Spectral Rolloff": np.mean(librosa.feature.spectral_rolloff(y=y2, sr=sr2)),
                      "MFCC1": np.mean(mfcc2[0]), "MFCC2": np.mean(mfcc2[1])}
            st.plotly_chart(plot_radar(fdict1, fdict2))

            pdf_buffer = generate_pdf_report((r1, r2), [plot_waveform(y1, sr1), plot_waveform(y2, sr2)], mode="compare")
            st.download_button("📥 Download Comparison Report as PDF", data=pdf_buffer,
                               file_name="NeuroTap_Comparison_Report.pdf", mime="application/pdf")
        else:
            st.warning("Please upload both voice samples.")