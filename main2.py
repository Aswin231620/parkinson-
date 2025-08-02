        import streamlit as st
        import numpy as np
        import pickle
        import librosa
        import soundfile as sf
        import tempfile

        # Load model
        with open("voice_model.pkl", "rb") as f:
            model = pickle.load(f)


        # Extract MFCC
        def extract_voice_features(uploaded_file):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                    data, samplerate = sf.read(uploaded_file)
                    sf.write(temp.name, data, samplerate)
                    y, sr = librosa.load(temp.name, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                return mfcc_mean
            except Exception as e:
                st.error(f"Error extracting MFCC: {e}")
                return None


        # Predict risk
        def predict_voice_with_confidence(mfcc_vector):
            if mfcc_vector is None:
                return "Error", 0, "Could not process voice file."

            proba = model.predict_proba([mfcc_vector])[0][1]
            percent = round(proba * 100)

            if percent < 20:
                level = "🟩 Very Low Risk"
                desc = "Your voice shows no sign of Parkinson’s."
            elif percent < 40:
                level = "🟩 Low Risk"
                desc = "Minor voice variations found — healthy."
            elif percent < 60:
                level = "🟨 Monitor"
                desc = "Some voice traits overlap. Recheck suggested."
            elif percent < 80:
                level = "🟧 Moderate Ris"
                desc = "Speech patterns suggest possible early signs."
            else:
                level = "🟥 High Risk "
                desc = "Strong vocal patterns linked to Parkinson’s. Please consult a doctor."

            return level, percent, desc


        # UI
        st.title("🧠 NeuroTap – Parkinson's Detection from Voice")
        voice_file = st.file_uploader("🎙 Upload your voice sample (.wav or .mp3)",
                                      type=["wav", "mp3"])

        if st.button("🔍 Predict Parkinson’s Risk"):
            if voice_file is not None:
                features = extract_voice_features(voice_file)
                level, score, description = predict_voice_with_confidence(features)
                st.subheader(f"{level}")
                st.write(f"📊 Confidence: {score}%")
                st.info(description)
            else:
                st.warning("Please upload a voice sample.")
