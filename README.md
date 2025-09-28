# 🧠 Parkinson's Disease Detection App

An AI-powered web application that analyzes voice and other data to help detect early signs of Parkinson's disease.

## 🚀 Features
- Upload audio or voice feature data and get prediction results.
- Multiple trained ML models (Random Forest, XGBoost, SVM, LightGBM).
- Detailed model reports and ROC curves.
- User-friendly Streamlit UI.

## 📁 Project Structure
- `app.py` – Main Streamlit web app.
- `model_reports/` – Trained models and metrics visualizations.
- `parkinsons (2)/` – Datasets used for training.
- `train_model.py` – Script to train and save models.

## 🧰 Installation

```bash
git clone https://github.com/Aswin231620/parkinson-.git
cd parkinson-
pip install -r requirements.txt
streamlit run app.py
