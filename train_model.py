import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
df = pd.read_csv("parkinsons_data.csv")

X = df.drop(columns=["name", "status"])
y = df["status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = "voice_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Confirm it's saved
if os.path.exists(model_path):
    print(f"✅ Model successfully saved as: {model_path}")
else:
    print("❌ Model was NOT saved.")
