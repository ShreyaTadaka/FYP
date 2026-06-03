import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATASET_PATH = "data/AccentDataset"
ACCENTS = ["American_accent", "British_accent", "Indian_accent"]

LABEL_MAP = {
    "American_accent": 0,
    "British_accent": 1,
    "Indian_accent": 2
}

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    if len(audio) < 2048:
        return None
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X, y = [], []

for accent in ACCENTS:
    folder = os.path.join(DATASET_PATH, accent)
    for file in os.listdir(folder):
        if file.endswith((".wav", ".mp3")):
            features = extract_mfcc(os.path.join(folder, file))
            if features is not None:
                X.append(features)
                y.append(LABEL_MAP[accent])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/accent_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Accent model trained & saved")
