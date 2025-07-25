import os

# Update this path to your local directory
DATASET_PATH = 'Audio_Speech_Actors_01-24'

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

audio_files = []

for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if os.path.isdir(actor_path):
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                full_path = os.path.join(actor_path, filename)
                parts = filename.split('-')
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, 'unknown')

                audio_files.append({
                    "path": full_path,
                    "emotion": emotion,
                    "actor": actor_folder
                })

# print(f"Found {len(audio_files)} audio files.")
# print("Example:", audio_files[0])

import librosa
import numpy as np


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # default sampling rate

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_scaled = np.mean(chroma.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_scaled = np.mean(mel.T, axis=0)

    # Combine all features
    combined = np.hstack([mfcc_scaled, chroma_scaled, mel_scaled])
    return combined

X = []  # features
y = []  # labels (emotion)

for item in audio_files:
    features = extract_features(item["path"])
    X.append(features)
    y.append(item["emotion"])

import numpy as np

X = np.array(X)
y = np.array(y)

# Save them
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

