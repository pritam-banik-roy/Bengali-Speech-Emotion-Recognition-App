import os
import librosa
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page config
st.set_page_config(page_title="Bengali Speech Emotion Recognition", layout="wide")

# Debugging: Check working directory
st.write(f"Current Working Directory: {os.getcwd()}")

# Define dataset path (Update if needed)
data_dir = "dataset"

# Check if dataset exists
if not os.path.exists(data_dir):
    st.error("Dataset directory not found. Ensure it's included in the project.")
else:
    st.success("Dataset directory found!")

# Emotion mapping
def extract_emotion_from_filename(filename):
    try:
        emotion_code = filename.split("-")[2]
        emotions = {"01": "Happy", "02": "Sad", "03": "Angry", "04": "Surprise", "05": "Neutral"}
        return emotions.get(emotion_code, "Unknown")
    except:
        return "Unknown"

# Feature extraction
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return None

# Load dataset
def load_data(data_dir):
    features, labels = [], []
    try:
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(data_dir, file_name)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(extract_emotion_from_filename(file_name))
        return np.array(features), np.array(labels)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return np.array([]), np.array([])

features, labels = load_data(data_dir)

# Check if data was loaded correctly
if features.shape[0] == 0 or labels.shape[0] == 0:
    st.error("No valid audio files found in dataset directory.")
    st.stop()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI
st.title("🎙️ Bengali Speech Emotion Recognition")
st.markdown("### Upload a WAV file to detect emotion")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
if uploaded_file is not None:
    file_name = uploaded_file.name
    emotion = extract_emotion_from_filename(file_name)
    
    # Save uploaded file temporarily
    temp_file = "temp.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features and predict
    features = extract_features(temp_file)
    if features is not None:
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]
        
        st.audio(temp_file, format="audio/wav")
        st.write(f"### 🎭 Predicted Emotion: {predicted_emotion}")
        st.write(f"### 📂 Extracted Emotion from the Audio File: {emotion}")

        # Plot waveform
        audio, sr = librosa.load(temp_file)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title("Waveform of Uploaded Audio")
        st.pyplot(fig)
    else:
        st.error("Error extracting features from uploaded file.")

# Display model accuracy and confusion matrix
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.sidebar.pyplot(fig)
