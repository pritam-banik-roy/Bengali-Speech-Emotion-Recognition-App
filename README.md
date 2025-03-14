# Bengali Speech Emotion Recognition

## Overview
This project focuses on recognizing emotions from Bengali speech using deep learning techniques. The model processes audio signals and classifies them into different emotion categories.

## Features
- Bengali speech dataset processing
- Feature extraction using MFCCs
- Deep learning model for emotion classification
- Streamlit-based user interface for real-time inference

## Dataset
The project uses a curated Bengali speech emotion dataset containing labeled audio samples representing emotions such as:
- Happy
- Sad
- Angry
- Surprise
- Neutral

## Installation
Clone the repository:
```bash
git clone (https://github.com/pritam-banik-roy/Bengali-Speech-Emotion-Recognition-App.git)
cd bengali-speech-emotion-recognition
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Train the Model:**
   ```bash
   python train.py
   ```
   This will train the deep learning model on the dataset.

2. **Test the Model:**
   ```bash
   python test.py
   ```
   Evaluate the model performance on test data.

3. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```
   This starts the Streamlit-based UI for real-time emotion recognition from speech.

## Model Architecture
- **Feature Extraction:** MFCCs from audio samples
- **Classifier:** CNN + LSTM for capturing temporal and spatial patterns
- **Optimization:** Adam optimizer with categorical cross-entropy loss

## Streamlit Web App
The UI allows users to:
- Upload an audio file
- Record and analyze live speech
- View predicted emotion with confidence score

## Results
The model achieves an accuracy of **X%** on the test dataset (to be updated based on experiments).

## Future Enhancements
- Increase dataset size for better generalization
- Experiment with transformer-based architectures
- Deploy as a cloud-based API for wider accessibility

## Contributors
- [PRITAM BANIK ROY](https://github.com/pritam-banik-roy)



