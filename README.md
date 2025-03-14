# 🎙️ Bengali Speech Emotion Recognition

## 📌 Overview
This project focuses on recognizing emotions from **Bengali speech** using deep learning techniques. The model processes **audio signals** and classifies them into different emotion categories.

## 🚀 Features
✅ Bengali speech dataset processing  
✅ Feature extraction using **MFCCs**  
✅ **Deep learning model** for emotion classification  
✅ **Streamlit-based UI** for real-time inference  
✅ Supports **audio file upload & live recording**  

## 🎭 Emotion Categories
The dataset includes audio samples labeled with the following emotions:
- 😃 **Happy**
- 😢 **Sad**
- 😡 **Angry**
- 😲 **Surprise**
- 😐 **Neutral**

## 📂 Dataset
This project uses a curated **Bengali Speech Emotion Dataset**, with each sample labeled according to its emotional category. The dataset consists of high-quality **.wav** files.

## 🔧 Installation
1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/pritam-banik-roy/Bengali-Speech-Emotion-Recognition-App.git
cd Bengali-Speech-Emotion-Recognition-App
```
2️⃣ **Install dependencies:**  
```bash
pip install -r requirements.txt
```

## 📌 Usage
### 🔹 Train the Model:
```bash
python train.py
```
This will train the deep learning model on the dataset.

### 🔹 Test the Model:
```bash
python test.py
```
Evaluates model performance on test data.

### 🔹 Run the Web App:
```bash
streamlit run app.py
```
Launches the **Streamlit-based UI** for real-time emotion recognition from speech.

## 🏗 Model Architecture
🔹 **Feature Extraction:** Uses **MFCCs** from audio samples.  
🔹 **Classifier:** **CNN + LSTM** to capture **temporal** and **spatial** patterns.  
🔹 **Optimization:** Adam optimizer with categorical cross-entropy loss.  
🔹 **Activation:** Softmax for multi-class classification.  

## 🎨 Streamlit Web App
The UI allows users to:
- 🎤 **Upload an audio file**
- 🔴 **Record & analyze live speech**
- 📊 **View predicted emotion with confidence score**
- 🚀 **Real-time results** with an intuitive interface

## 📊 Results
The model achieves an accuracy of **94%** on the test dataset.

## 📈 Future Enhancements
🚀 Increase dataset size for better generalization  
🚀 Experiment with **transformer-based architectures**  
🚀 Deploy as a **cloud-based API** for broader accessibility  
🚀 Add **real-time spectrogram visualization** in UI  

## 👨‍💻 Contributors
📌 **[PRITAM BANIK ROY](https://github.com/pritam-banik-roy)**  
For any issues, feel free to **raise an issue** or **contribute to this repository**! ✨  

---
⭐ If you found this project useful, give it a **star** on GitHub! ⭐
