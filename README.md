
---

## ✅ 2. Speech Emotion Recognition (SER) – `README_SER.md`

```markdown
# Speech Emotion Recognition (SER)

## 📌 Layihənin Məqsədi
Bu modelin əsas funksiyası danışıq səsindən emosiyaları tanımaqdır. Model "happy", "sad", "angry", "neutral" və digər emosiyaları səsdəki əlamətlərə əsaslanaraq təyin edir.

---

## 🧠 Model Mimarisi
- `MFCC` əsaslı xüsusiyyət çıxarılması
- `StandardScaler` ilə normalizasiya
- Klassifikasiya üçün `ML` modeli (SVC, RandomForest və ya CNN-LSTM)
- Label Encoding və Inverse Transform

---

## 🛠️ Texnologiyalar və Kitabxanalar
- `scikit-learn`
- `librosa`
- `joblib`
- `numpy`
- `sounddevice` (real-time üçün)


## 📁 Fayl Quruluşu
SER/
├── extract_features.py # MFCC çıxarılması
├── train_model.py # ML modelin öyrədilməsi
├── test_model.py # Modelin test edilməsi
├── real_time_ser.py # Mikrofon ilə real zamanlı test
├── emotion_model.pkl # Yadda saxlanılmış model
├── scaler.pkl # Yadda saxlanılmış scaler
├── label_encoder.pkl # Label Encoder
└── README_SER.md

---

## ▶️ İstifadə
### Modelin Öyrədilməsi:
```bash
python train_model.py --dataset="./data/emotions.csv"
##💡 Gələcək Planlar
- Neural modelə keçid (CNN, LSTM, wav2vec2.0)
- İnkişaf etmiş xüsusiyyət mühəndisliyi
- Dilə spesifik model adaptasiyası (azərbaycan dili üçün incə tənzimləmə)
