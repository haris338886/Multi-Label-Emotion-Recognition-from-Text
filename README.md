# 😊 Multi-Label Emotion Detection using DistilBERT

## 📌 Overview

This project focuses on detecting **multiple emotions** from text using **DistilBERT**, a lightweight and efficient transformer model. It uses the **GoEmotions** dataset from Google and is built using **PyTorch**. The model predicts combinations of emotions (e.g., joy + surprise) for each sentence.

---

## 🔍 Dataset & Preprocessing

- **Dataset**: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) – a human-annotated dataset with 27 emotion labels and over 58,000 Reddit comments.
- **Samples Used**: 1,000 samples (subset for rapid testing and prototyping)

### ✅ Preprocessing Steps

- Converted binary-labeled columns into multi-label lists
- Used `MultiLabelBinarizer` for multi-hot encoding of emotion labels
- Tokenized text using `DistilBertTokenizer` with padding and truncation
- Wrapped inputs in a custom **PyTorch Dataset** for training

---

## 🤖 Model Selection & Rationale

- **Model**: `DistilBertForSequenceClassification` (from Hugging Face Transformers)
- **Reason**: Balanced trade-off between speed and performance for text classification tasks
- **Task Type**: Multi-label classification (using sigmoid activation and BCE loss)

---

## ⚠️ Challenges Faced

| Challenge                | Solution                                            |
|--------------------------|-----------------------------------------------------|
| Label preprocessing      | Fixed via `MultiLabelBinarizer`                    |
| Emotion imbalance        | Acknowledged (future fix: class weights/oversampling) |
| Overfitting in testing   | Used only 1 epoch for prototyping                   |
| Lack of standard metrics | Added F1-score (micro) and Hamming Loss             |

---

## 📈 Results & Metrics

- **Hamming Loss**: ~0.08 (lower is better)
- **F1 Score (micro)**: ~0.72
- **Observation**:
  - Performs well on common emotions (joy, anger)
  - Some confusion between similar emotions (e.g., sadness vs disappointment)

---
