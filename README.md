# 🛍️ ShopSense AI: E-commerce Assistant Chatbot Using Deep Learning

A production-ready Deep Learning intent classification and conversational chatbot system for e-commerce platforms. Built from scratch with **TensorFlow/Keras**, **NLTK**, and **Streamlit**, featuring comparative benchmarking between **Standard LSTM** and **Bidirectional LSTM (Bi-LSTM)** neural network architectures.

---

## 📌 Project Overview
E-commerce customer queries exhibit natural linguistic variation, informal vocabulary, and bidirectional contextual dependencies (e.g., *"Suggest a good laptop under ₹50,000"*, *"Where is my order?"*, *"My payment failed"*). 

This application uses Natural Language Processing (NLP) to normalize, tokenize, and lemmatize queries, followed by deep sequence models that map queries to intent categories and provide contextual responses, confidence scores, and fallback handling for unsupported requests.

---

## 🏆 Key Experimental Results

| Metric | Standard LSTM | Bidirectional LSTM (Winner) | Improvement |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **23.08%** | **57.69%** | **+34.61% (2.5x)** |
| **Macro Precision** | 0.1400 | **0.5167** | **+269%** |
| **Macro Recall** | 0.2000 | **0.5333** | **+166%** |
| **Macro F1-Score** | 0.1643 | **0.5086** | **+209%** |
| **Weighted F1-Score** | 0.1896 | **0.5458** | **+188%** |
| **Inference Latency** | 4.84 ms | **6.44 ms** | Sub-10ms real-time |

> **Conclusion**: Bidirectional LSTM captures semantic relationships across both past and future words simultaneously, dramatically outperforming uni-directional LSTM on nuanced shopping intent classification.

---

## 🎯 Supported E-commerce Intents
1. `Product_Recommendation`: Budget/feature-based product suggestions (e.g., *"Suggest a laptop under 50k"*).
2. `Product_Search`: Category and brand browsing (e.g., *"Show wireless headphones"*).
3. `Product_Details`: Technical specifications, dimensions, and warranty info.
4. `Order_Tracking`: Courier status and delivery timelines (e.g., *"Where is my order?"*).
5. `Order_Cancellation`: Revoking orders before dispatch.
6. `Returns`: Return windows and doorstep pickup scheduling.
7. `Refunds`: Refund status, timelines, and bank/UPI transfers.
8. `Payment_Issues`: Declined cards, double debits, and failed checkouts.
9. `Customer_Support`: Human assistance and helpline contacts.
10. `Greeting`: Welcome messages and pleasantries.

---

## 🏗️ Repository Architecture

```text
faq-dl-chatbot/
├── data/
│   └── ecommerce_intents.json     # Labelled dataset (129 variations across 10 intents)
├── src/
│   ├── __init__.py
│   ├── preprocessor.py            # NLTK tokenization, lemmatization & sequence vectorizer
│   ├── models.py                  # Keras LSTM & Bi-LSTM neural architectures
│   ├── train.py                   # Stratified training pipeline with EarlyStopping
│   ├── evaluate.py                # Comparative evaluation, metrics & confusion matrices
│   └── chatbot_engine.py          # Real-time inference engine & fallback handling
├── saved_models/                  # Serialized Keras models, tokenizer, and encoders
├── reports/                       # Training curves and confusion matrix comparison plots
│   ├── training_curves_comparison.png
│   ├── confusion_matrices_comparison.png
│   └── evaluation_summary.json
├── app.py                         # Interactive Streamlit Web Application
├── REPORT.md                      # Comprehensive Final Lab Report
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repository-url>
cd faq-dl-chatbot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download NLTK Corpora
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3. Train the Models (Optional - Pre-trained weights included)
```bash
python src/train.py
```

### 4. Run Comparative Model Evaluation
```bash
python src/evaluate.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 Tech Stack
- **Language**: Python 3.11
- **Deep Learning**: TensorFlow 2.x / Keras 3.x
- **NLP**: NLTK (WordNetLemmatizer, Punkt)
- **Evaluation**: Scikit-Learn (Classification report, Confusion matrix)
- **Visualization**: Matplotlib, Seaborn
- **Frontend / Deployment**: Streamlit
