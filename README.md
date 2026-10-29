# 🩺 SkinDisease Classifier

A production-ready deep learning application and interactive **Streamlit** web app for dermatological skin lesion classification based on the **HAM10000** benchmark dataset.

---

## 🌟 Key Features

1. **Dedicated Architecture — SkinDisease Classifier**:
   - **Backbone:** MobileNetV3-Large fine-tuned using transfer learning on a strictly balanced dataset (100 images per class across all 7 diagnostic categories).
   - **Model Footprint:** Fast, lightweight weights (~49 MB) that fit directly in GitHub commits and run with sub-second CPU inference on Streamlit Cloud.

2. **The 7 Diagnostic Categories + 8th "Not Detected" Category**:
   - `akiec`: Actinic Keratoses and Intraepithelial Carcinoma (Bowen's disease) [Pre-cancerous / Malignant]
   - `bcc`: Basal Cell Carcinoma [Malignant]
   - `bkl`: Benign Keratosis-like Lesions (solar lentigines, seborrheic keratoses) [Benign]
   - `df`: Dermatofibroma [Benign]
   - `mel`: Melanoma [High-Risk Malignant]
   - `nv`: Melanocytic Nevi (Common Moles) [Benign]
   - `vasc`: Vascular Lesions (cherry angiomas, pyogenic granulomas) [Benign]
   - **`not_detected` (8th Class)**: Dual-gate filter combining **lesion saliency / contrast checks** (rejecting clear unblemished skin) and a **calibrated confidence cutoff slider** (rejecting low-confidence / out-of-distribution inputs).

3. **Visual Explainability (Grad-CAM)**:
   - Built-in Gradient-weighted Class Activation Mapping (Grad-CAM) highlighting the exact pigment networks and lesion borders the neural network focused on.

4. **Integrated Model Performance & Evaluation Dashboard**:
   - **Real-time Evaluation Metrics:** Validation Accuracy (**`70.2%`**), Balanced Accuracy (**`70.3%`**), TTA Accuracy (**`71.5%`**), and Validation Loss (**`0.83`**).
   - **Interactive Confusion Matrix:** Color-coded matrix showing true vs. predicted counts across all 7 classes.
   - **Per-Class Metrics Table:** Granular precision, recall/sensitivity, F1-score, and support metrics.
   - **Learning Curves:** Plotly convergence charts displaying loss and accuracy across training epochs.

5. **Clean & Bright Streamlit UI**:
   - Bright, modern medical dashboard aesthetic with high readability and zero visual clutter.
   - Simple drag-and-drop file upload for fast diagnosis.

---

## ☁️ How to Host on Streamlit Community Cloud (Step-by-Step)

Streamlit Community Cloud provides **free, permanent 24/7 hosting** directly from your GitHub repository.

### Step 1: Push Your Code to GitHub
Your repository is hosted at:
👉 **`https://github.com/sahilssh/skindisease-classifier`**

### Step 2: Deploy on Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
2. Click the **"New app"** button.
3. Fill in the deployment form:
   * **Repository:** `sahilssh/skindisease-classifier`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **"Deploy!"**.

---

## 💻 Local Quick Start

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Launch the Streamlit web application
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Repository Structure

```text
skindisease-classifier/
├── app.py                      # Interactive Streamlit Web Application
├── src/
│   ├── config.py               # Dataset labels, risk mappings, hyperparameters
│   ├── dataset.py              # PyTorch Dataset + Albumentations augmentation pipeline
│   ├── model.py                # MobileNetV3 backbone + Grad-CAM implementation
│   ├── data_prep.py            # Group-aware stratified train/val/test splitting
│   ├── train.py                # Transfer learning training loop with Apple Silicon MPS
│   └── inference.py            # SkinDiseaseClassifier inference engine
├── scripts/
│   └── download_data.py        # Automated kagglehub dataset downloader
├── models/
│   ├── best_model.pth          # Fine-tuned model checkpoint (49 MB)
│   ├── evaluation_metrics.json # Confusion matrix, precision, recall, accuracy
│   └── training_history.json   # 10-epoch loss and accuracy metrics
├── requirements.txt            # Streamlit Cloud dependencies
└── README.md                   # Documentation
```

---

## ⚖️ Medical Disclaimer
This software is an artificial intelligence research and educational prototype. It is **not** a certified medical device and is **not** intended to replace clinical evaluation, dermoscopy, or biopsy performed by a board-certified dermatologist.
