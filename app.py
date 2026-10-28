import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    CLASSES, CLASS_NAMES, CLASS_RISK_LEVEL, CLASS_DESCRIPTIONS,
    NOT_DETECTED_CLASS, DEFAULT_CONFIDENCE_THRESHOLD, MODELS_DIR
)
from src.inference import SkinDiseaseClassifier

# Page Configuration - Clean, Bright, Wide
st.set_page_config(
    page_title="SkinDisease Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bright, Modern Medical Dashboard Styling
st.markdown("""
<style>
    /* App background & typography */
    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.05);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    
    /* KPI Card styling */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 600;
    }

    /* Result cards */
    .result-box {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    return SkinDiseaseClassifier()

@st.cache_data
def load_metrics():
    eval_path = MODELS_DIR / "evaluation_metrics.json"
    hist_path = MODELS_DIR / "training_history.json"
    eval_metrics = json.load(open(eval_path)) if eval_path.exists() else None
    history = json.load(open(hist_path)) if hist_path.exists() else None
    return eval_metrics, history

classifier = load_classifier()
eval_metrics, training_history = load_metrics()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dermatology.png", width=64)
    st.markdown("### **SkinDisease Classifier**")
    st.caption("HAM10000 Transfer Learning • MobileNetV3")

    st.markdown("---")
    st.subheader("⚙️ Detection Controls")
    
    threshold = st.slider(
        "Threshold for 'Not Detected'",
        min_value=0.30,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="If the model's confidence is below this cutoff or if no distinct lesion contour is found, the image is marked as 'Not Detected / Inconclusive'."
    )
    
    cam_alpha = st.slider(
        "Grad-CAM Heatmap Opacity",
        min_value=0.2,
        max_value=0.8,
        value=0.45,
        step=0.05,
        help="Adjust the overlay visibility of the visual explainability heatmap."
    )

    st.markdown("---")
    st.subheader("ℹ️ System Status")
    if classifier.has_trained_weights:
        st.success("✅ **Fine-Tuned Weights Active**")
        st.caption("Trained on balanced HAM10000 dataset")
    else:
        st.info("ℹ️ Running in feature extractor mode")

    st.markdown("""
    **Categories (7 Classes + 8th OOD):**
    * `akiec`: Actinic Keratoses
    * `bcc`: Basal Cell Carcinoma
    * `bkl`: Benign Keratosis
    * `df`: Dermatofibroma
    * `mel`: Melanoma
    * `nv`: Melanocytic Nevi
    * `vasc`: Vascular Lesions
    * `not_detected`: Inconclusive / Normal Skin
    """)

# --- MAIN HEADER ---
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🩺 SkinDisease Classifier</div>
    <div class="hero-subtitle">Deep learning dermatological lesion classification with Grad-CAM explainability, confidence gating, and performance verification.</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_metrics = st.tabs(["🔍 **Lesion Diagnosis**", "📊 **Model Performance & Evaluation Matrices**"])

# ==============================================================================
# TAB 1: LESION DIAGNOSIS
# ==============================================================================
with tab_predict:
    st.markdown("#### 📤 Upload Skin Lesion Image")
    uploaded_file = st.file_uploader(
        "Upload a close-up photo of the skin lesion (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="For accurate evaluation, provide a clear, focused, well-lit image of the lesion."
    )

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        
        with st.spinner("Analyzing dermatological features & generating Grad-CAM..."):
            result = classifier.predict(
                image_to_process,
                confidence_threshold=threshold,
                generate_cam=True
            )

        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        risk = result["risk_info"]
        probs = result["probabilities"]
        is_detected = result["is_detected"]

        st.markdown("---")
        st.markdown("### 📋 Diagnostic Assessment")

        # Top Status Alert Banner
        if pred_class == NOT_DETECTED_CLASS:
            st.warning(
                f"### ⚠️ **{result['class_name']}**\n\n"
                f"**Clinical Status:** No distinct skin lesion was detected with sufficient confidence "
                f"(cutoff: {threshold*100:.0f}%). The image appears uniform, clear, or out-of-distribution."
            )
        elif risk["severity"] == "danger":
            st.error(
                f"### 🚨 **{result['class_name']}** — Risk Level: {risk['risk']}\n\n"
                f"**Clinical Action:** {risk['action']}"
            )
        elif risk["severity"] == "warning":
            st.warning(
                f"### ⚠️ **{result['class_name']}** — Risk Level: {risk['risk']}\n\n"
                f"**Clinical Action:** {risk['action']}"
            )
        else:
            st.success(
                f"### ✅ **{result['class_name']}** — Risk Level: {risk['risk']}\n\n"
                f"**Clinical Action:** {risk['action']}"
            )

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Disease", pred_class.upper())
        with m2:
            st.metric("Diagnosis Confidence", f"{confidence*100:.1f}%")
        with m3:
            st.metric("Clinical Risk Category", risk["risk"])
        with m4:
            st.metric("8th-Class Filter", "Passed" if is_detected else "Filtered (Not Detected)")

        # Visual Explainability (Grad-CAM)
        st.markdown("#### 🔬 Visual Explainability (Grad-CAM)")
        st.caption("Gradient-weighted Class Activation Mapping reveals the specific pigment networks and lesion borders that drove the model's decision:")

        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(result["original_image"], caption="1. Original Skin Image", use_container_width=True)
        with col_img2:
            if result["gradcam_heatmap"] is not None:
                st.image(result["gradcam_heatmap"], caption="2. Activation Heatmap", use_container_width=True, clamp=True)
        with col_img3:
            if result["gradcam_overlay"] is not None:
                custom_overlay = classifier.grad_cam.overlay_heatmap(
                    result["original_image"],
                    result["gradcam_heatmap"],
                    alpha=cam_alpha
                )
                st.image(custom_overlay, caption="3. Heatmap Overlay on Lesion", use_container_width=True)

        # Multi-Class Probability Breakdown
        st.markdown("#### 📈 Multi-Class Probability Distribution")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        class_labels = [f"{CLASS_NAMES[k]} ({k.upper()})" for k, _ in sorted_probs]
        score_vals = [v * 100 for _, v in sorted_probs]

        colors = []
        for k, _ in sorted_probs:
            sev = CLASS_RISK_LEVEL[k]["severity"]
            if sev == "danger":
                colors.append("#EF4444")
            elif sev == "warning":
                colors.append("#F59E0B")
            else:
                colors.append("#10B981")

        fig = go.Figure(go.Bar(
            x=score_vals,
            y=class_labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in score_vals],
            textposition='outside'
        ))

        fig.add_vline(
            x=threshold * 100,
            line_width=2,
            line_dash="dash",
            line_color="#64748B",
            annotation_text=f"Detection Cutoff ({threshold*100:.0f}%)",
            annotation_position="top right"
        )

        fig.update_layout(
            xaxis_title="Confidence Probability (%)",
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(l=20, r=40, t=20, b=20),
            xaxis=dict(range=[0, 110]),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📚 Clinical Information & Disease Details"):
            st.markdown(f"**About {result['class_name']}:**")
            st.info(result["description"])
            st.markdown("""
            **ABCDE Self-Check Guidelines:**
            * **A - Asymmetry:** One half doesn't match the other.
            * **B - Border:** Edges are irregular, scalloped, or blurred.
            * **C - Color:** Color variation (shades of brown, black, red, or white).
            * **D - Diameter:** Lesion diameter is larger than 6mm.
            * **E - Evolving:** Rapid changes in size, shape, color, or symptoms.
            """)
    else:
        st.info("👆 Please upload an image above to begin automatic dermatological analysis.")

# ==============================================================================
# TAB 2: MODEL PERFORMANCE & EVALUATION MATRICES
# ==============================================================================
with tab_metrics:
    st.markdown("### 📊 Model Evaluation, Accuracy & Loss Metrics")
    st.caption("Quantitative performance metrics evaluated on the balanced hold-out validation dataset.")

    if eval_metrics is not None:
        # Top KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Validation Accuracy</div>
                <div class="kpi-value">{eval_metrics['accuracy']*100:.1f}%</div>
                <div class="kpi-sub">Across 7 Balanced Classes</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Macro Precision</div>
                <div class="kpi-value">{eval_metrics['macro_precision']*100:.1f}%</div>
                <div class="kpi-sub">Balanced Class Mean</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Validation Loss</div>
                <div class="kpi-value">{eval_metrics['val_loss']:.3f}</div>
                <div class="kpi-sub">Cross-Entropy Loss</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Macro F1-Score</div>
                <div class="kpi-value">{eval_metrics['macro_f1']*100:.1f}%</div>
                <div class="kpi-sub">Harmonic Mean Metric</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row: Confusion Matrix + Classification Report
        col_cm, col_report = st.columns([1.1, 1.0])

        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix")
            st.caption("Ground Truth (rows) vs. Predicted Disease Class (columns):")
            
            cm = np.array(eval_metrics["confusion_matrix"])
            labels = [c.upper() for c in eval_metrics["classes"]]
            
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Count")
            )
            fig_cm.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_report:
            st.markdown("#### 📋 Per-Class Performance Matrix")
            st.caption("Detailed precision, recall (sensitivity), and F1-score per lesion category:")
            
            report = eval_metrics["classification_report"]
            table_data = []
            for c in eval_metrics["classes"]:
                if c in report:
                    table_data.append({
                        "Class": c.upper(),
                        "Condition Name": CLASS_NAMES[c].split(" (")[0],
                        "Precision": f"{report[c]['precision']*100:.1f}%",
                        "Recall": f"{report[c]['recall']*100:.1f}%",
                        "F1-Score": f"{report[c]['f1-score']*100:.1f}%",
                        "Samples": int(report[c]['support'])
                    })

            df_report = pd.DataFrame(table_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)

        # Row: Loss and Accuracy Over Epochs
        if training_history is not None:
            st.markdown("---")
            st.markdown("#### 📉 Training & Validation Learning Curves")
            st.caption("Convergence history over 10 transfer learning epochs on Apple Silicon GPU:")

            col_loss, col_acc = st.columns(2)
            epochs = [h["epoch"] for h in training_history]
            train_loss = [h["train_loss"] for h in training_history]
            val_loss = [h["val_loss"] for h in training_history]
            train_acc = [h["train_acc"] * 100 for h in training_history]
            val_acc = [h["val_acc"] * 100 for h in training_history]
            val_f1 = [h["val_f1"] * 100 for h in training_history]

            with col_loss:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss", line=dict(color="#2563EB", width=2)))
                fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss", line=dict(color="#EF4444", width=2, dash="dot")))
                fig_loss.update_layout(
                    title="Cross-Entropy Loss Curve",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    legend=dict(orientation="h", y=1.1, x=0.7)
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_acc:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers", name="Train Acc (%)", line=dict(color="#10B981", width=2)))
                fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Val Acc (%)", line=dict(color="#0284C7", width=2)))
                fig_acc.add_trace(go.Scatter(x=epochs, y=val_f1, mode="lines+markers", name="Val F1 (%)", line=dict(color="#F59E0B", width=2, dash="dash")))
                fig_acc.update_layout(
                    title="Accuracy & F1-Score Progression",
                    xaxis_title="Epoch",
                    yaxis_title="Percentage (%)",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    legend=dict(orientation="h", y=1.1, x=0.5)
                )
                st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.warning("Evaluation metrics have not been generated yet. Please run evaluation.")

# --- FOOTER & DISCLAIMER ---
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ Medical & Research Disclaimer:</b><br>
    This software is an AI research prototype built on the HAM10000 dataset for assistive screening and demonstration purposes only. 
    It is <b>NOT</b> a certified diagnostic medical device and cannot replace professional clinical consultation or histopathological biopsy by a licensed dermatologist.
</div>
""", unsafe_allow_html=True)
