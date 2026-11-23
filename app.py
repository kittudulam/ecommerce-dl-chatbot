"""Interactive Streamlit Web Application for Deep Learning E-commerce Assistant."""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from PIL import Image

from src.chatbot_engine import EcommerceChatbotEngine

# Page configuration
st.set_page_config(
    page_title="ShopSense AI | E-commerce Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.4rem;
    }
    .conf-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #D1FAE5;
        color: #065F46;
    }
    .fallback-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #FEE2E2;
        color: #991B1B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    """Cache and load the inference engine."""
    return EcommerceChatbotEngine(
        models_dir=str(ROOT_DIR / "saved_models"),
        default_model="bilstm",
        confidence_threshold=0.50,
    )


engine = get_engine()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1557821552-17105176677c?w=500&auto=format&fit=crop&q=60",
        use_container_width=True,
    )
    st.markdown("## ⚙️ Model & Diagnostics")

    # 1. Model Switcher
    model_choice = st.selectbox(
        "🧠 Active Deep Learning Architecture",
        options=["Bidirectional LSTM (Bi-LSTM)", "Standard LSTM"],
        index=0,
        help="Select which trained neural network architecture generates predictions.",
    )
    active_model_key = "bilstm" if "Bi-LSTM" in model_choice else "lstm"

    # 2. Confidence Threshold Slider
    threshold = st.slider(
        "🎯 Fallback Confidence Threshold (τ)",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Queries classified with confidence lower than this threshold will trigger a fallback message.",
    )

    st.divider()

    # 3. Model Benchmark Overview
    st.markdown("### 📊 Architecture Benchmark")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bi-LSTM Acc", "57.69%", "+34.61%")
    with col2:
        st.metric("LSTM Acc", "23.08%", "Baseline")

    st.caption("⚡ Average Inference Latency: **~6.4 ms**")

    # 4. Evaluation Plots Expander
    with st.expander("📈 View Training Curves"):
        curves_path = ROOT_DIR / "reports" / "training_curves_comparison.png"
        if curves_path.exists():
            st.image(str(curves_path), caption="Accuracy & Loss Curves", use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate plots.")

    with st.expander("🧩 View Confusion Matrices"):
        cm_path = ROOT_DIR / "reports" / "confusion_matrices_comparison.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Side-by-side Confusion Matrices", use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate plots.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# MAIN CHAT APPLICATION
# =========================================================
st.markdown("<div class='main-title'>🛍️ ShopSense AI Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Deep Learning-powered customer support assistant trained on bidirectional sequence patterns.</div>",
    unsafe_allow_html=True,
)

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Welcome to ShopSense. I can help with product recommendations, order tracking, returns, cancellations, and payment support. What can I do for you today?",
            "intent": "Greeting",
            "confidence": 100.0,
            "is_fallback": False,
            "latency": 0.0,
            "probabilities": {},
        }
    ]

# Display quick suggestion prompt chips
st.markdown("**Quick Sample Questions:**")
quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

prompt_to_send = None

with quick_col1:
    if st.button("💻 Laptop under ₹50,000", use_container_width=True):
        prompt_to_send = "Suggest a good laptop under ₹50,000."
with quick_col2:
    if st.button("📦 Where is my order?", use_container_width=True):
        prompt_to_send = "Where is my order?"
with quick_col3:
    if st.button("❌ Cancel my order", use_container_width=True):
        prompt_to_send = "I want to cancel my order."
with quick_col4:
    if st.button("💳 Payment failed", use_container_width=True):
        prompt_to_send = "My payment failed. What should I do?"

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("intent") and msg.get("confidence") is not None:
            # Badges
            intent_label = msg["intent"]
            conf = msg["confidence"]
            is_fb = msg.get("is_fallback", False)
            latency = msg.get("latency", 0.0)

            badge_html = f"""
            <div style='margin-top: 0.4rem;'>
                <span class='intent-badge'>🎯 Intent: {intent_label}</span>
                <span class='{'fallback-badge' if is_fb else 'conf-badge'}'>
                    {'⚠️ Fallback' if is_fb else '📊 Conf'}: {conf:.1f}%
                </span>
                <span style='font-size: 0.75rem; color: #6B7280; margin-left: 0.4rem;'>⏱️ {latency}ms</span>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

            # Optional probability distribution expander
            probs = msg.get("probabilities", {})
            if probs:
                with st.expander("🔍 View Intent Probability Distribution"):
                    df_probs = pd.DataFrame(
                        list(probs.items())[:6], columns=["Intent", "Probability"]
                    )
                    df_probs["Percentage (%)"] = df_probs["Probability"] * 100
                    st.bar_chart(
                        data=df_probs.set_index("Intent")["Percentage (%)"],
                        horizontal=True,
                    )

# Capture user query either from quick chips or standard chat input
user_input = st.chat_input("Ask about products, orders, returns, payments...")
active_query = prompt_to_send or user_input

if active_query:
    # 1. Append user query to chat history
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Run Deep Learning inference
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent..."):
            result = engine.get_response(
                query=active_query,
                model_name=active_model_key,
                threshold=threshold,
            )

        st.markdown(result["response"])

        # Display metadata badges
        badge_html = f"""
        <div style='margin-top: 0.4rem;'>
            <span class='intent-badge'>🎯 Intent: {result['predicted_intent']}</span>
            <span class='{'fallback-badge' if result['is_fallback'] else 'conf-badge'}'>
                {'⚠️ Fallback' if result['is_fallback'] else '📊 Conf'}: {result['confidence_percent']}%
            </span>
            <span style='font-size: 0.75rem; color: #6B7280; margin-left: 0.4rem;'>⏱️ {result['latency_ms']}ms ({result['model_used']})</span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        with st.expander("🔍 View Intent Probability Distribution"):
            df_probs = pd.DataFrame(
                list(result["probabilities"].items())[:6],
                columns=["Intent", "Probability"],
            )
            df_probs["Percentage (%)"] = df_probs["Probability"] * 100
            st.bar_chart(
                data=df_probs.set_index("Intent")["Percentage (%)"],
                horizontal=True,
            )

    # 3. Save assistant message to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "intent": result["predicted_intent"],
            "confidence": result["confidence_percent"],
            "is_fallback": result["is_fallback"],
            "latency": result["latency_ms"],
            "probabilities": result["probabilities"],
        }
    )
