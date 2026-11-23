"""Real-time Inference Engine and Fallback Handling for E-commerce Chatbot."""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf

from src.preprocessor import IntentDataPipeline


class EcommerceChatbotEngine:
    """Production inference engine supporting both LSTM and Bi-LSTM models,
    intent classification, confidence scoring, and fallback handling.
    """

    FALLBACK_RESPONSES = [
        "I'm sorry, I'm not entirely confident I understood your shopping request. Could you please rephrase, or ask about product recommendations, orders, returns, or payments?",
        "I couldn't quite catch that. For specific assistance, you can try asking 'Where is my order?', 'Suggest laptops under 50000', or contact our 24/7 customer support at 1800-200-8899.",
        "I'm still learning! Could you please clarify your question, or would you like me to connect you with a live customer care representative?",
    ]

    def __init__(
        self,
        models_dir: str = "saved_models",
        default_model: str = "bilstm",
        confidence_threshold: float = 0.50,
    ):
        self.models_dir = Path(models_dir)
        self.confidence_threshold = confidence_threshold
        self.pipeline = IntentDataPipeline.load(str(self.models_dir))

        # Lazy load models
        self._models: Dict[str, tf.keras.Model] = {}
        self.active_model_name = default_model
        self._load_model(default_model)

    def _load_model(self, model_name: str) -> tf.keras.Model:
        """Load Keras model on demand and cache in memory."""
        name = model_name.lower()
        if name not in self._models:
            filename = "bilstm_model.keras" if "bi" in name else "lstm_model.keras"
            path = self.models_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            self._models[name] = tf.keras.models.load_model(path)
        return self._models[name]

    def set_active_model(self, model_name: str) -> None:
        """Switch between 'lstm' and 'bilstm' at runtime."""
        name = model_name.lower()
        self._load_model(name)
        self.active_model_name = name

    def get_response(
        self,
        query: str,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Classify user query and return intent, confidence score, and answer.
        
        Applies confidence thresholding to trigger fallback for out-of-domain
        or ambiguous questions.
        """
        chosen_model = model_name.lower() if model_name else self.active_model_name
        model = self._load_model(chosen_model)
        active_threshold = threshold if threshold is not None else self.confidence_threshold

        t0 = time.time()

        # 1. Vectorize query and check content-word domain overlap
        from src.preprocessor import tokenize_and_lemmatize
        stopwords = {
            "who", "what", "where", "when", "why", "how", "is", "are", "am", "was", "were",
            "the", "a", "an", "of", "in", "on", "at", "to", "for", "it", "this", "that", "do", "does", "did"
        }
        tokens = tokenize_and_lemmatize(query).split()
        content_words = [t for t in tokens if t not in stopwords]
        known_content = [t for t in content_words if t in self.pipeline.tokenizer.word_index]

        padded_seq = self.pipeline.transform_query(query)

        # 2. Predict intent probabilities
        raw_probs = model.predict(padded_seq, verbose=0)[0]
        latency_ms = round((time.time() - t0) * 1000, 2)

        # 3. Extract top intent and confidence
        top_idx = int(np.argmax(raw_probs))
        confidence = float(raw_probs[top_idx])

        # If all content words are completely unknown to our domain (pure Out-of-Domain query),
        # penalize confidence to trigger Fallback responsibly
        if content_words and len(known_content) == 0:
            confidence = min(confidence, 0.25)

        predicted_tag = self.pipeline.label_encoder.classes_[top_idx]

        # 4. Generate sorted probability distribution dictionary
        all_probs = {
            cls_name: round(float(raw_probs[i]), 4)
            for i, cls_name in enumerate(self.pipeline.label_encoder.classes_)
        }
        sorted_probs = dict(sorted(all_probs.items(), key=lambda item: item[1], reverse=True))

        # 5. Confidence Thresholding & Fallback Logic
        is_fallback = False
        if confidence < active_threshold:
            is_fallback = True
            bot_reply = random.choice(self.FALLBACK_RESPONSES)
        else:
            # Pick a domain response mapped to the predicted intent
            available_responses = self.pipeline.responses_map.get(predicted_tag, [])
            bot_reply = (
                random.choice(available_responses)
                if available_responses
                else "Thank you for reaching out! How else can I assist your shopping today?"
            )

        return {
            "query": query,
            "predicted_intent": predicted_tag,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "response": bot_reply,
            "is_fallback": is_fallback,
            "threshold_used": active_threshold,
            "model_used": "Bidirectional LSTM" if "bi" in chosen_model else "Standard LSTM",
            "latency_ms": latency_ms,
            "probabilities": sorted_probs,
        }


if __name__ == "__main__":
    engine = EcommerceChatbotEngine(confidence_threshold=0.50)
    print("Testing EcommerceChatbotEngine with sample queries:\n")

    test_queries = [
        "Suggest a good laptop under ₹50,000.",
        "Where is my order?",
        "I want to cancel my order.",
        "How can I return a product?",
        "My payment failed. What should I do?",
        "Who is the president of France?",  # Unsupported / Out-of-Domain query
    ]

    for q in test_queries:
        res = engine.get_response(q)
        print(f"User: {q}")
        print(f"  -> Model: {res['model_used']} | Intent: {res['predicted_intent']} | Conf: {res['confidence_percent']}% | Fallback: {res['is_fallback']}")
        print(f"  -> Bot: {res['response']}\n")
