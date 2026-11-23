"""Text preprocessing and tokenization pipeline for FAQ Intent Classification."""

import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np


def ensure_nltk_resources() -> None:
    """Ensure required NLTK corpora and tokenizers are loaded."""
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet", "corpora/omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            name = res.split("/")[-1]
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


# Initialize resources and lemmatizer
ensure_nltk_resources()
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip special symbols, and collapse whitespaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation while keeping alphanumeric and basic spacing
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Normalize, tokenize with NLTK, lemmatize each word, and rejoin.
    
    Example:
        'How do I reset my passwords?' -> 'how do i reset my password'
        'paying with cards' -> 'pay with card'
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    try:
        tokens = nltk.word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()
    
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


def load_intents_data(filepath: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Load intents.json and extract pattern texts, tags, and responses mapping.
    
    Returns:
        patterns: List of raw question variations.
        tags: Corresponding intent tag for each pattern.
        responses: Mapping of intent tag -> list of answers.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns: List[str] = []
    tags: List[str] = []
    responses: Dict[str, List[str]] = {}

    for intent in data.get("intents", []):
        tag = intent["tag"]
        responses[tag] = intent.get("responses", [])
        for pattern in intent.get("patterns", []):
            patterns.append(pattern)
            tags.append(tag)

    return patterns, tags, responses


class IntentDataPipeline:
    """End-to-end data pipeline managing text preprocessing, vocabulary indexing,
    sequence padding, and label encoding.
    """

    def __init__(self, max_sequence_length: int = 16):
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.label_encoder = None
        self.vocab_size: int = 0
        self.num_classes: int = 0
        self.responses_map: Dict[str, List[str]] = {}

    def fit_transform(
        self, patterns: List[str], tags: List[str], responses: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit tokenizer and label encoder on raw inputs and return (X, y) tensors."""
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder

        self.responses_map = responses

        # 1. Preprocess and lemmatize all patterns
        processed_patterns = [tokenize_and_lemmatize(p) for p in patterns]

        # 2. Fit Keras Tokenizer
        self.tokenizer = Tokenizer(oov_token="<OOV>", filters="")
        self.tokenizer.fit_on_texts(processed_patterns)
        self.vocab_size = len(self.tokenizer.word_index) + 1  # +1 for 0-padding

        # Convert to padded integer sequences
        sequences = self.tokenizer.texts_to_sequences(processed_patterns)
        X = pad_sequences(
            sequences, maxlen=self.max_sequence_length, padding="post", truncating="post"
        )

        # 3. Fit LabelEncoder and one-hot encode
        self.label_encoder = LabelEncoder()
        integer_labels = self.label_encoder.fit_transform(tags)
        self.num_classes = len(self.label_encoder.classes_)
        y = to_categorical(integer_labels, num_classes=self.num_classes)

        return np.array(X), np.array(y)

    def transform_query(self, query: str) -> np.ndarray:
        """Transform a single raw user query into a padded sequence tensor."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if not self.tokenizer:
            raise ValueError("Tokenizer has not been fitted or loaded.")

        processed = tokenize_and_lemmatize(query)
        sequence = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(
            sequence, maxlen=self.max_sequence_length, padding="post", truncating="post"
        )
        return padded

    def save(self, directory: str) -> None:
        """Save fitted tokenizer, label encoder, responses map, and config."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "tokenizer.pickle", "wb") as f:
            pickle.dump(self.tokenizer, f)

        with open(save_dir / "label_encoder.pickle", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(save_dir / "responses.json", "w", encoding="utf-8") as f:
            json.dump(self.responses_map, f, indent=2)

        meta = {
            "max_sequence_length": self.max_sequence_length,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "classes": self.label_encoder.classes_.tolist() if self.label_encoder else [],
        }
        with open(save_dir / "pipeline_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "IntentDataPipeline":
        """Load a saved pipeline from serialized artifacts."""
        save_dir = Path(directory)
        with open(save_dir / "pipeline_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(max_sequence_length=meta["max_sequence_length"])
        instance.vocab_size = meta["vocab_size"]
        instance.num_classes = meta["num_classes"]

        with open(save_dir / "tokenizer.pickle", "rb") as f:
            instance.tokenizer = pickle.load(f)

        with open(save_dir / "label_encoder.pickle", "rb") as f:
            instance.label_encoder = pickle.load(f)

        with open(save_dir / "responses.json", "r", encoding="utf-8") as f:
            instance.responses_map = json.load(f)

        return instance
