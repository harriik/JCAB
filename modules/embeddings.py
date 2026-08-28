"""
Embeddings module for JCAB.
Handles Sentence-BERT initialization and embedding generation.
"""
import logging
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME, EXPECTED_EMBEDDING_DIMENSION

logger = logging.getLogger("JCAB")

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    logger.info("Loading Sentence-BERT model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dimension_method = getattr(model, "get_embedding_dimension", None)
    if dimension_method is None:
        dimension_method = model.get_sentence_embedding_dimension
    dimension = dimension_method()
    if dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Sentence-BERT model {EMBEDDING_MODEL_NAME!r} produces {dimension} features; "
            f"the trained models require {EXPECTED_EMBEDDING_DIMENSION}."
        )
    return model

def generate_embedding(conversation_text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Convert conversation text into the same type of
    normalized 384-dimensional Sentence-BERT embedding
    used during the preliminary model training.
    """
    import sys, os
    # Preserve original stderr
    original_stderr = sys.stderr
    # Redirect stderr to null to silence tqdm flush errors on Windows Streamlit
    sys.stderr = open(os.devnull, 'w')
    try:
        embedding = model.encode(
            [conversation_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    finally:
        # Restore original stderr
        sys.stderr.close()
        sys.stderr = original_stderr

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
        
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"Expected one embedding row, received shape {embedding.shape}.")

    if embedding.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
            f"received {embedding.shape[1]}."
        )
        
    return embedding
