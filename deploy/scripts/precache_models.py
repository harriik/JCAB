"""
Pre-caches HuggingFace models (Sentence-BERT and faster-whisper)
during Docker image build so containers start instantly without network downloads.
"""
import os
import sys

def precache():
    print("[PRECACHE] Downloading and caching Sentence-BERT (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        dim = model.get_sentence_embedding_dimension()
        print(f"[PRECACHE] Sentence-BERT cached successfully (dim={dim}).")
    except Exception as e:
        print(f"[PRECACHE ERROR] Failed to cache Sentence-BERT: {e}")
        sys.exit(1)

    print("[PRECACHE] Downloading and caching faster-whisper (tiny.en int8)...")
    try:
        from faster_whisper import WhisperModel
        whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        print("[PRECACHE] faster-whisper cached successfully.")
    except Exception as e:
        print(f"[PRECACHE ERROR] Failed to cache faster-whisper: {e}")
        sys.exit(1)

    print("[PRECACHE] All models pre-cached successfully.")

if __name__ == "__main__":
    precache()
