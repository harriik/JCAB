"""
Preprocessing module for JCAB.
Handles text normalization and conversation construction.
"""

def clean_text(text: str) -> str:
    """
    Basic text normalization.
    This does not alter the semantic content.
    """
    if text is None:
        return ""
    return " ".join(text.strip().split())


def build_conversation(initial_answer: str, followup_answer: str) -> str:
    """
    Construct the same conversation representation
    used by the preliminary ML training pipeline.
    """
    initial_answer = clean_text(initial_answer)
    followup_answer = clean_text(followup_answer)
    
    return initial_answer + " [SEP] " + followup_answer
