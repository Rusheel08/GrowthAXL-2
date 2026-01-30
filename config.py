import os
import random

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHAT_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.1-8b-instant"

LANGFUSE_ENABLED = True
JUDGE_ENABLED = True

TRACE_SAMPLE_RATE = 0.2

def should_sample(rate: float = TRACE_SAMPLE_RATE) -> bool:
    return random.random() < rate