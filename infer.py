# infer.py - Simple CSF Inference
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ============================================================
# SETUP
# ============================================================

MODEL_DIR = "models"

tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
session = ort.InferenceSession(f"{MODEL_DIR}/model.onnx", providers=['CPUExecutionProvider'])

SLOTS = ["event", "intent", "time", "condition", "agent", "object", "location", "purpose", "modifier"]
LABELS = {
    "event": ["GO", "STAY", "BUY", "WORK", "MEET", "EAT", "LEARN"],
    "intent": ["NONE", "PLAN", "WANT", "DECIDE"],
    "time": ["NONE", "TODAY", "TOMORROW", "YESTERDAY", "NOW"],
    "condition": [
        "NONE",
        # Weather
        "IF_RAIN", "IF_SUNNY", "IF_COLD", "IF_HOT", "IF_WINDY",
        # Time
        "IF_LATE", "IF_EARLY", "IF_WEEKEND", "IF_NIGHT", "IF_MORNING",
        # Health
        "IF_SICK", "IF_TIRED", "IF_HUNGRY", "IF_THIRSTY", "IF_FULL",
        # Schedule
        "IF_BUSY", "IF_FREE", "IF_HOLIDAY", "IF_WORKING",
        # Mood
        "IF_BORED", "IF_HAPPY", "IF_SAD", "IF_STRESSED", "IF_ANGRY",
        # Social
        "IF_ALONE", "IF_WITH_FRIENDS", "IF_WITH_FAMILY",
        # Activity
        "IF_FINISH_WORK", "IF_FINISH_SCHOOL", "IF_FINISH_EATING", "IF_WATCH_MOVIE", "IF_LISTEN_MUSIC",
        # Financial
        "IF_HAVE_MONEY", "IF_NO_MONEY",
    ],
    "agent": ["ME", "YOU", "HE", "SHE", "THEY"],
    "object": ["NONE", "FOOD", "BOOK", "MEDICINE", "THING"],
    "location": ["NONE", "HOME", "SCHOOL", "HOSPITAL", "OFFICE", "STORE"],
    "purpose": ["NONE", "REST"],
    "modifier": ["NONE", "FAST", "SLOW", "ALONE"]
}
GLOSS_ORDER = ["modifier", "time", "condition", "agent", "location", "object", "event", "purpose"]

# ============================================================
# INFERENCE
# ============================================================

def predict(text):
    # Tokenize
    encoded = tokenizer.encode(text)

    # Get IDs and attention mask from tokenizer
    input_ids = encoded.ids[:64]
    attention_mask = encoded.attention_mask[:64]

    # Pad if needed
    pad_len = 64 - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    # Run model
    outputs = session.run(None, {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64)
    })

    # Parse results
    return {slot: LABELS[slot][np.argmax(outputs[i])] for i, slot in enumerate(SLOTS)}


def to_gloss(csf):
    return " ".join(
        csf[s] for s in GLOSS_ORDER
        if csf.get(s) and csf[s] != "NONE" and not (s == "agent" and csf[s] == "ME")
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Process command-line arguments
        text = " ".join(sys.argv[1:])
        result = predict(text)
        gloss = to_gloss(result)
        print(f"📝 {text}")
        print(f"🤟 {gloss}")
        print(f"\nCSF: {result}")
    else:
        # Run examples
        examples = [
            "I go to school tomorrow.",
            "明日、学校に行く。",
            "Nếu mưa thì tôi ở nhà.",
            "Je travaille à l'hôpital demain.",
            "If I'm sick, I rest at home.",
            "I am eating pizza",
        ]

        print("CSF Inference\n" + "=" * 50)
        for text in examples:
            gloss = to_gloss(predict(text))
            print(f"\n📝 {text}")
            print(f"🤟 {gloss}")
