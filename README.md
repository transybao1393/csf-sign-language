# CSF: Contrastive Semantic Features for Direct Multilingual Sign Language Generation

[![arXiv](https://img.shields.io/badge/arXiv-2601.01964-b31b1b.svg)](https://arxiv.org/abs/2601.01964)
![Model Size](https://img.shields.io/badge/Model-0.74MB-blue)
![CPU Latency](https://img.shields.io/badge/CPU-3.02ms-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.03%25-green)
![Languages](https://img.shields.io/badge/Languages-4-purple)

**99.03% accuracy • 0.74MB model • 3.02ms CPU inference • Direct multilingual → ASL GLOSS**

> Bypass English entirely: Vietnamese/Japanese/French/English → Semantic Slots → Sign Language GLOSS

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Clone & install
git clone https://github.com/transybao1393/csf-sign-language.git
cd csf-sign-language
pip install -r requirements.txt

# 2. Vietnamese → ASL GLOSS
python infer.py "Nếu mưa thì tôi ở nhà"
# Output: IF_RAIN HOME STAY

# 3. Japanese → ASL GLOSS  
python infer.py "明日、学校に行きます"
# Output: TOMORROW SCHOOL GO

# 4. Conditional expressions
python infer.py "If I'm tired, I rest at home"
# Output: IF_TIRED HOME STAY REST
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Average Accuracy** | 99.03% |
| **Model Size** | 0.74 MB |
| **CPU Latency** | 3.02 ms |
| **Throughput** | 331 inferences/sec |
| **Languages** | 4 (EN, VI, JA, FR) |
| **Condition Types** | 35 |

### Per-Slot Accuracy

| Slot | Classes | Accuracy |
|------|---------|----------|
| event | 7 | 97.8% |
| intent | 4 | 99.2% |
| time | 5 | 99.6% |
| **condition** | **35** | **99.4%** |
| agent | 5 | 99.0% |
| object | 5 | 99.2% |
| location | 6 | 97.9% |
| purpose | 2 | 99.7% |
| modifier | 4 | 99.5% |

---

## 🧠 CSF Architecture

```
Source Text (Vietnamese/Japanese/French/English)
    ↓
Custom BPE Tokenizer (8,000 tokens)
    ↓
Transformer Encoder (4 layers × 256d × 4 heads)
    ↓
9 Classification Heads
    ↓
CSF Slots → GLOSS Sequence
```

### Model Specifications

| Component | Value |
|-----------|-------|
| Hidden Size | 256 |
| Attention Heads | 4 |
| Layers | 4 |
| FFN Size | 1,024 |
| Vocabulary | 8,000 (custom BPE) |
| Parameters | ~1.5M |
| ONNX Size | 433.7 KB |

---

## 🎯 35 Condition Types

A key contribution is our comprehensive condition taxonomy:

| Category | Conditions |
|----------|------------|
| **Weather** (5) | IF_RAIN, IF_SUNNY, IF_COLD, IF_HOT, IF_WINDY |
| **Time** (5) | IF_LATE, IF_EARLY, IF_WEEKEND, IF_NIGHT, IF_MORNING |
| **Health** (5) | IF_SICK, IF_TIRED, IF_HUNGRY, IF_THIRSTY, IF_FULL |
| **Schedule** (4) | IF_BUSY, IF_FREE, IF_HOLIDAY, IF_WORKING |
| **Mood** (5) | IF_BORED, IF_HAPPY, IF_SAD, IF_STRESSED, IF_ANGRY |
| **Social** (3) | IF_ALONE, IF_WITH_FRIENDS, IF_WITH_FAMILY |
| **Activity** (5) | IF_FINISH_WORK, IF_FINISH_SCHOOL, IF_FINISH_EATING, IF_WATCH_MOVIE, IF_LISTEN_MUSIC |
| **Financial** (2) | IF_HAVE_MONEY, IF_NO_MONEY |

---

## 📂 Repository Contents

```
csf-sign-language/
├── models/
│   ├── model.onnx              # 434 KB - ONNX model
│   ├── tokenizer.json          # 321 KB - Custom BPE tokenizer
│   ├── config.json             # Model configuration
│   └── labels.json             # Slot label mappings
├── training/
│   └── CSF_Browser_Training.ipynb  # Training notebook
├── infer.py                    # Inference script
├── demo_browser.html           # Browser demo (ONNX.js)
├── requirements.txt            # Dependencies
├── data/
│   └── csf_train_19k.jsonl     # Training dataset (18,885 samples)
└── README.md
```

---

## 🎯 Usage Examples

### Python Inference

```python
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Load model and tokenizer
tokenizer = Tokenizer.from_file("models/tokenizer.json")
session = ort.InferenceSession("models/model.onnx")

SLOTS = ["event", "intent", "time", "condition", "agent", "object", "location", "purpose", "modifier"]
LABELS = {
    "event": ["GO", "STAY", "BUY", "WORK", "MEET", "EAT", "LEARN"],
    "condition": ["NONE", "IF_RAIN", "IF_SUNNY", "IF_COLD", "IF_HOT", "IF_WINDY",
                  "IF_LATE", "IF_EARLY", "IF_WEEKEND", "IF_NIGHT", "IF_MORNING",
                  "IF_SICK", "IF_TIRED", "IF_HUNGRY", "IF_THIRSTY", "IF_FULL",
                  "IF_BUSY", "IF_FREE", "IF_HOLIDAY", "IF_WORKING",
                  "IF_BORED", "IF_HAPPY", "IF_SAD", "IF_STRESSED", "IF_ANGRY",
                  "IF_ALONE", "IF_WITH_FRIENDS", "IF_WITH_FAMILY",
                  "IF_FINISH_WORK", "IF_FINISH_SCHOOL", "IF_FINISH_EATING", 
                  "IF_WATCH_MOVIE", "IF_LISTEN_MUSIC", "IF_HAVE_MONEY", "IF_NO_MONEY"],
    # ... other slots
}

def predict(text):
    enc = tokenizer.encode(text)
    outputs = session.run(None, {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64)
    })
    return {slot: LABELS[slot][np.argmax(outputs[i])] for i, slot in enumerate(SLOTS)}

# Examples
print(predict("If I'm hungry, I eat food"))
# {'event': 'EAT', 'condition': 'IF_HUNGRY', ...}

print(predict("Nếu mưa thì tôi ở nhà"))
# {'event': 'STAY', 'condition': 'IF_RAIN', 'location': 'HOME', ...}
```

### Multilingual Examples

| Input | GLOSS Output |
|-------|--------------|
| I go to school tomorrow. | TOMORROW SCHOOL GO |
| If it rains, I stay home. | IF_RAIN HOME STAY |
| If I'm bored, I watch Netflix. | IF_BORED HOME STAY |
| After work, I go home. | IF_FINISH_WORK HOME GO |
| If I have money, I go shopping. | IF_HAVE_MONEY STORE BUY |
| Nếu mưa thì tôi ở nhà. (VI) | IF_RAIN HOME STAY |
| Nếu đói thì tôi ăn. (VI) | IF_HUNGRY EAT |
| Si je suis fatigué, je me repose. (FR) | IF_TIRED HOME STAY REST |

### Browser Demo

Run the interactive demo in your browser (no Python required for inference):

```bash
# Start a local server (required for CORS)
python -m http.server 8000

# Open in browser
# http://localhost:8000/demo_browser.html
```

Or use any local server:

```bash
# Node.js
npx serve .

# PHP
php -S localhost:8000
```

The browser demo uses [ONNX.js](https://onnxruntime.ai/docs/tutorials/web/) and [Transformers.js](https://huggingface.co/docs/transformers.js) to run inference entirely client-side.

---

## 🛠️ Installation

```bash
# CPU only (recommended for inference)
pip install torch transformers tokenizers onnxruntime numpy

# GPU (for training)
pip install torch transformers tokenizers onnxruntime-gpu

# Run inference
python infer.py "Your text here"
```

---

## 🔬 Training

### Dataset

- **Total samples**: 18,885
- **Languages**: English, Vietnamese, Japanese, French
- **Condition types**: 35 (across 8 categories)
- **Train/Val split**: 90/10

### Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 2e-4 |
| Epochs | 15 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Hardware | NVIDIA A100 |
| Training Time | ~15 minutes |

### Reproduce Training

```bash
# 1. Open training notebook in Google Colab
# 2. Select GPU runtime (A100 recommended)
# 3. Run all cells
# 4. Model saved to Google Drive
```

---

## 📈 Citation

```bibtex
@article{tran2025csf,
  title={CSF: Contrastive Semantic Features for Direct Multilingual Sign Language Generation},
  author={Tran, Sy Bao},
  journal={arXiv preprint arXiv:2601.01964},
  year={2025}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers/), [ONNX Runtime](https://onnxruntime.ai/)
- Trained on Google Colab with NVIDIA A100
- Inspired by accessibility needs of deaf communities worldwide

---

## 📧 Contact

**Tran Sy Bao**  
Independent Researcher, Ho Chi Minh City, Vietnam  
Email: transybao93@gmail.com

---

⭐ **Star this repo if it helps your research!**  
🐛 **Found a bug? [Open an issue](https://github.com/transybao1393/csf-sign-language/issues)!**

---

*Code corresponding to [arXiv:2601.01964](https://arxiv.org/abs/2601.01964)*  
*Released December 2025*