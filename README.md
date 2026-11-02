# 🔎 Named Entity Recognition (NER) with BERT — Gradio App

An interactive web application that detects and classifies named entities in English text using a **BERT model fine-tuned on CoNLL-2003**.

Paste any English text, click **Analyze**, and get the original text with color-coded entities, a legend, and a table of all extracted entities.

## ✨ Features

- Highlights entities directly inside the original text, each with a small type label
- Color legend and per-type entity counts
- Entity table: text, type, category, confidence, and character span
- Accurate highlighting using character offsets (`offset_mapping`), with no regex, so repeated words are handled correctly
- Proper handling of BERT subword tokens and BIO entity boundaries
- Long-text support via a sliding window (128 tokens, 32-token overlap)
- Automatic CUDA / CPU detection
- Works locally and on Kaggle

## 🎨 Entity Types

| Type | Meaning | Color |
|---|---|---|
| PER | Person names | 🔵 Blue |
| ORG | Organizations | 🟢 Green |
| LOC | Locations | 🔴 Red |
| MISC | Miscellaneous | 🟣 Purple |

## 📊 Model Comparison (CoNLL-2003)

| Model | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| LSTM + GloVe | 74.74% | 58.47% | 65.61% |
| BiLSTM + GloVe | 78.05% | 61.27% | 68.65% |
| BiLSTM + CRF | 79.76% | 62.18% | 69.88% |
| **BERT** | **89.83%** | **91.37%** | **90.59%** |

BERT performed best, so it is the only model used in this app. The app is inference-only; no training happens here.

## 📁 Project Structure

```
.
├── app.py              # Gradio application + inference logic
├── requirements.txt    # Python dependencies
├── README.md
└── bert_ner_model/     # Fine-tuned BERT model + tokenizer (not included, see below)
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Add the model

Place your fine-tuned model folder next to `app.py` and name it `bert_ner_model`. It must contain the model weights, `config.json` (with `id2label`), and the tokenizer files (a fast tokenizer is required).

> Model weights are large. Do not push them to GitHub directly; use Git LFS or host them on the Hugging Face Hub / Kaggle instead.

To use a different location, set the `MODEL_DIR` environment variable:

```bash
# Linux / macOS
export MODEL_DIR=/path/to/bert_ner_model
# Windows (PowerShell)
$env:MODEL_DIR = "C:\path\to\bert_ner_model"
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Running on Kaggle

```python
import os
os.environ["MODEL_DIR"] = "/kaggle/input/<dataset-name>/bert_ner_model"
```

Then run `app.py` in a notebook cell. Kaggle is detected automatically and a public share link is created.

## ⚙️ How It Works

1. **Tokenization:** the text is tokenized with the BERT fast tokenizer (`max_length=128`, `stride=32`, `return_overflowing_tokens=True`), so long inputs become overlapping windows.
2. **Prediction:** the model runs in `eval()` mode under `torch.no_grad()`. Predicted IDs are converted with `model.config.id2label`.
3. **Subword merging:** subwords are grouped into words with `word_ids`; each word takes the label of its first subword.
4. **Overlap resolution:** if a word appears in two windows, the prediction from the window where it is furthest from the edge is kept.
5. **Entity grouping:** consecutive `B-X` / `I-X` words are merged into one entity. The entity text is sliced from the original input using character offsets, so punctuation and spacing are preserved.
6. **Rendering:** the original text is rebuilt with HTML-escaped segments and colored entity spans, then shown with a legend and a results table.

## 🛠️ Tech Stack

Python · PyTorch · Hugging Face Transformers · Gradio · HTML/CSS

## 🔧 Configuration

Edit at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `MAX_LEN` | 128 | Max tokens per window |
| `STRIDE` | 32 | Token overlap between windows |
| `BATCH_SIZE` | 16 | Windows per forward pass |
| `MODEL_DIR` | `./bert_ner_model` | Model folder (or env var) |

## ❗ Troubleshooting

- **`FileNotFoundError: Model directory not found`**: check the folder name or set `MODEL_DIR`.
- **Labels appear as `LABEL_0`, `LABEL_1`...**: `id2label` was not saved in `config.json` during training.
- **Fast tokenizer error**: save the tokenizer with `tokenizer.save_pretrained(...)` using a fast tokenizer.

## 📄 License

Add your license here (e.g., MIT).

## 🙏 Acknowledgments

- [CoNLL-2003 shared task](https://www.clips.uantwerpen.be/conll2003/ner/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio](https://www.gradio.app/)
