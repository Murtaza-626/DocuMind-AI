# Local PDF RAG + QLoRA Fine-Tuning

This project refactors a single-file Streamlit PDF assistant into a modular architecture that supports:

- Local PDF ingestion and chunking
- JSONL generation for supervised fine-tuning (instruction/input/output)
- Local QLoRA fine-tuning with Unsloth
- Local inference with strict grounded prompts
- RAG using SentenceTransformers embeddings + Chroma vector DB

## Project Structure

- `config.py`: Centralized settings for paths, chunking, retrieval, training, and generation.
- `data_processor.py`: PDF loading, text cleaning, chunking, JSONL formatting, Chroma build/persist.
- `train_model.py`: Unsloth QLoRA training script for LoRA adapter creation.
- `inference.py`: Local adapter/model loading, vector DB retrieval, strict Q&A + summarization generation.
- `app.py`: Streamlit UI with two phases:
  - Phase 1: Upload and train
  - Phase 2: Grounded Q&A and strict summarization

## Environment Setup

1. Create and activate a Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install --upgrade pip
pip install streamlit langchain langchain-community langchain-text-splitters pypdf chromadb sentence-transformers datasets trl transformers peft accelerate bitsandbytes unsloth
```

Notes:
- Unsloth + 4-bit QLoRA is intended for CUDA GPUs.
- On Windows, WSL2 with CUDA passthrough is often the most reliable setup for training.

## How Training Works

1. Upload PDF in the app.
2. Click **Prepare Data**:
   - Extract pages
   - Clean and chunk text
   - Build and persist Chroma DB
   - Build Alpaca-style JSONL records for SFT
3. Click **Start QLoRA Training**:
   - Load base model `HuggingFaceTB/SmolLM2-135M-Instruct`
   - Train LoRA adapter in 4-bit mode
   - Save adapter to `models/adapters/pdf_lora_adapter`

## Run the App

```powershell
streamlit run app.py
```

## Optional: Run Training from CLI

After preparing data, you can launch training directly:

```powershell
python train_model.py
```

## Key Behavior Guarantees

- Q&A prompt strictly enforces grounded answers.
- If retrieval confidence is low or answer is missing in context, the model returns:
  - `The document does not contain this information.`
- Summarization prompt enforces concise bullet-point output with no conversational filler.

## Where Artifacts Are Saved

- Uploaded PDFs: `data/raw_pdfs/`
- Training JSONL: `data/processed/train_alpaca.jsonl`
- Chroma DB: `data/vector_db/`
- LoRA adapter: `models/adapters/pdf_lora_adapter/`

## Suggested Model Upgrade

For stronger quality (if GPU VRAM allows), switch in `config.py`:

- `base_model_name = "unsloth/Llama-3.2-3B-Instruct"`

Keep 4-bit QLoRA enabled to reduce memory usage.
