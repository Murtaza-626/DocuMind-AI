from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class Settings:
    # Paths
    data_dir: Path = ROOT_DIR / "data"
    raw_pdf_dir: Path = ROOT_DIR / "data" / "raw_pdfs"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    vector_db_dir: Path = ROOT_DIR / "data" / "vector_db"
    training_data_file: Path = ROOT_DIR / "data" / "processed" / "train_alpaca.jsonl"
    adapters_dir: Path = ROOT_DIR / "models" / "adapters"
    adapter_output_dir: Path = ROOT_DIR / "models" / "adapters" / "pdf_lora_adapter"
    # adapter_output_dir: Path = Path("/content/drive/MyDrive/DocuMind_Model")

    # Model config
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Text preprocessing / chunking
    chunk_size: int = 700
    chunk_overlap: int = 120
    max_chunks_for_training: int = 1200

    # Retrieval config
    retrieval_k: int = 4
    min_relevance_score: float = 0.15

    # QLoRA config (CPU-optimized)
    max_length: int = 2048
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = False

    # Training config (optimized for CPU / 8GB RAM)
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 1
    save_steps: int = 50
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 42

    # Generation config — keep short for CPU: each extra token is another full
    # forward pass through 30 layers; 150/200 tokens takes ~2-4 min on 8 GB RAM.
    qa_max_new_tokens: int = 150
    summary_max_new_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 0.9

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
