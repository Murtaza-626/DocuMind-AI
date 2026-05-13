import os
import ssl
import gc
import logging
import traceback

# Fix SSL certificate verification issues on Windows
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from langchain_core.documents import Document           # UPDATED (was langchain.schema)
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Module-level singleton so the embedding model is loaded exactly once
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def _get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance (loaded once per process)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s' (first time)...", model_name)
        _embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def load_pdf_documents(pdf_path: str) -> List[Document]:
    logger.info("Loading PDF: %s", pdf_path)
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        logger.info("Loaded %d pages from PDF.", len(docs))
        return docs
    except Exception:
        logger.error("Failed to load PDF '%s':\n%s", pdf_path, traceback.format_exc())
        raise


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)
    return text.strip()


def chunk_documents(documents: Sequence[Document], cfg: Settings = settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    cleaned_docs: List[Document] = []
    for d in documents:
        cleaned_docs.append(
            Document(
                page_content=clean_text(d.page_content),
                metadata=d.metadata or {},
            )
        )

    return splitter.split_documents(cleaned_docs)


def _extractive_bullets(text: str, max_items: int = 4) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return "- No extractable facts were found in this chunk."

    bullets = sentences[:max_items]
    return "\n".join(f"- {b}" for b in bullets)


def chunks_to_alpaca_records(chunks: Sequence[Document], cfg: Settings = settings) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    for i, chunk in enumerate(chunks[: cfg.max_chunks_for_training]):
        context = chunk.page_content
        page = chunk.metadata.get("page", "unknown")
        source = chunk.metadata.get("source", "uploaded_pdf")

        summary_instruction = (
            "Summarize the provided context as concise factual bullet points. "
            "Use only evidence from the context and do not add outside information."
        )
        summary_output = _extractive_bullets(context)
        records.append(
            {
                "instruction": summary_instruction,
                "input": context,
                "output": summary_output,
                "task": "summarization",
                "source": str(source),
                "page": str(page),
                "chunk_id": str(i),
            }
        )

        qa_instruction = (
            "Answer the question using only the provided context. "
            "If the answer is not in the context, respond exactly with: "
            "The document does not contain this information."
        )
        qa_question = "What are the key facts stated in this context?"
        qa_input = f"Context:\n{context}\n\nQuestion:\n{qa_question}"
        qa_output = summary_output
        records.append(
            {
                "instruction": qa_instruction,
                "input": qa_input,
                "output": qa_output,
                "task": "qa_grounded",
                "source": str(source),
                "page": str(page),
                "chunk_id": str(i),
            }
        )

    return records


def save_jsonl(records: Sequence[Dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


# Maximum number of documents to send to ChromaDB in one batch to avoid OOM
_CHROMA_BATCH_SIZE = 200


def build_and_persist_vector_db(chunks: Sequence[Document], cfg: Settings = settings) -> Chroma:
    """Build (or extend) a Chroma vector store, inserting in safe batches."""
    embeddings = _get_embedding_model(cfg.embedding_model_name)
    chunk_list = list(chunks)

    if not chunk_list:
        logger.warning("No chunks to insert into vector DB.")
        return Chroma(
            persist_directory=str(cfg.vector_db_dir),
            embedding_function=embeddings,
        )

    logger.info("Building vector DB with %d chunks (batch size %d)...", len(chunk_list), _CHROMA_BATCH_SIZE)

    # Insert the first batch using from_documents (creates the collection)
    first_batch = chunk_list[:_CHROMA_BATCH_SIZE]
    try:
        vector_db = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=str(cfg.vector_db_dir),
        )
    except Exception:
        logger.error("ChromaDB creation failed on first batch:\n%s", traceback.format_exc())
        raise

    # Insert remaining batches incrementally
    remaining = chunk_list[_CHROMA_BATCH_SIZE:]
    for i in range(0, len(remaining), _CHROMA_BATCH_SIZE):
        batch = remaining[i : i + _CHROMA_BATCH_SIZE]
        batch_num = (i // _CHROMA_BATCH_SIZE) + 2
        logger.info("  Inserting batch %d (%d docs)...", batch_num, len(batch))
        try:
            vector_db.add_documents(batch)
        except Exception:
            logger.error("ChromaDB batch %d insert failed:\n%s", batch_num, traceback.format_exc())
            raise
        gc.collect()

    logger.info("Vector DB built and persisted successfully.")
    # ChromaDB 0.4+ auto-persists when persist_directory is set; no manual persist needed
    return vector_db


def process_pdf_for_training(pdf_path: str, cfg: Settings = settings) -> Dict[str, str]:
    logger.info("=== process_pdf_for_training START ===")
    try:
        docs = load_pdf_documents(pdf_path)
        chunks = chunk_documents(docs, cfg=cfg)
        logger.info("Chunked into %d pieces.", len(chunks))

        records = chunks_to_alpaca_records(chunks, cfg=cfg)
        train_file = save_jsonl(records, cfg.training_data_file)
        logger.info("Saved %d training records to %s", len(records), train_file)

        build_and_persist_vector_db(chunks, cfg=cfg)
        gc.collect()

        logger.info("=== process_pdf_for_training DONE ===")
        return {
            "pdf_path": str(pdf_path),
            "num_pages": str(len(docs)),
            "num_chunks": str(len(chunks)),
            "num_training_records": str(len(records)),
            "train_jsonl": str(train_file),
            "vector_db_dir": str(cfg.vector_db_dir),
        }
    except Exception:
        logger.error("process_pdf_for_training FAILED:\n%s", traceback.format_exc())
        raise


def process_pdfs_for_training(pdf_paths: Sequence[str], cfg: Settings = settings) -> Dict[str, str]:
    logger.info("=== process_pdfs_for_training START (%d files) ===", len(list(pdf_paths)))
    try:
        paths = list(pdf_paths)
        all_chunks: List[Document] = []
        total_pages = 0

        # Process and chunk files sequentially to prevent RAM exhaustion
        for idx, path in enumerate(paths, 1):
            logger.info("Processing PDF %d/%d: %s", idx, len(paths), path)
            # Load one PDF
            docs = load_pdf_documents(path)
            total_pages += len(docs)

            # Chunk it immediately while in the loop
            chunks = chunk_documents(docs, cfg=cfg)
            all_chunks.extend(chunks)
            logger.info("  -> %d pages, %d chunks so far", len(docs), len(all_chunks))

            # Free the raw document list between PDFs
            del docs, chunks
            gc.collect()

        # Build the dataset and vector DB using the combined chunks
        records = chunks_to_alpaca_records(all_chunks, cfg=cfg)
        train_file = save_jsonl(records, cfg.training_data_file)
        logger.info("Saved %d training records to %s", len(records), train_file)

        build_and_persist_vector_db(all_chunks, cfg=cfg)
        gc.collect()

        logger.info("=== process_pdfs_for_training DONE ===")
        return {
            "pdf_paths": ", ".join(str(p) for p in paths),
            "num_pdfs": str(len(paths)),
            "num_pages": str(total_pages),
            "num_chunks": str(len(all_chunks)),
            "num_training_records": str(len(records)),
            "train_jsonl": str(train_file),
            "vector_db_dir": str(cfg.vector_db_dir),
        }
    except Exception:
        logger.error("process_pdfs_for_training FAILED:\n%s", traceback.format_exc())
        raise
