import _fix_win_encoding  # noqa: F401  — must be FIRST import (patches io/pathlib on Windows)

import gc
import os
import ssl
import logging

log = logging.getLogger(__name__)

# Fix SSL certificate verification issues on Windows
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

from dataclasses import dataclass
from typing import List, Tuple

import torch

# Limit PyTorch threads to avoid competing with Windows memory manager
# during CPU-only inference — keeps the process stable on low-RAM systems.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import Settings, settings


QA_SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Write a direct, continuous paragraph of 1 to 3 sentences. "
    "Do NOT use bullet points, numbered lists, headings, or extra symbols. "
    "Do NOT include introductory phrases. Start the answer immediately. "
    "If the answer is missing, say exactly: 'The document does not contain this information.'"
)

SUMMARY_SYSTEM_PROMPT = (
    "Write a concise summary of the provided context in a single, continuous paragraph. "
    "Focus entirely on the exact topic requested. "
    "Do NOT use bullet points, numbered lists, headings, or extra symbols. "
    "Do NOT include introductory phrases like 'Here is a summary'. Start the summary immediately. "
    "If the context lacks relevant information, say exactly: 'The document does not contain this information.'"
)


@dataclass
class LocalRAGEngine:
    model: object
    tokenizer: object
    vector_db: Chroma
    cfg: Settings


def load_vector_db(cfg: Settings = settings) -> Chroma:
    from data_processor import _get_embedding_model
    embeddings = _get_embedding_model(cfg.embedding_model_name)
    return Chroma(
        persist_directory=str(cfg.vector_db_dir),
        embedding_function=embeddings,
    )


def load_finetuned_model(cfg=settings):
    gc.collect()

    try:
        log.info("Loading base model: %s", cfg.base_model_name)
        # Must use float32 for CPU inference — float16 CPU kernels are incomplete in
        # PyTorch on Windows and cause a hard C-level crash during model.generate().
        # local_files_only prevents network lookups that can stall or timeout.
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,
            local_files_only=True,
        )
        gc.collect()
        log.info("Base model loaded. Attaching LoRA adapter...")

        # Wrap with adapter without merging — merge_and_unload() spikes RAM by ~2x.
        model = PeftModel.from_pretrained(
            base_model,
            str(cfg.adapter_output_dir),
            is_trainable=False,
        )
        # Drop the extra local reference — PeftModel owns the weights internally.
        del base_model
        gc.collect()

        # Put model in inference mode — no gradient tracking, no dropout.
        # Do NOT call model.float() here: the model was loaded as float32 already;
        # calling .float() on a PeftModel iterates every parameter unnecessarily and
        # can spike peak RAM on Windows when adapter weights are in a mixed state.
        model.eval()
        log.info("Adapter attached. Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.adapter_output_dir),
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("Model + tokenizer ready.")
        return model, tokenizer

    except Exception as e:
        gc.collect()
        raise e

def create_rag_engine(cfg: Settings = settings) -> LocalRAGEngine:
    model, tokenizer = load_finetuned_model(cfg)
    vector_db = load_vector_db(cfg)
    return LocalRAGEngine(model=model, tokenizer=tokenizer, vector_db=vector_db, cfg=cfg)


def _retrieve_context_with_scores(
    vector_db: Chroma,
    query: str,
    k: int,
) -> List[Tuple[Document, float]]:
    return vector_db.similarity_search_with_relevance_scores(query, k=k)


# def _generate(
#     model,
#     tokenizer,
#     system_prompt: str,
#     user_prompt: str,
#     max_new_tokens: int,
#     temperature: float,
#     top_p: float,
# ) -> str:
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
#     # apply_chat_template uses the model's native format (ChatML for SmolLM2)
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         return_tensors="pt",
#         add_generation_prompt=True,
#     ).to(model.device)

#     input_len = input_ids.shape[1]

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             do_sample=temperature > 0,
#             temperature=temperature,
#             top_p=top_p,
#             use_cache=True,
#         )

#     # Decode only the newly generated tokens — no manual string splitting needed
#     new_tokens = outputs[0][input_len:]
#     return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def _generate(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    # Match the ALPACA_TEMPLATE used in train_model.py
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{system_prompt}\n\n"
        f"### Input:\n{user_prompt}\n\n"
        "### Response:\n"
    )

    # Truncate prompt if it would exceed the model's practical context window.
    # Over-long inputs cause quadratic attention cost and can crash on CPU.
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    input_ids = encoding.input_ids.to("cpu")
    input_len = input_ids.shape[1]

    do_sample = temperature > 0
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=1.15,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_question(engine: LocalRAGEngine, question: str) -> str:
    results = _retrieve_context_with_scores(
        engine.vector_db,
        query=question,
        k=engine.cfg.retrieval_k,
    )

    if not results:
        return "The document does not contain this information."

    best_score = max(score for _, score in results)
    if best_score < engine.cfg.min_relevance_score:
        return "The document does not contain this information."

    context = "\n\n".join(doc.page_content for doc, _ in results)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = _generate(
        engine.model,
        engine.tokenizer,
        QA_SYSTEM_PROMPT,
        user_prompt,
        max_new_tokens=engine.cfg.qa_max_new_tokens,
        temperature=engine.cfg.temperature,
        top_p=engine.cfg.top_p,
    )

    return response or "The document does not contain this information."


def summarize_document(engine: LocalRAGEngine, focus: str = "Summarize the core facts") -> str:
    results = _retrieve_context_with_scores(
        engine.vector_db,
        query=focus,
        k=max(engine.cfg.retrieval_k, 6),
    )

    if not results:
        return "The document does not contain this information."

    context = "\n\n".join(doc.page_content for doc, _ in results)
    user_prompt = f"Context:\n{context}\n\nFocus: {focus}"

    response = _generate(
        engine.model,
        engine.tokenizer,
        SUMMARY_SYSTEM_PROMPT,
        user_prompt,
        max_new_tokens=engine.cfg.summary_max_new_tokens,
        temperature=engine.cfg.temperature,
        top_p=engine.cfg.top_p,
    )

    return response or "The document does not contain this information."

