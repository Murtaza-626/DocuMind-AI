import _fix_win_encoding  # noqa: F401  — must be FIRST import (patches io/pathlib on Windows)

from datetime import datetime
from pathlib import Path
import gc
import sys
import traceback
import streamlit as st
import os
from config import settings
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONUTF8"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Use 1 thread each — prevents OMP/MKL from spawning competing threads that
# exhaust the limited free RAM (1-2 GB) on this machine during model loading.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def _lazy_imports():
    # Import heavy dependencies only when a feature is used.
    from data_processor import process_pdf_for_training
    from data_processor import process_pdfs_for_training
    from inference import answer_question, create_rag_engine, summarize_document

    return {
        "process_pdf_for_training": process_pdf_for_training,
        "process_pdfs_for_training": process_pdfs_for_training,
        "answer_question": answer_question,
        "create_rag_engine": create_rag_engine,
        "summarize_document": summarize_document,
    }


def _get_train_libs():
    # Import training dependencies only when training is requested.
    from train_model import train_lora_adapter

    return {
        "train_lora_adapter": train_lora_adapter,
    }


st.set_page_config(page_title="Local PDF RAG + QLoRA Trainer", layout="wide")
st.title("Local PDF RAG + LoRA Fine-Tuning")
st.caption("Phase 1: Upload and train | Phase 2: Local grounded Q&A and summarization")

if "prepared" not in st.session_state:
    st.session_state.prepared = False
if "trained" not in st.session_state:
    st.session_state.trained = False
if "latest_pdf" not in st.session_state:
    st.session_state.latest_pdf = None
if "prep_info" not in st.session_state:
    st.session_state.prep_info = {}
if "train_info" not in st.session_state:
    st.session_state.train_info = {}
if "engine" not in st.session_state:
    st.session_state.engine = None


@st.cache_resource(show_spinner=False)
def _load_engine_cached():
    libs = _lazy_imports()
    return libs["create_rag_engine"](settings)


def _save_uploaded_pdf(uploaded_file) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings.raw_pdf_dir / f"{ts}_{uploaded_file.name}"
    with out_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


tab_train, tab_infer = st.tabs(["Phase 1 - Upload and Train", "Phase 2 - Q&A and Summarize"])

with tab_train:
    st.subheader("1) Upload PDF")
    uploaded = st.file_uploader("Upload a PDF document", 
                                type=["pdf"], key="train_pdf", 
                                accept_multiple_files = True)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Prepare Data (Chunk + JSONL + Vector DB)", use_container_width=True):
            if not uploaded:
                st.warning("Please upload a PDF first.")
            else:
                with st.spinner("Processing PDF and building datasets..."):
                    try:
                        libs = _lazy_imports()
                        if isinstance(uploaded, list):
                            pdf_paths = [_save_uploaded_pdf(item) for item in uploaded]
                            prep_info = libs["process_pdfs_for_training"](
                                [str(path) for path in pdf_paths], settings
                            )
                            st.session_state.latest_pdf = [str(path) for path in pdf_paths]
                        else:
                            pdf_path = _save_uploaded_pdf(uploaded)
                            prep_info = libs["process_pdf_for_training"](str(pdf_path), settings)
                            st.session_state.latest_pdf = str(pdf_path)
                    except Exception as exc:
                        traceback.print_exc()
                        sys.stdout.flush()
                        sys.stderr.flush()
                        st.error(
                            f"**Data preparation failed.**\n\n"
                            f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                            "Check the terminal for the full traceback."
                        )
                        st.stop()

                # Free memory after heavy processing
                gc.collect()
                st.session_state.prepared = True
                st.session_state.prep_info = prep_info
                st.success("Data preparation complete.")


    with col_b:
        # Check if physical files exist on the hard drive
        dataset_exists = Path(settings.training_data_file).exists()
        vector_db_exists = settings.vector_db_dir.exists()

        if st.button("Start QLoRA Training", use_container_width=True):
            # Check physical existence instead of session_state
            if not (dataset_exists and vector_db_exists):
                st.warning("Please run data preparation first (Upload PDFs and click Prepare Data).")
            else:
                with st.spinner("Training LoRA adapter. This may take a while..."):
                    try:
                        train_libs = _get_train_libs()
                        train_info = train_libs["train_lora_adapter"](
                            train_jsonl_path=settings.training_data_file,
                            output_dir=settings.adapter_output_dir,
                            cfg=settings,
                        )
                        
                        gc.collect()
                        st.session_state.trained = True
                        st.session_state.train_info = train_info
                        _load_engine_cached.clear()
                        st.session_state.engine = None
                        st.success("Training complete. Adapter saved locally.")
                        
                    except Exception as exc:
                        traceback.print_exc()
                        sys.stdout.flush()
                        sys.stderr.flush()
                        st.error(
                            f"**Training failed.**\n\n"
                            f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                            "Check the terminal for the full traceback."
                        )
                        st.stop()

    if st.session_state.prep_info:
        st.markdown("### Preparation Summary")
        st.json(st.session_state.prep_info)

    if st.session_state.train_info:
        st.markdown("### Training Summary")
        st.json(st.session_state.train_info)

with tab_infer:
    st.subheader("2) Local Inference with Grounded RAG")

    adapter_ready = (settings.adapter_output_dir / "adapter_config.json").exists()

    if not adapter_ready:
        st.warning("No trained adapter found. Complete Phase 1 training first.")
    else:
        if st.button("Load Inference Engine", use_container_width=True):
            # Clear any stale cached engine before loading fresh.
            _load_engine_cached.clear()
            st.session_state.engine = None
            gc.collect()
            gc.collect()

            with st.spinner("Loading model and vector DB — this takes 1-3 min on CPU, please wait…"):
                try:
                    import torch
                    # Release any GPU/CPU caches before the heavy load.
                    if hasattr(torch.cuda, "empty_cache"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    gc.collect()
                    st.session_state.engine = _load_engine_cached()
                    st.session_state["engine_just_loaded"] = True
                    st.rerun()
                except MemoryError as exc:
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    _load_engine_cached.clear()
                    gc.collect()
                    st.error(
                        "**Out of memory.** Not enough RAM to load the model.\n\n"
                        "- Close your browser tabs and other applications, then try again.\n"
                        "- Or restart the app: `streamlit run app.py`"
                    )
                    st.session_state.engine = None
                except Exception as exc:
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    _load_engine_cached.clear()
                    gc.collect()
                    st.error(
                        f"**Inference startup failed.**\n\n"
                        f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                        "Check the terminal window for the full traceback."
                    )
                    st.session_state.engine = None

        engine = st.session_state.engine
        if engine is None:
            st.info("Click **Load Inference Engine** above to start Q&A and summarization.")
            st.stop()

        if st.session_state.pop("engine_just_loaded", False):
            st.success("Inference engine loaded. Ask a question or run a summary below.")

        qa_col, sum_col = st.columns(2)

        with qa_col:
            st.markdown("### Ask Questions")
            question = st.text_area(
                "Question",
                placeholder="Ask a question grounded in your uploaded PDF...",
                key="qa_question",
                height=140,
            )
            if st.button("Run Grounded Q&A", use_container_width=True):
                if not question.strip():
                    st.warning("Enter a question first.")
                else:
                    with st.spinner("Generating grounded answer..."):
                        libs = _lazy_imports()
                        answer = libs["answer_question"](engine, question.strip())
                    st.markdown("#### Answer")
                    st.write(answer)

        with sum_col:
            st.markdown("### Summarize Document")
            focus = st.text_area(
                "Optional summary focus",
                placeholder="Example: Summarize legal obligations and deadlines",
                key="sum_focus",
                height=140,
            )
            if st.button("Run Strict Summary", use_container_width=True):
                focus_text = focus.strip() if focus.strip() else "Summarize the core facts"
                with st.spinner("Generating factual bullet summary..."):
                    libs = _lazy_imports()
                    summary = libs["summarize_document"](engine, focus=focus_text)
                st.markdown("#### Summary")
                st.write(summary)

st.divider()
st.caption(
    "This app uses local embeddings + Chroma retrieval and a locally fine-tuned LoRA adapter. "
)
