"""
High-Performance & Low-Memory Ingest Script for Jain Scripture Documents.
Supports Windows, Google Colab, and Termux (Android) with Low-RAM Protections & Checkpoints.
"""

import sys
import os
import io
import json
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db_v2"
CHECKPOINT_FILE = BASE_DIR / "processed_files.json"
OCR_CACHE_DIR = BASE_DIR / "ocr_cache"

CATEGORIES = ["agams", "sutras", "commentaries", "general"]

if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\poppler-24.x\Library\bin"
else:
    POPPLER_PATH = None

# Auto-detect Termux / Android environment
IS_TERMUX = os.path.exists("/data/data/com.termux") or "android" in sys.platform.lower()

OCR_LANGS = "hin+eng"
MIN_TEXT_LEN = 25
OCR_DPI = 150 if IS_TERMUX else 200  # 150 DPI for low RAM
MAX_WORKERS = 1 if IS_TERMUX else 4  # 1 worker for Termux to prevent RAM OOM kill


def load_checkpoints() -> set:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_files", []))
        except Exception as e:
            print(f"Warning: Could not read checkpoint file: {e}", flush=True)
    return set()


def save_checkpoint(processed_set: set, new_file_key: str):
    processed_set.add(new_file_key)
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed_files": sorted(list(processed_set))}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save checkpoint for {new_file_key}: {e}", flush=True)


def process_ocr_image(args):
    img, page_num, pdf_name, category = args
    try:
        text = pytesseract.image_to_string(img, lang=OCR_LANGS).strip()
        if text:
            return Document(
                page_content=text,
                metadata={
                    "category": category,
                    "source_file": pdf_name,
                    "page": page_num,
                    "extraction_method": "ocr",
                },
            )
    except Exception as e:
        print(f"      OCR error on page {page_num} of {pdf_name}: {e}", flush=True)
    return None


def load_pdf_fast(pdf_path: Path, category: str) -> list[Document]:
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OCR_CACHE_DIR / f"{pdf_path.name}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
                print(f"      [✓ LOADED FROM DISK CACHE]: {pdf_path.name}", flush=True)
                return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]
        except Exception:
            pass

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    docs = []
    pages_needing_ocr = []

    # 1. Quick text extraction check
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        raw_text = (page.extract_text() or "").strip()
        if len(raw_text) >= MIN_TEXT_LEN:
            docs.append(
                Document(
                    page_content=raw_text,
                    metadata={
                        "category": category,
                        "source_file": pdf_path.name,
                        "page": page_num,
                        "extraction_method": "text",
                    },
                )
            )
        else:
            pages_needing_ocr.append(page_num)

    if pages_needing_ocr:
        print(f"      Running OCR on {len(pages_needing_ocr)}/{total_pages} scanned pages...", flush=True)

        kwargs = {}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH

        batch_size = 10 if IS_TERMUX else 25
        for b_start in range(0, len(pages_needing_ocr), batch_size):
            batch_page_nums = pages_needing_ocr[b_start : b_start + batch_size]
            first_p = batch_page_nums[0]
            last_p = batch_page_nums[-1]

            try:
                images = convert_from_path(
                    str(pdf_path),
                    dpi=OCR_DPI,
                    first_page=first_p,
                    last_page=last_p,
                    **kwargs,
                )

                ocr_tasks = []
                for p_num, img in zip(range(first_p, last_p + 1), images):
                    if p_num in batch_page_nums:
                        ocr_tasks.append((img, p_num, pdf_path.name, category))

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    results = list(executor.map(process_ocr_image, ocr_tasks))
                    for doc in results:
                        if doc:
                            docs.append(doc)

                del images
                gc.collect()

                print(f"      --> Processed OCR pages {first_p} to {last_p} of {pdf_path.name}", flush=True)

            except Exception as e:
                print(f"      Error rendering PDF batch {first_p}-{last_p} of {pdf_path.name}: {e}", flush=True)

    try:
        serializable_docs = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"      Cache save warning: {e}", flush=True)

    return docs


LOADER_BY_EXT = {
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def process_and_commit_file(path: Path, category: str, vectorstore: Chroma, text_splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    ext = path.suffix.lower()
    file_docs = []

    if ext == ".pdf":
        file_docs = load_pdf_fast(path, category)
    elif ext == ".csv":
        try:
            loader = CSVLoader(str(path), encoding="utf-8")
            file_docs = loader.load()
        except Exception:
            loader = CSVLoader(str(path), encoding="latin-1")
            file_docs = loader.load()
        for doc in file_docs:
            doc.metadata["category"] = category
            doc.metadata["source_file"] = path.name
            doc.metadata["extraction_method"] = "csv"
    else:
        loader_cls = LOADER_BY_EXT.get(ext)
        if loader_cls:
            file_docs = loader_cls(str(path)).load()
            for doc in file_docs:
                doc.metadata["category"] = category
                doc.metadata["source_file"] = path.name
                doc.metadata["extraction_method"] = "text"

    if not file_docs:
        return []

    chunks = text_splitter.split_documents(file_docs)
    if chunks:
        batch_size = 100 if IS_TERMUX else 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)

    del file_docs
    gc.collect()

    return chunks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Starting Fast RAG Ingestion Pipeline with Checkpoint Recovery ===", flush=True)
    if IS_TERMUX:
        print("Environment Detected: TERMUX (Android) - Low-Memory Safeguards Enabled!", flush=True)
    print(f"Hardware Acceleration: {'ENABLED (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else 'DISABLED (CPU)'}", flush=True)

    processed_set = load_checkpoints()
    print(f"Found {len(processed_set)} previously completed files in checkpoint record.", flush=True)

    print(f"Loading Multilingual HuggingFace Embeddings on {device.upper()}...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": device}
    )

    vectorstore = Chroma(
        collection_name="jain_knowledge_v1",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", "॥", ". ", " ", ""],
    )

    total_files_processed = 0

    for category in CATEGORIES:
        folder = DATA_DIR / category
        if not folder.exists():
            continue

        files = sorted([p for p in folder.rglob("*") if p.is_file()])
        print(f"\n[{category}] Found {len(files)} files.", flush=True)

        for idx, path in enumerate(files, 1):
            file_key = f"{category}/{path.name}"

            if file_key in processed_set:
                print(f"  [{idx}/{len(files)}] [✓ SKIPPING - ALREADY PROCESSED]: {path.name}", flush=True)
                continue

            print(f"  [{idx}/{len(files)}] [--> PROCESSING NEW FILE]: {path.name}", flush=True)
            try:
                chunks = process_and_commit_file(path, category, vectorstore, text_splitter)
                save_checkpoint(processed_set, file_key)
                total_files_processed += 1
                print(f"      [✓ SAVED CHECKPOINT]: {file_key} ({len(chunks)} chunks committed to database)", flush=True)
            except Exception as e:
                print(f"      Failed to process {file_key}: {e}", flush=True)

            gc.collect()

    print(f"\n🎉 Ingestion Session Complete!", flush=True)
    print(f"Newly Processed Files: {total_files_processed}")
    print(f"Total Files Recorded in Checkpoint: {len(processed_set)}")
    print(f"Vector database saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
