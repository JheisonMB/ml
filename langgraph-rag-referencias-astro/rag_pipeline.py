import json
import os
import re
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

PDF_DIR = Path(os.environ.get("ASTRO_DENOISE_PDFS", "/home/jheisonmblivecom/Projects/UniverLab/astro-denoise/referencias/asreview/pdfs"))
INDEX_DIR = Path(__file__).parent / "index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
COLLECTION = "astro_denoise_refs_cosine"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_PAGE_CHARS = 200
BATCH_SIZE = 128

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def paper_metadata(pdf_path):
    year_match = re.match(r"(\d{4}|XXXX)_", pdf_path.name)
    title = re.sub(r"^(\d{4}|XXXX)_[A-Za-z]+_", "", pdf_path.stem).replace("_", " ")
    return {
        "source": pdf_path.name,
        "year": year_match.group(1) if year_match else "XXXX",
        "title_guess": title,
    }


def read_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        if len(text) >= MIN_PAGE_CHARS:
            pages.append(Document(page_content=text, metadata={"page": number}))
    return pages


def chunk_pdf(pdf_path):
    base = paper_metadata(pdf_path)
    chunks = []
    for page in read_pages(pdf_path):
        for piece in splitter.split_documents([page]):
            piece.metadata = {**base, **piece.metadata}
            chunks.append(piece)
    for position, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{base['source']}#{position}"
    return chunks


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"indexed": {}, "failed": {}}


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


def get_vectorstore():
    INDEX_DIR.mkdir(exist_ok=True)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=str(INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(limit=None):
    vectorstore = get_vectorstore()
    manifest = load_manifest()
    pdfs = sorted(PDF_DIR.glob("*.pdf"))[:limit]
    started = time.time()
    for number, pdf_path in enumerate(pdfs, start=1):
        if pdf_path.name in manifest["indexed"]:
            continue
        try:
            chunks = chunk_pdf(pdf_path)
            if not chunks:
                raise ValueError("sin texto extraíble")
            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start : start + BATCH_SIZE]
                vectorstore.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
            manifest["indexed"][pdf_path.name] = len(chunks)
        except Exception as error:
            manifest["failed"][pdf_path.name] = f"{type(error).__name__}: {error}"[:200]
        save_manifest(manifest)
        print(
            f"[{number}/{len(pdfs)}] {pdf_path.name[:60]} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
    return vectorstore, manifest


if __name__ == "__main__":
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _, result = build_index(limit_arg)
    print("indexados:", len(result["indexed"]), "fallidos:", len(result["failed"]))
