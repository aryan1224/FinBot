import os

import chromadb
from pypdf import PdfReader

from modules.chunking import chunk_text_by_headers
from modules.embedding import embed_texts

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "data", "chroma")
COLLECTION_NAME = "bank_policies"


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ingest_pdf(pdf_path: str) -> int:
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in {os.path.basename(pdf_path)}")

    chunks = chunk_text_by_headers(text)
    if not chunks:
        raise ValueError(f"Chunking produced no chunks for {os.path.basename(pdf_path)}")

    embeddings = embed_texts(chunks)

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    base_id = os.path.splitext(os.path.basename(pdf_path))[0]
    ids = [f"{base_id}-{i:04d}" for i in range(len(chunks))]
    metadatas = [
        {"source": os.path.basename(pdf_path), "chunk_index": i} for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    total = collection.count()
    print(
        f"[ingestion] wrote {len(chunks)} chunks for {os.path.basename(pdf_path)} "
        f"to {CHROMA_PATH} -- collection '{COLLECTION_NAME}' now has {total} total chunks."
    )

    return len(chunks)
