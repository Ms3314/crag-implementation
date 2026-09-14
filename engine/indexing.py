"""PDF loading, chunking and the FAISS index (build / save / load)."""

import time
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from engine.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BACKEND,
    EMBED_BATCH,
    INDEX_DIR,
    PDF_PATHS,
)


def build_chunks(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    docs = []
    for path in PDF_PATHS:
        docs.extend(PyPDFLoader(path).load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)

    # Clean surrogates from PDF extraction so downstream APIs don't choke.
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")
    return chunks


def _index_batch(batch: List[Document], embeddings, attempts: int = 5) -> FAISS:
    """Build a FAISS index for one batch, retrying transient API errors (504s, etc.)."""
    for attempt in range(attempts):
        try:
            return FAISS.from_documents(batch, embeddings)
        except Exception as exc:  # noqa: BLE001 - transient network/API failures
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            print(
                f"  batch failed ({type(exc).__name__}: {str(exc)[:90]}), "
                f"retrying in {wait}s ..."
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def build_index(
    embeddings, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> FAISS:
    chunks = build_chunks(chunk_size, chunk_overlap)

    print(
        f"Embedding with backend='{EMBED_BACKEND}', batch={EMBED_BATCH}, "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
    )
    t0 = time.time()
    store = _index_batch(chunks[:EMBED_BATCH], embeddings)
    for i in range(EMBED_BATCH, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        store.merge_from(_index_batch(batch, embeddings))
        print(f"  embedded {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)}")
    print(f"Built index with {store.index.ntotal} vectors in {round(time.time() - t0, 1)}s")
    return store


def save_index(store: FAISS) -> None:
    store.save_local(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR}/")


def load_index(embeddings) -> FAISS:
    store = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    print(f"Loaded index with {store.index.ntotal} vectors from {INDEX_DIR}/")
    return store
