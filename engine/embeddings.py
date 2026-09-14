"""Embedding backend factory."""

from engine.config import (
    EMBED_BACKEND,
    HF_EMBED_MODEL,
    OLLAMA_EMBED_MODEL,
    OPENAI_EMBED_MODEL,
)


def get_embeddings():
    if EMBED_BACKEND == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    if EMBED_BACKEND == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    # default: huggingface serverless
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(model=HF_EMBED_MODEL)
