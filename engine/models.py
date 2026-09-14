"""LangChain model clients and the cross-encoder reranker."""

import os

# Imported first so engine.config loads .env / sets HF env vars before
# transformers/huggingface_hub get imported below.
from engine.config import RERANK_MODEL

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

llm = ChatOllama(model="qwen3:8b", temperature=0.7)
llm2 = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

reranker = HuggingFaceCrossEncoder(
    model_name=RERANK_MODEL,
    model_kwargs={"token": os.getenv("HF_TOKEN")},
)
