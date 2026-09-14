"""Graph state shared between nodes."""

from typing import List, TypedDict

from langchain_core.documents import Document


class State(TypedDict, total=False):
    question: str
    docs: List[Document]
    answer: str
    strips: List[str]          # smaller pieces made from retrieved chunks
    kept_strips: List[str]     # the pieces the reranker kept
    refined_context: str       # final context handed to the LLM
    good_docs: list[Document]
    reason: str
    verdict: str
    web_query: str

    # there shud be something for web result
    web_results: List[Document]
