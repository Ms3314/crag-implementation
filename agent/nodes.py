"""Graph nodes: retrieve, evaluate, refine, web search and generate."""

import re
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent.state import State
from engine.config import LOWER_TH, RERANK_TOP_K, UPPER_TH
from engine.models import llm2, reranker


def make_retrieve(retriever):
    def retrieve(state: State) -> State:
        return {"docs": retriever.invoke(state["question"])}

    return retrieve


def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def refine(state: State) -> State:
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_results"]
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state["web_results"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    strips = decompose_to_sentences(context)
    if not strips:
        return {"strips": [], "kept_strips": [], "refined_context": ""}

    pairs = [(q, strip) for strip in strips]
    scores = reranker.score(pairs)

    ranked = sorted(zip(strips, scores), key=lambda pair: pair[1], reverse=True)
    for strip, score in ranked:
        print(f"Score: {score:.4f} | {strip[:80]}")

    # ms-marco cross-encoder scores are unbounded (often negative), so select the
    # top-K most relevant strips rather than thresholding at an absolute value.
    kept = [strip for strip, _ in ranked[:RERANK_TOP_K]]

    print(f"Refinement: {len(strips)} strips -> kept {len(kept)}")
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": "\n\n".join(kept),
    }


class DocEvalScore(BaseModel):
    score: float
    reason: str


def eval_each_doc_node(state: State) -> State:

    q = state["question"]
    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []

    for d in state["docs"]:
        strucllm = llm2.with_structured_output(DocEvalScore, method="json_schema")

        out = strucllm.invoke([
            SystemMessage(
                content=
                    "You are a strict retrieval evaluator for RAG.\n You will be given ONE retrieved chunk and a question.\n Return a relevance score in [0.0, 1.0].\n - 1.0: chunk alone is sufficient to answer fully/mostly\n - 0.0: chunk is irrelevant\n Be conservative with high scores.\n Also return a short reason.\n Output JSON only.",
            ),
            HumanMessage(
                content=f"the question is {q} and the chunk is {d}"
            )]
        )

        scores.append(out.score)
        reasons.append(out.reason)

        # 5) for CORRECT case we will refine only docs with score > LOWER_TH
        if out.score > LOWER_TH:
            good.append(d)

    # 2) CORRECT if at least one doc > UPPER_TH
    if any(s > UPPER_TH for s in scores):
        print("------------one of them is correct go ahead------------")
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    # 3) INCORRECT if all docs < LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        print("------------all are incorrect------------")
        why = "No chunk was sufficient."
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. {why}",
        }

    # 4) Anything in between => AMBIGUOUS
    why = "Mixed relevance signals."
    print("Mixieeeeeeeeeee")
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }


# this will happen just after the eval_each_doc_node
def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "web_search"


def fail(state: State):
    return {
        "answer": f"FAIL : {state["reason"]}"
    }


def ambiguous(state: State):
    return {
        "answer": f"Ambiguous : {state["reason"]}"
    }


class BetterWebQueryOutput(BaseModel):
    web_query: str


# this returns the better web query
def make_web_query_better(state: State):
    # make this qe
    print("web query is being made")
    q = state["question"]
    infrence = llm2.with_structured_output(BetterWebQueryOutput, method="json_schema")
    out = infrence.invoke([
        SystemMessage(
            content=
            """
                Rewrite the user question into a web search query composed of keywords. Rules:
                - Keep it short (6–14 words).
                - If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).
                - Do NOT answer the question.
                - Return JSON with a single key: query."
            """
        ),
        HumanMessage(
            content=f"the query is : {q}"
        )
    ])
    return {
        "web_query": out.web_query
    }


def web_search(state: State):
    print("++++++++++++++++web search is being done+++++++++++++++++++++++")
    from langchain_tavily import TavilySearch
    q = ""
    if len(state["web_query"]) > 0:
        q += state["web_query"]
    else:
        q += state["question"]
    tool = TavilySearch(
        max_results=5,
    )
    results = tool.invoke({"query": q})["results"]
    print("these are the results: ", results)
    web_results = []
    print("==========")
    for r in results:
        # print(items)
        # print("==========")
        # r = items['results']
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")

        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_results.append(Document(page_content=text, metadata={"url": url, "title": title}))
    return {
        "web_results": web_results
    }


SYSTEM_PROMPT = "Answer only from the context. If not in context, say you don't know."


def generate(state: State) -> State:
    context = state.get("refined_context") or "\n\n".join(
        d.page_content for d in state["docs"]
    )
    out = llm2.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Question: {state['question']}\n\nContext:\n{context}"
            ),
        ]
    )
    return {"answer": out.content}
