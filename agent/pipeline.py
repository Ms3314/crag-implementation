"""LangGraph wiring for the CRAG pipeline."""

from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    eval_each_doc_node,
    fail,
    generate,
    make_retrieve,
    make_web_query_better,
    refine,
    route_after_eval,
    web_search,
)
from agent.state import State


def build_app(store: FAISS):
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    g = StateGraph(State)
    g.add_node("retrieve", make_retrieve(retriever))
    g.add_node("refine", refine)
    g.add_node("generate", generate)
    g.add_node("eval_each_doc_node", eval_each_doc_node)
    g.add_node("fail", fail)
    g.add_node("web_search", web_search)
    g.add_node("make_web_query_better", make_web_query_better)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc_node")
    g.add_conditional_edges("eval_each_doc_node", route_after_eval, {"refine": "refine", "web_search": "make_web_query_better"})

    g.add_edge("make_web_query_better", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)
    # g.add_edge("ambiguous", END)
    g.add_edge("fail", END)
    return g.compile()
