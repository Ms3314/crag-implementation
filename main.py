"""Entry point for the CRAG demo."""

import os

from agent.pipeline import build_app
from engine.config import INDEX_DIR, QUESTION, REBUILD_INDEX
from engine.embeddings import get_embeddings
from engine.indexing import build_index, load_index, save_index


def main() -> None:
    embeddings = get_embeddings()

    # this means if we shud re embedd the whole thing
    if REBUILD_INDEX or not os.path.isdir(INDEX_DIR):
        store = build_index(embeddings)
        save_index(store)
    else:
        store = load_index(embeddings)

    app = build_app(store)

    print(f"\nQuestion: {QUESTION}\n")
    result = app.invoke(
        {"question": QUESTION, "docs": [], "answer": ""},
    )
    print("\n===== ANSWER =====\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
