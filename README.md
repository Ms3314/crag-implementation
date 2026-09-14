# CRAG Demo

A simple implementation of **Corrective Retrieval Augmented Generation (CRAG)**.

Inspired by the paper: *Corrective Retrieval Augmented Generation* — Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (2024) — https://arxiv.org/abs/2401.15884

This is a minimal, CLI-only version of the idea (no FastAPI, no React): it retrieves chunks from a FAISS index, uses an LLM to judge whether the retrieved context is correct, and falls back to a web search when it isn't.

<img width="7533" height="4773" alt="image" src="https://github.com/user-attachments/assets/fec33ff6-76ae-4c05-bb4b-781392c2504a" />


## LangGraph implementation 

<img width="279" height="753" alt="image" src="https://github.com/user-attachments/assets/3b0d2ab4-b7d4-4689-ac2c-8c0200617993" />


## Run

```bash
pip install -r requirements.txt
python main.py
```

Configure it by editing `engine/config.py` (question, PDFs, chunk size, thresholds, embedding backend).
