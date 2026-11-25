"""
BM25 Retriever Tool for HotpotQA 2-hop Retrieval (M1 Tool).
Uses in-process rank_bm25 on pooled sample passages.
Complies with Decision D2 (c=1), Decision A-25.
"""

import time
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

@dataclass
class RetrievalResult:
    passages: List[Dict[str, str]]  # list of {"title": str, "text": str}
    context_text: str
    success: bool
    cost: float
    latency_s: float
    error_msg: Optional[str] = None

class BM25Tool:
    def __init__(self, corpus: List[Dict[str, str]], cost: float = 1.0):
        """
        corpus: list of {"title": str, "text": str}
        """
        self.corpus = corpus
        self.cost = cost
        self.tokenized_corpus = [self._tokenize(doc["title"] + " " + doc["text"]) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def execute(self, query: str, top_k: int = 3) -> RetrievalResult:
        start_time = time.perf_counter()
        try:
            tokenized_query = self._tokenize(query)
            if not tokenized_query:
                elapsed = time.perf_counter() - start_time
                return RetrievalResult(
                    passages=[],
                    context_text="",
                    success=False,
                    cost=self.cost,
                    latency_s=elapsed,
                    error_msg="Empty retrieval query"
                )

            top_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)
            context_blocks = [f"[{doc['title']}]: {doc['text']}" for doc in top_docs]
            context_str = "\n".join(context_blocks)

            elapsed = time.perf_counter() - start_time
            return RetrievalResult(
                passages=top_docs,
                context_text=context_str,
                success=True,
                cost=self.cost,
                latency_s=elapsed
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return RetrievalResult(
                passages=[],
                context_text="",
                success=False,
                cost=self.cost,
                latency_s=elapsed,
                error_msg=str(e)
            )
