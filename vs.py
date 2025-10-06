#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

--------------
# 1) Run all three systems on 50 validation questions with concurrency:
python3 vs.py \
  --split validation --nqa-count 50 \
  --systems none llama htnr \
  --gen-model gpt-5-nano --eval-model gpt-5-mini \
  --k 8 --embedder-model BAAI/bge-small-en-v1.5 \
  --chunk-tokens 2048 \
  --concurrency 30 --per-system-concurrency 10

# 2) HTNR only, larger k and random seed for sampling (single-threaded)
python3 vs.py --systems htnr --k 10 --seed 1337

# 3) Longest 30 validation stories with persistent memories
python3 vs.py --split validation --nqa-count 30 --select-longest --retain-context

Environment
-----------
Set your API key(s):
  export OPENAI_API_KEY=sk-...

Notes
-----
• 이 버전은 "컨텍스트 준비(리트리벌)"는 순차로 수행, "생성(챗 컴플리션)"을 비동기로 최대 동시성만큼 병렬 실행합니다.
• LlamaIndex path는 manual retrieval (retriever.retrieve) + 공통 generator로 공정 비교 유지.
• RAGAS 평가는 생성이 끝난 뒤 시스템별로 순차 수행(안정성/비용 위해). 필요시 별도 병렬화 가능.
• NarrativeQA에서 긴 본문/정답 없는 row는 skip.
• Ground truth: 질문별 가장 긴 reference answer 선택.
• 이 스크립트는 htnr_memory_v57_ann_plus.py(모듈: v4)를 PYTHONPATH/cwd에서 import합니다.
• --retain-context를 끄면(기본) 질문마다 리트리벌 메모리를 초기화합니다.
• --select-longest를 쓰면 전체 split 중 길이 기준 상위 N(--nqa-count) 샘플만 사용합니다.
• 인게스트/생성 이벤트는 logs 디렉터리에 JSONL로 즉시 기록되어 중단 후에도 분석 가능합니다.
• 각 청크는 지정한 생성 모델에 'NO CONTEXT' 응답을 요청해 확인한 후에만 메모리에 순차 적재됩니다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

llm_factory = None  # populated dynamically if available

# Optional tokenizer for precise chunking
try:  # pragma: no cover - optional dependency
    import tiktoken
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback
    _TOKEN_ENCODER = None

# -----------------------------
# OpenAI (generator)
# -----------------------------
try:
    from openai import OpenAI as _OpenAI
    from openai import AsyncOpenAI as _AsyncOpenAI
    # Exceptions (openai>=1.40)
    from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("pip install openai>=1.40.0 필요") from exc

# -----------------------------
# Ragas (evaluator)
# -----------------------------
try:
    from ragas import EvaluationDataset, evaluate
    try:
        from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall
        try:
            from ragas.metrics import AnswerRelevancy as _AnswerRel
        except Exception:  # pragma: no cover - version guard
            from ragas.metrics import ResponseRelevancy as _AnswerRel  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("ragas 설치/버전 확인 필요") from exc
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "ragas 평가용 LLM/임베딩을 위해 'langchain-openai' 설치 필요: pip install langchain-openai"
        ) from exc
    try:  # pragma: no cover - optional new API
        from ragas.llms.base import llm_factory
    except Exception:
        llm_factory = None  # type: ignore[assignment]
except Exception as exc:
    raise

# -----------------------------
# HuggingFace datasets (NarrativeQA)
# -----------------------------
try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("pip install datasets 필요") from exc

# -----------------------------
# LlamaIndex (retriever)
# -----------------------------
try:
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI as LIOpenAI
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "pip install llama-index-core llama-index-embeddings-huggingface llama-index-llms-openai 필요"
    ) from exc

# -----------------------------
# HTNRMemory V57-ANN++ (user module)
# -----------------------------
try:
    from v4 import HTNRMemory, Hyper, Numerics
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("현재 디렉토리에 htnr_memory_v57_ann_plus.py 가 있어야 합니다.") from exc


# ---------------------------------------------------------------------------
# Data structures / helpers
# ---------------------------------------------------------------------------
@dataclass
class QAItem:
    qid: str
    doc_id: str
    question: str
    ground_truth: str
    title: str
    chunks: List[str]


@dataclass
class GenJob:
    """A single generation job (system x question)."""
    system: str               # "none" | "llama" | "htnr"
    qid: str
    question: str
    contexts: List[str]
    ground_truth: str


def _count_tokens_approx(text: str) -> int:
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        try:
            return len(_TOKEN_ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split()) if text.strip() else 0


class BenchmarkLogger:
    """Append-only logger writing JSONL records for ingestion and generation events."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.ingest_dir = os.path.join(base_dir, "ingest")
        self.generation_dir = os.path.join(base_dir, "generation")
        self.chunk_ack_dir = os.path.join(base_dir, "chunk_ack")
        os.makedirs(self.ingest_dir, exist_ok=True)
        os.makedirs(self.generation_dir, exist_ok=True)
        os.makedirs(self.chunk_ack_dir, exist_ok=True)

    def _write_jsonl(self, directory: str, filename: str, payload: Dict[str, Any]) -> None:
        path = os.path.join(directory, filename)
        payload = dict(payload)
        payload.setdefault("timestamp", time.time())
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_ingest_chunk(
        self,
        *,
        system: str,
        qid: str,
        doc_id: str,
        chunk_index: int,
        total_chunks: int,
        text: str,
    ) -> None:
        record = {
            "event": "ingest_chunk",
            "system": system,
            "qid": qid,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "token_count": _count_tokens_approx(text),
            "char_count": len(text),
            "preview": text[:200].replace("\n", " ") if text else "",
        }
        self._write_jsonl(self.ingest_dir, f"{system}.jsonl", record)

    def log_ingest_complete(
        self,
        *,
        system: str,
        qid: str,
        doc_id: str,
        chunks_ingested: int,
    ) -> None:
        record = {
            "event": "ingest_complete",
            "system": system,
            "qid": qid,
            "doc_id": doc_id,
            "chunks_ingested": chunks_ingested,
        }
        self._write_jsonl(self.ingest_dir, f"{system}.jsonl", record)

    def log_generation(self, system: str, row: Dict[str, Any], *, user_prompt: str) -> None:
        payload = dict(row)
        payload["system"] = system
        payload.setdefault("timestamp", time.time())
        payload["user_prompt"] = user_prompt
        payload["prompt_token_estimate"] = _count_tokens_approx(user_prompt)
        self._write_jsonl(self.generation_dir, f"{system}.jsonl", payload)

    def log_chunk_ack(
        self,
        *,
        system: str,
        qid: str,
        doc_id: str,
        chunk_index: int,
        total_chunks: int,
        response: str,
        success: bool,
    ) -> None:
        record = {
            "event": "chunk_ack",
            "system": system,
            "qid": qid,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "response": response,
            "success": success,
        }
        self._write_jsonl(self.chunk_ack_dir, f"{system}.jsonl", record)


PROMPT_SYSTEM_TEXT = (
    "You are a careful research assistant. Answer ONLY using the CONTEXTS. "
    "If the contexts are insufficient or unrelated, say 'I don't have enough information.'"
)


def _build_user_prompt(question: str, contexts: Sequence[str]) -> str:
    if contexts:
        context_prompt = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
        return f"Question: {question}\n\nCONTEXTS:\n{context_prompt}"
    return f"Question: {question}\n\nCONTEXTS:\n(no context)"


def _build_prompt_messages(question: str, contexts: Sequence[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT_SYSTEM_TEXT},
        {"role": "user", "content": _build_user_prompt(question, contexts)},
    ]


def _stream_chunk_ack(
    client: _OpenAI,
    *,
    chunk_text: str,
    doc_id: str,
    chunk_index: int,
    total_chunks: int,
    seed: Optional[int],
    logger: Optional[BenchmarkLogger],
    system: str,
    qid: str,
    ack_model: str,
) -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "You will receive NarrativeQA context chunks. For each chunk, respond EXACTLY with 'NO CONTEXT'. "
                "Do not summarize, translate, or add any text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context chunk {chunk_index + 1}/{total_chunks} for doc {doc_id}.\n"
                "Acknowledge receipt by replying with 'NO CONTEXT'.\n\n"
                f"{chunk_text}"
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=ack_model,
            max_completion_tokens=4,
            messages=messages,
            seed=seed,
        )
        reply = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Chunk ack 실패(doc=%s chunk=%s): %s", doc_id, chunk_index, exc)
        reply = "(error)"

    success = reply.strip().upper() == "NO CONTEXT"
    if logger:
        logger.log_chunk_ack(
            system=system,
            qid=qid,
            doc_id=doc_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            response=reply,
            success=success,
        )

    if not success:
        logging.debug(
            "Chunk ack unexpected response (doc=%s chunk=%s reply=%r)",
            doc_id,
            chunk_index,
            reply,
        )


class HTNRGemmaEmbedder:
    """Adapter for SentenceTransformer models that exposes HTNRMemory interface."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m", *, batch_size: int = 32,
                 device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("pip install sentence-transformers 필요") from exc

        self.model = SentenceTransformer(model_name, device=device)
        self._dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = int(max(1, batch_size))

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts: Any) -> np.ndarray:
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)

        outputs: List[np.ndarray] = []
        for start in range(0, len(batch), self.batch_size):
            chunk = batch[start:start + self.batch_size]
            embeddings = []
            for entry in chunk:
                text = entry if isinstance(entry, str) else str(entry)
                is_query = _looks_like_question(text)
                try:
                    if is_query and hasattr(self.model, "encode_query"):
                        vec = self.model.encode_query(text)
                    elif hasattr(self.model, "encode_document"):
                        vec = self.model.encode_document(text)
                    else:
                        vec = self.model.encode([text])[0]
                except Exception:
                    vec = self.model.encode([text])[0]
                embeddings.append(np.asarray(vec, dtype=np.float32))
            outputs.extend(embeddings)
        if not outputs:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack(outputs, axis=0).astype(np.float32)


def _looks_like_question(text: str) -> bool:
    if not text:
        return False
    if "?" in text:
        return True
    return bool(re.search(r"^(why|how|what|when|who|where|which|왜|어떻게|무엇|언제|누가|어디|어느)\b", text.strip(), re.I))


def _extract_text_blob(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "story", "content"):
            if key in value:
                inner = _extract_text_blob(value[key])
                if inner:
                    return inner
        tokens = value.get("tokens") if hasattr(value, "get") else None
        if isinstance(tokens, Sequence):
            parts = [_extract_text_blob(token) for token in tokens]
            joined = " ".join(part for part in parts if part)
            if joined:
                return joined
        return ""
    if isinstance(value, Sequence):
        parts = [_extract_text_blob(v) for v in value]
        return " ".join(part for part in parts if part)
    return ""


def _extract_document_text(document: Dict[str, Any], *, allow_summary: bool) -> str:
    if not document:
        return ""
    text = _extract_text_blob(document.get("text")) if hasattr(document, "get") else ""
    if text:
        return text
    if allow_summary and hasattr(document, "get"):
        return _extract_text_blob(document.get("summary"))
    return ""


def _document_length_score(text: str) -> int:
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        try:
            return len(_TOKEN_ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split()) if text.strip() else 0


def chunk_document(text: str, *, max_context_chars: int) -> List[str]:
    if not text:
        return []
    if max_context_chars <= 0:
        return [text.strip()]
    if _TOKEN_ENCODER is not None:
        try:
            token_ids = _TOKEN_ENCODER.encode(text)
            if token_ids:
                chunks = []
                for start in range(0, len(token_ids), max_context_chars):
                    sub_ids = token_ids[start:start + max_context_chars]
                    chunk_text = _TOKEN_ENCODER.decode(sub_ids).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                if chunks:
                    return chunks
        except Exception:
            pass
    words = text.split()
    if not words:
        return [text.strip()]
    chunks: List[str] = []
    for start in range(0, len(words), max_context_chars):
        piece = " ".join(words[start:start + max_context_chars]).strip()
        if piece:
            chunks.append(piece)
    return chunks if chunks else [text.strip()]


# ---------------------------------------------------------------------------
# NarrativeQA loader
# ---------------------------------------------------------------------------

def load_narrativeqa(
    split: str,
    *,
    take: int,
    seed: int,
    chunk_tokens: int,
    allow_summary_fallback: bool,
    select_longest: bool,
) -> List[QAItem]:
    dataset = load_dataset("deepmind/narrativeqa", split=split)

    doc_chunks_cache: Dict[str, List[str]] = {}
    doc_length_cache: Dict[str, int] = {}
    candidates: List[Tuple[int, QAItem]] = []

    for idx in range(len(dataset)):
        row = dataset[idx]
        question_block = row.get("question") if isinstance(row, dict) else None
        question_text = (question_block or {}).get("text") if isinstance(question_block, dict) else None
        document = row.get("document") if isinstance(row, dict) else None
        title = (document.get("summary") or {}).get("title") if isinstance(document, dict) else None
        answers = row.get("answers") if isinstance(row, dict) else None

        normalized_answers = []
        if isinstance(answers, Sequence):
            for ans in answers:
                if isinstance(ans, dict):
                    text = ans.get("text")
                    if isinstance(text, str) and text.strip():
                        normalized_answers.append(text.strip())

        if not question_text or not normalized_answers or not isinstance(document, dict):
            continue

        ground_truth = max(normalized_answers, key=len)
        doc_id = str(document.get("id") or (question_block or {}).get("document_id") or f"doc-{idx}")
        if doc_id in doc_chunks_cache:
            chunk_list = doc_chunks_cache[doc_id]
            length_score = doc_length_cache.get(doc_id)
            if length_score is None:
                length_score = _document_length_score(" \n".join(chunk_list))
                doc_length_cache[doc_id] = length_score
        else:
            long_text = _extract_document_text(document, allow_summary=allow_summary_fallback)
            if not long_text:
                continue
            chunk_list = chunk_document(long_text, max_context_chars=chunk_tokens)
            if not chunk_list:
                continue
            doc_chunks_cache[doc_id] = chunk_list
            length_score = _document_length_score(long_text)
            doc_length_cache[doc_id] = length_score

        question_id = (
            (question_block or {}).get("question_id")
            or (question_block or {}).get("qid")
            or (question_block or {}).get("id")
            or (question_block or {}).get("number")
            or idx
        )
        qid = f"{doc_id}::{question_id}"
        qa_item = QAItem(
            qid=str(qid),
            doc_id=str(doc_id),
            question=question_text.strip(),
            ground_truth=ground_truth.strip(),
            title=(title or "").strip() or str(doc_id),
            chunks=chunk_list,
        )
        candidates.append((length_score, qa_item))

    if not candidates:
        raise RuntimeError(
            "NarrativeQA에서 유효한 샘플을 찾지 못했습니다. split과 인터넷 연결을 확인하세요."
        )

    if select_longest:
        candidates.sort(key=lambda pair: pair[0], reverse=True)
    else:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    selected_items = [item for _, item in candidates[:take]]
    if not selected_items:
        raise RuntimeError("요청한 수 만큼 샘플을 확보하지 못했습니다. --nqa-count 값을 줄여보세요.")
    return selected_items


# ---------------------------------------------------------------------------
# LLM answer generation (sync + async)
# ---------------------------------------------------------------------------

def build_answer(model: str, client: _OpenAI, question: str, contexts: List[str], *,
                 max_completion_tokens: int = 512, seed: Optional[int] = None) -> str:
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=int(max_completion_tokens),
        messages=_build_prompt_messages(question, contexts),
        seed=seed,
    )
    return (response.choices[0].message.content or "").strip()


async def build_answer_async(
    model: str,
    aclient: _AsyncOpenAI,
    question: str,
    contexts: List[str],
    *,
    max_completion_tokens: int = 2048,
    seed: Optional[int] = None,
    retries: int = 5,
    base_backoff: float = 1.5,
) -> str:
    """Async version with simple exponential backoff for rate limits / transient errors."""
    messages = _build_prompt_messages(question, contexts)

    for attempt in range(retries + 1):
        try:
            resp = await aclient.chat.completions.create(
                model=model,
                max_completion_tokens=int(max_completion_tokens),
                messages=messages,
                seed=seed,
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
            if attempt >= retries:
                logging.warning("OpenAI 호출 반복 실패(마지막 에러: %s). 빈약한 응답을 반환합니다.", e)
                return "I don't have enough information."
            # jittered backoff
            delay = (base_backoff ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
        except Exception as e:
            logging.warning("OpenAI 호출 중 비예상 예외: %s", e)
            return "I don't have enough information."


# ---------------------------------------------------------------------------
# LlamaIndex helpers
# ---------------------------------------------------------------------------

@dataclass
class LlamaIndexState:
    index: Optional[VectorStoreIndex]
    retriever: Optional[Any]
    top_k: int
    seen_docs: Set[str]

    def reset(self) -> None:
        self.index = None
        self.retriever = None
        self.seen_docs.clear()


def init_llamaindex_state(*, embed_model_name: str, gen_model: str, top_k: int) -> LlamaIndexState:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        query_instruction="task: search result | query: ",
        text_instruction="title: none | text: ",
        normalize=True,
    )
    Settings.llm = LIOpenAI(model=gen_model)
    return LlamaIndexState(index=None, retriever=None, top_k=int(top_k), seen_docs=set())


def ensure_llamaindex_doc(
    state: LlamaIndexState,
    item: QAItem,
    logger: Optional[BenchmarkLogger] = None,
    *,
    ack_client: _OpenAI,
    ack_model: str,
    seed: Optional[int],
) -> None:
    if item.doc_id in state.seen_docs:
        return

    chunks = [(idx, chunk.strip()) for idx, chunk in enumerate(item.chunks) if (chunk or "").strip()]
    ingested = 0
    for idx, text in chunks:
        _stream_chunk_ack(
            ack_client,
            chunk_text=text,
            doc_id=item.doc_id,
            chunk_index=idx,
            total_chunks=len(chunks),
            seed=seed,
            logger=logger,
            system="llama",
            qid=item.qid,
            ack_model=ack_model,
        )
        metadata = {"doc_id": item.doc_id, "title": item.title, "chunk_id": idx}
        document = Document(text=text, metadata=metadata)
        try:
            if state.index is None:
                state.index = VectorStoreIndex.from_documents([document])
            else:
                insert_many = getattr(state.index, "insert_documents", None)
                if callable(insert_many):
                    insert_many([document])
                else:
                    state.index.insert(document)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("LlamaIndex insert 실패(doc=%s): %s", metadata, exc)
            continue
        ingested += 1
        if logger:
            logger.log_ingest_chunk(
                system="llama",
                qid=item.qid,
                doc_id=item.doc_id,
                chunk_index=idx,
                total_chunks=len(chunks),
                text=text,
            )

    if state.index is None:
        return

    state.retriever = state.index.as_retriever(similarity_top_k=state.top_k)
    state.seen_docs.add(item.doc_id)
    if logger:
        logger.log_ingest_complete(
            system="llama",
            qid=item.qid,
            doc_id=item.doc_id,
            chunks_ingested=ingested,
        )


def retrieve_llamaindex(state: LlamaIndexState, question: str, *, max_char_per_ctx: int) -> List[str]:
    contexts: List[str] = []
    if state.retriever is None:
        return contexts
    try:
        nodes = state.retriever.retrieve(question)
    except Exception as exc:
        logging.warning("LlamaIndex retrieval 실패: %s", exc)
        return contexts

    for node in nodes[: state.top_k]:
        text = getattr(node, "text", None) or getattr(node, "node", None)
        if hasattr(text, "get_content"):
            text = text.get_content()
        if isinstance(text, str):
            contexts.append(_clip_text(text, max_char_per_ctx))
    return contexts


# ---------------------------------------------------------------------------
# HTNR helpers
# ---------------------------------------------------------------------------

def build_htnr_memory(*, embed_model_name: str,
                      max_leaves: int, ann_min_leaves: int, k_ctx: int,
                      seed: Optional[int]) -> HTNRMemory:
    embedder = HTNRGemmaEmbedder(model_name=embed_model_name, batch_size=32)
    hyper = Hyper(
        max_leaves=int(max_leaves),
        ann_min_leaves=int(ann_min_leaves),
        K_contexts=max(3, min(8, int(k_ctx))),
    )
    memory = HTNRMemory(embedder, hyper=hyper, numerics=Numerics(), seed=seed)
    return memory


def retrieve_htnr(memory: HTNRMemory, question: str, *, top_k: int, max_char_per_ctx: int) -> List[str]:
    try:
        contexts = memory.retrieve_for_query(question, K_cap=int(top_k))
    except Exception as exc:
        logging.warning("HTNR retrieval 실패: %s", exc)
        return []

    clipped = []
    for ctx in contexts:
        if isinstance(ctx, dict):
            text = ctx.get("content")
        else:
            text = ctx
        if isinstance(text, str) and text.strip():
            clipped.append(_clip_text(text, max_char_per_ctx))
    return clipped


def ingest_htnr_doc(
    memory: HTNRMemory,
    item: QAItem,
    seen_docs: Set[str],
    logger: Optional[BenchmarkLogger] = None,
    *,
    ack_client: _OpenAI,
    ack_model: str,
    seed: Optional[int],
) -> None:
    if item.doc_id in seen_docs:
        return

    ingested = 0
    filtered_chunks = [(idx, chunk.strip()) for idx, chunk in enumerate(item.chunks) if (chunk or "").strip()]
    total_chunks = len(filtered_chunks)
    for idx, text in filtered_chunks:
        _stream_chunk_ack(
            ack_client,
            chunk_text=text,
            doc_id=item.doc_id,
            chunk_index=idx,
            total_chunks=total_chunks,
            seed=seed,
            logger=logger,
            system="htnr",
            qid=item.qid,
            ack_model=ack_model,
        )
        try:
            memory.process_and_add(text, source=f"doc:{item.doc_id}#chunk{idx}")
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("HTNR ingest 실패(doc=%s chunk=%s): %s", item.doc_id, idx, exc)
            continue
        ingested += 1
        if logger:
            logger.log_ingest_chunk(
                system="htnr",
                qid=item.qid,
                doc_id=item.doc_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                text=text,
            )

    memory.flush_buffer()
    seen_docs.add(item.doc_id)
    if logger:
        logger.log_ingest_complete(
            system="htnr",
            qid=item.qid,
            doc_id=item.doc_id,
            chunks_ingested=ingested,
        )


# ---------------------------------------------------------------------------
# RAGAS helpers
# ---------------------------------------------------------------------------

def build_ragas_metrics(eval_model: str, include_similarity: bool):
    def _make_llm_wrapper() -> Any:
        llm = ChatOpenAI(model=eval_model)
        try:  # pragma: no cover - defensive, some versions lack attribute
            llm.max_retries = 6
        except Exception:
            pass
        return LangchainLLMWrapper(llm)

    if llm_factory:
        logging.debug("Ignoring ragas llm_factory to enforce temperature=1 for %s", eval_model)

    llm_provider = _make_llm_wrapper()
    metrics = [
        Faithfulness(llm=llm_provider),
        FactualCorrectness(llm=llm_provider),
        LLMContextRecall(llm=llm_provider),
        _AnswerRel(llm=llm_provider),
    ]
    if include_similarity:
        try:
            embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            from ragas.metrics import SemanticSimilarity
            metrics.append(SemanticSimilarity(embeddings=embeddings))
        except Exception as exc:
            logging.warning("SemanticSimilarity metric 비활성화 (사유: %s)", exc)
    return metrics


def make_ragas_dataset(rows: List[Dict[str, Any]]) -> EvaluationDataset:
    if not rows:
        raise ValueError("RAGAS 평가 입력이 비어 있습니다.")
    try:
        return EvaluationDataset.from_list(rows)  # type: ignore[arg-type]
    except AttributeError:
        payload = {
            "question": [row["question"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "ground_truth": [row["ground_truth"] for row in rows],
        }
        return EvaluationDataset.from_dict(payload)
    except Exception as exc:
        raise RuntimeError(f"RAGAS 데이터셋 구성 실패: {exc}") from exc


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_match(pred: str, truth: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(truth) else 0.0


def token_f1(pred: str, truth: str) -> float:
    pred_tokens = normalize_text(pred).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), truth_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text.strip()
    return text[:limit].rstrip() + " …"


# ---------------------------------------------------------------------------
# Async generation runner
# ---------------------------------------------------------------------------

async def run_generation_async(
    jobs: List[GenJob],
    *,
    model: str,
    max_completion_tokens: int,
    seed: Optional[int],
    total_concurrency: int,
    per_system_concurrency: Optional[int],
    system_order: Sequence[str],
    logger: Optional[BenchmarkLogger] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute OpenAI generations asynchronously with (a) a global concurrency cap and
    (b) an optional per-system cap.
    """
    job_systems = {job.system for job in jobs}
    systems_present = [s for s in system_order if s in job_systems]
    if not systems_present:
        systems_present = sorted(job_systems)
    aclient = _AsyncOpenAI()

    global_sem = asyncio.Semaphore(max(1, int(total_concurrency)))
    system_sems: Dict[str, asyncio.Semaphore] = {}
    if per_system_concurrency is not None:
        cap = max(1, int(per_system_concurrency))
        for system in systems_present:
            system_sems[system] = asyncio.Semaphore(cap)

    async def _generate_one(job: GenJob) -> Tuple[GenJob, str]:
        # nesting semaphores: global first, then per-system if present
        if system_sems:
            async with global_sem, system_sems[job.system]:
                ans = await build_answer_async(
                    model, aclient, job.question, job.contexts,
                    max_completion_tokens=max_completion_tokens,
                    seed=seed,
                )
        else:
            async with global_sem:
                ans = await build_answer_async(
                    model, aclient, job.question, job.contexts,
                    max_completion_tokens=max_completion_tokens,
                    seed=seed,
                )
        return job, ans

    tasks = [_generate_one(job) for job in jobs]

    results_map: Dict[str, List[Dict[str, Any]]] = {system: [] for system in systems_present}
    pbar = tqdm(total=len(tasks), desc="Generating (async)", unit="req")
    for coro in asyncio.as_completed(tasks):
        job, answer = await coro
        row = {
            "qid": job.qid,
            "question": job.question,
            "contexts": job.contexts,
            "answer": answer,
            "ground_truth": job.ground_truth,
            "em": exact_match(answer, job.ground_truth),
            "f1": token_f1(answer, job.ground_truth),
        }
        results_map.setdefault(job.system, []).append(row)
        if logger:
            logger.log_generation(
                job.system,
                row,
                user_prompt=_build_user_prompt(job.question, job.contexts),
            )
        pbar.update(1)
    pbar.close()
    # prune empty systems
    return {k: v for k, v in results_map.items() if v}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HTNR vs LlamaIndex vs No-RAG on NarrativeQA with RAGAS (async generation supported)"
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--nqa-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["none", "llama", "htnr"],
        choices=["none", "llama", "htnr"],
    )
    parser.add_argument("--k", type=int, default=8, help="retrieval top-k for LlamaIndex/HTNR")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=2048,
        help="max tokens per NarrativeQA chunk (0=use full document)",
    )
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--embedder-model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--gen-model", type=str, default="gpt-5-mini")
    parser.add_argument("--eval-model", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1000,
        help="max tokens for generator completions",
    )
    parser.add_argument(
        "--use-embed-metric",
        action="store_true",
        help="include SemanticSimilarity metric via OpenAIEmbeddings",
    )
    parser.add_argument("--out-dir", type=str, default="bench_outputs")
    parser.add_argument(
        "--retain-context",
        action="store_true",
        help="keep ingested NarrativeQA documents in retrieval memories across questions",
    )
    parser.add_argument(
        "--allow-summary-fallback",
        action="store_true",
        help="use NarrativeQA summaries when full document text is unavailable",
    )
    parser.add_argument(
        "--select-longest",
        action="store_true",
        help="pick the longest --nqa-count samples (by document length) instead of random sampling",
    )
    # NEW: concurrency controls
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="global max number of concurrent OpenAI generation calls (e.g., 30)",
    )
    parser.add_argument(
        "--per-system-concurrency",
        type=int,
        default=None,
        help="optional cap per system (e.g., 10). If unset, only the global cap applies.",
    )

    args = parser.parse_args()

    forget_after_question = not args.retain_context

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = os.path.join(args.out_dir, f"{timestamp}_logs")
    logger = BenchmarkLogger(log_root)
    ack_client = _OpenAI()
    ack_model = args.gen_model

    print(f"[INFO] Loading NarrativeQA split={args.split}, n={args.nqa_count} …")
    items = load_narrativeqa(
        args.split,
        take=args.nqa_count,
        seed=args.seed,
        chunk_tokens=args.chunk_tokens,
        allow_summary_fallback=args.allow_summary_fallback,
        select_longest=args.select_longest,
    )

    # ----------------------------
    # Phase 1. Prepare contexts (sync)
    # ----------------------------
    print("[INFO] Preparing contexts (sync retrieval) …")
    jobs: List[GenJob] = []

    # Baseline (no retrieval)
    if "none" in args.systems:
        for item in items:
            jobs.append(GenJob(system="none", qid=item.qid, question=item.question,
                               contexts=[], ground_truth=item.ground_truth))

    # LlamaIndex retrieval (per-item)
    if "llama" in args.systems:
        print("[INFO] Preparing LlamaIndex (lazy ingest per item) …")
        llama_state = init_llamaindex_state(
            embed_model_name=args.embedder_model,
            gen_model=args.gen_model,
            top_k=args.k,
        )
        for item in items:
            ensure_llamaindex_doc(
                llama_state,
                item,
                logger,
                ack_client=ack_client,
                ack_model=ack_model,
                seed=args.seed,
            )
            contexts = retrieve_llamaindex(
                llama_state,
                item.question,
                max_char_per_ctx=args.max_context_chars,
            )
            jobs.append(GenJob(system="llama", qid=item.qid, question=item.question,
                               contexts=contexts, ground_truth=item.ground_truth))
            if forget_after_question:
                llama_state.reset()

    # HTNR retrieval (per-item)
    if "htnr" in args.systems:
        print("[INFO] Preparing HTNR memory (lazy ingest per item) …")
        htnr_memory: Optional[HTNRMemory] = None
        htnr_seen_docs: Set[str] = set()
        for item in items:
            if htnr_memory is None:
                htnr_memory = build_htnr_memory(
                    embed_model_name=args.embedder_model,
                    max_leaves=max(64, args.k * 10),
                    ann_min_leaves=max(32, args.k * 4),
                    k_ctx=args.k,
                    seed=args.seed,
                )
                htnr_seen_docs = set()
            ingest_htnr_doc(
                htnr_memory,
                item,
                htnr_seen_docs,
                logger,
                ack_client=ack_client,
                ack_model=ack_model,
                seed=args.seed,
            )
            contexts = retrieve_htnr(
                htnr_memory,
                item.question,
                top_k=args.k,
                max_char_per_ctx=args.max_context_chars,
            )
            jobs.append(GenJob(system="htnr", qid=item.qid, question=item.question,
                               contexts=contexts, ground_truth=item.ground_truth))
            if forget_after_question:
                htnr_memory = None
                htnr_seen_docs = set()

    # ----------------------------
    # Phase 2. Generate answers (async)
    # ----------------------------
    print(f"[INFO] Async generation: global concurrency={args.concurrency}, "
          f"per-system={args.per_system_concurrency}")
    # We don't need the sync OpenAI client anymore for generation, but keep for compatibility
    # client = _OpenAI()

    results = asyncio.run(run_generation_async(
        jobs,
        model=args.gen_model,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed,
        total_concurrency=args.concurrency,
        per_system_concurrency=args.per_system_concurrency,
        system_order=args.systems,
        logger=logger,
    ))

    # ----------------------------
    # Phase 3. Save generations
    # ----------------------------
    for system, rows in results.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.out_dir, f"{timestamp}_{system}_generations.csv")
        df.to_csv(out_path, index=False)
        print(f"[SAVE] {system} generations -> {out_path}")

    # ----------------------------
    # Phase 4. RAGAS evaluation (sequential for stability)
    # ----------------------------
    metrics = build_ragas_metrics(args.eval_model, args.use_embed_metric)
    scoreboard: Dict[str, Dict[str, float]] = {}

    for system, rows in results.items():
        if not rows:
            continue
        ragas_rows = [{
            "question": row["question"],
            "contexts": row["contexts"],
            "answer": row["answer"],
            "ground_truth": row["ground_truth"],
            "user_input": row["question"],
            "retrieved_contexts": row["contexts"],
            "response": row["answer"],
            "reference": row["ground_truth"],
        } for row in rows]

        dataset = make_ragas_dataset(ragas_rows)
        print(f"[EVAL] RAGAS evaluating: {system} (n={len(ragas_rows)}) …")
        ragas_result = evaluate(dataset=dataset, metrics=metrics)
        ragas_df = ragas_result.to_pandas()
        ragas_out = os.path.join(args.out_dir, f"{timestamp}_{system}_ragas.csv")
        ragas_df.to_csv(ragas_out, index=False)
        print(f"[SAVE] {system} ragas -> {ragas_out}")

        means: Dict[str, float] = {}
        for column in ragas_df.columns:
            if column in {"user_input", "retrieved_contexts", "reference_contexts", "response", "reference"}:
                continue
            series = pd.to_numeric(ragas_df[column], errors="coerce")
            value = float(series.mean()) if not series.empty else float("nan")
            if not math.isnan(value):
                means[column] = value

        em_scores = [row.get("em", 0.0) for row in rows]
        f1_scores = [row.get("f1", 0.0) for row in rows]
        if em_scores:
            means["exact_match"] = float(np.mean(em_scores))
        if f1_scores:
            means["token_f1"] = float(np.mean(f1_scores))
        scoreboard[system] = means

    if scoreboard:
        print("\n=== Scoreboard (means) ===")
        order = [
            "faithfulness",
            "factual_correctness",
            "llm_context_recall",
            "semantic_similarity",
            "answer_relevancy",
            "exact_match",
            "token_f1",
        ]
        for system in [s for s in ["none", "llama", "htnr"] if s in results]:
            values = scoreboard.get(system, {})
            parts = [f"{metric}={values[metric]:.3f}" for metric in order if metric in values]
            print(f"[{system}] " + ", ".join(parts))

        scoreboard_path = os.path.join(args.out_dir, f"{timestamp}_scoreboard.json")
        with open(scoreboard_path, "w", encoding="utf-8") as fp:
            json.dump(scoreboard, fp, ensure_ascii=False, indent=2)
        print(f"[SAVE] scoreboard -> {scoreboard_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
