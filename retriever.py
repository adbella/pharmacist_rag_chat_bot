"""
retriever.py
BGE-M3 임베딩, ChromaDB, BM25, CrossEncoder 로드 및 앙상블 검색 함수.

모든 무거운 리소스는 @st.cache_resource로 캐싱하여
Streamlit 재실행 시 재로드를 막습니다.
"""

import time
import logging
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from processor import (
    get_kiwi_tokenizer,
    tokenize_corpus,
)
from generator import _retry_api_call

logger = logging.getLogger(__name__)


# 증상 중심 질문에서 검색 누락을 줄이기 위한 경량 확장 사전
_QUERY_HINTS: dict[str, list[str]] = {
    "눈": ["안구건조", "눈 피로", "루테인", "오메가3"],
    "침침": ["시야흐림", "눈 피로", "루테인", "오메가3"],
    "건조": ["안구건조", "인공눈물", "오메가3"],
    "시야": ["시야흐림", "루테인"],
}

_CONTEXT_ROUTER_ENABLED = os.getenv("CONTEXT_QUERY_ROUTER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _expand_query_for_retrieval(query: str) -> str:
    """질의에 증상 키워드가 있을 때 검색 힌트를 덧붙여 리콜을 보강합니다."""
    q = (query or "").strip()
    if not q:
        return q

    q_norm = q.lower()
    terms: list[str] = [q]
    for key, hints in _QUERY_HINTS.items():
        if key in q_norm:
            terms.extend(hints)

    # 기본 토큰도 일부 보존(짧은 조사/어미 제외)
    terms.extend(re.findall(r"[가-힣a-zA-Z0-9\-]{2,}", q))

    dedup: list[str] = []
    seen: set[str] = set()
    for t in terms:
        s = (t or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(s)

    return " ".join(dedup)


def _extract_router_terms(raw: str) -> list[str]:
    """LLM 라우터 응답에서 검색어 후보를 추출합니다."""
    text = (raw or "").strip()
    if not text:
        return []

    # 1) JSON 직접 파싱 시도
    try:
        obj = json.loads(text)
    except Exception:
        obj = None

    if isinstance(obj, dict):
        terms: list[str] = []
        for key in ("core_terms", "recall_terms", "terms", "keywords"):
            val = obj.get(key)
            if isinstance(val, list):
                for x in val:
                    if isinstance(x, str) and x.strip():
                        terms.append(x.strip())
        if terms:
            return terms

    # 2) 코드블록/문장 응답 fallback: 공백/콤마 분리
    fallback = re.split(r"[\s,;/|]+", text)
    return [t.strip() for t in fallback if t.strip()]


def _build_context_router_keywords(query: str, query_optimizer) -> str:
    """규칙 기반 확장 + (선택) LLM 맥락 라우팅을 결합해 검색어를 생성합니다."""
    base_terms = _expand_query_for_retrieval(query).split()

    if not _CONTEXT_ROUTER_ENABLED or query_optimizer is None:
        return " ".join(base_terms)

    router_prompt = (
        "너는 약학 검색 라우터다. 사용자 문장을 읽고 검색 리콜에 유리한 핵심어를 JSON으로만 출력해.\n"
        "출력 형식: {\"core_terms\":[...],\"recall_terms\":[...],\"avoid_terms\":[...]}\n"
        "규칙:\n"
        "- core_terms: 질문의 핵심 의도/증상/제형\n"
        "- recall_terms: 동의어/관련 성분/관련 카테고리\n"
        "- avoid_terms: 추측이 강하거나 비관련 용어\n"
        "- 최대 12개 이내, 짧은 토큰 중심\n"
        f"질문: {query}"
    )

    try:
        llm_raw = _retry_api_call(query_optimizer.invoke, router_prompt)
        llm_terms = _extract_router_terms(llm_raw)
    except Exception as e:
        logger.warning("[QueryRouter] LLM routing failed: %s", e)
        llm_terms = []

    merged: list[str] = []
    seen: set[str] = set()
    for term in base_terms + llm_terms:
        t = (term or "").strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        merged.append(t)

    logger.info("[QueryRouter] query='%s' expanded_terms=%s", query, merged[:18])
    return " ".join(merged)


# ──────────────────────────────────────────────────────────────────────
# GPU/CPU 자동 선택 헬퍼
# ──────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_gpu() -> None:
    """RTX 2070 최적 CUDA 설정."""
    if torch.cuda.is_available():
        # Tensor Core 활용 (Turing 이상)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # matmul 정밀도 – TF32 허용으로 속도 향상
        torch.set_float32_matmul_precision("high")


# 앱 기동 시 1회 실행
_configure_gpu()


# ──────────────────────────────────────────────────────────────────────
# 리소스 로더 (@st.cache_resource → 최초 1회만 실행)
# ──────────────────────────────────────────────────────────────────────

def load_embeddings() -> HuggingFaceEmbeddings:
    """
    BGE-M3-ko 임베딩 모델을 로드합니다.
    GPU: fp16 + batch_size 128로 고속 인코딩.
    CPU: fp32 폴백.
    """
    device = _get_device()
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "dragonkue/BGE-m3-ko")

    # GPU면 fp16, CPU면 fp32
    encode_kwargs: dict = {
        "normalize_embeddings": True,
        "batch_size": 128 if device == "cuda" else 32,
    }
    # langchain_huggingface는 model_kwargs를 **kwargs로 SentenceTransformer에 전달.
    # SentenceTransformer 5.x는 torch_dtype을 최상위 kwarg로 받지 않고,
    # 자체 model_kwargs 파라미터(HuggingFace AutoModel에 전달)로 받아야 함.
    model_kwargs: dict = {"device": device}
    if device == "cuda":
        # SentenceTransformer(model_name, device="cuda", model_kwargs={"torch_dtype": fp16})
        model_kwargs["model_kwargs"] = {"dtype": torch.float16}

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    logger.info(
        "[Embeddings] 로드 완료 (model=%s, device=%s, fp16=%s)",
        embedding_model_name,
        device,
        device == "cuda",
    )
    return embeddings


def load_vector_db(db_path: str) -> Chroma:
    """
    로컬 ChromaDB를 로드합니다.

    Args:
        db_path: persist_directory 경로 (예: './chroma_db_combined_stored')
    """
    embeddings = load_embeddings()
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    count = vector_db._collection.count()
    logger.info("[ChromaDB] 로드 완료: %d 문서", count)
    return vector_db


def load_vector_db_with_embeddings(db_path: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """이미 로드한 임베딩 객체를 재사용해 ChromaDB를 로드합니다."""
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    count = vector_db._collection.count()
    logger.info("[ChromaDB] 로드 완료: %d 문서", count)
    return vector_db


def load_reranker() -> CrossEncoder:
    """
    BAAI/bge-reranker-v2-m3 리랭킹 모델을 로드합니다.
    GPU: fp16 + max_length 768.
    """
    device = _get_device()
    reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    rerank_max_length = int(os.getenv("RERANK_MAX_LENGTH", "512"))
    reranker = CrossEncoder(
        reranker_model_name,
        device=device,
        max_length=rerank_max_length,
        model_kwargs={
            "dtype": torch.float16 if device == "cuda" else torch.float32,
        },
    )
    logger.info(
        "[CrossEncoder] 로드 완료 (model=%s, device=%s, fp16=%s)",
        reranker_model_name,
        device,
        device == "cuda",
    )
    return reranker


def build_bm25_retriever(db_path: str | None = None, vector_db: Chroma | None = None) -> tuple[BM25Retriever, object]:
    """
    ChromaDB에서 전체 문서를 가져와 Kiwi 형태소 분석 후 BM25 인덱스를 구축합니다.

    Args:
        db_path: ChromaDB 경로

    Returns:
        (bm25_retriever, kiwi) 튜플
    """
    t0 = time.time()
    if vector_db is None:
        if not db_path:
            raise ValueError("build_bm25_retriever requires either db_path or vector_db")
        vector_db = load_vector_db(db_path)

    # 1. 전체 DB 데이터 추출
    all_data = vector_db.get()
    original_docs = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    logger.info("[BM25] DB 추출 완료: %d 문서, %.2fs", len(original_docs), time.time() - t0)

    # 2. Kiwi 초기화 및 형태소 분석 (멀티코어 배치 처리)
    t1 = time.time()
    kiwi = get_kiwi_tokenizer()
    tokenized_corpus = tokenize_corpus(original_docs, kiwi)
    logger.info("[BM25] Kiwi 토크나이징 완료: %.2fs", time.time() - t1)

    # 3. BM25 리트리버 구축 (이미 토큰화된 텍스트를 공백 조인해 전달,
    #    preprocess_func으로 재분리하여 rank_bm25의 내부 코퍼스에 적재)
    t2 = time.time()
    joined_texts = [" ".join(tokens) for tokens in tokenized_corpus]
    bm25_retriever = BM25Retriever.from_texts(
        joined_texts,
        metadatas=[doc.metadata for doc in original_docs],
        preprocess_func=lambda x: x.split(),  # 공백 split → 토큰 리스트 복원
        k=35,
    )
    # 원본 Document 복원 (BM25Retriever가 반환할 때 사용)
    bm25_retriever.docs = original_docs
    logger.info("[BM25] 인덱스 구축 완료: %.2fs", time.time() - t2)
    logger.info("[BM25] 전체 초기화 소요: %.2fs", time.time() - t0)

    return bm25_retriever, kiwi


# ──────────────────────────────────────────────────────────────────────
# 앙상블 검색 (BM25 + Vector, RRF 통합) — 병렬 실행 버전
# ──────────────────────────────────────────────────────────────────────

def get_ensemble_results(
    query: str,
    kiwi,
    bm25_retriever: BM25Retriever,
    vector_db: Chroma,
    query_optimizer=None,
    search_keywords: str | None = None,
    k: int = 20,
    weight_bm25: float = 0.8,
    weight_vector: float = 0.2,
    return_metrics: bool = False,
) -> list[Document] | tuple[list[Document], dict[str, float]]:
    """
    콜랩 원본 코드를 참고한 지능형 앙상블 검색 실행.
    - MMR 검색으로 다양성 확보
    - 특정 태그(N, X, S, VA) 기반 키워드 필터링
    """
    t0 = time.time()
    metrics = {
        "router_s": 0.0,
        "tokenize_s": 0.0,
        "bm25_s": 0.0,
        "vector_s": 0.0,
        "fusion_s": 0.0,
        "search_total_s": 0.0,
    }

    # 1. Query Expansion (Query Expansion)
    t_router = time.time()
    if search_keywords is None:
        search_keywords = _build_context_router_keywords(query, query_optimizer)
    metrics["router_s"] = time.time() - t_router

    # 2. 키워드 토큰화 (Colab 원본 코드 기준 태그 필터링)
    t_tok = time.time()
    query_tokens = [
        t.form
        for t in kiwi.tokenize(search_keywords)
        if t.tag.startswith(('N', 'X', 'S', 'VA'))
    ]
    query_text = " ".join(query_tokens)
    metrics["tokenize_s"] = time.time() - t_tok

    # 3. 각 검색 수행 (BM25 + MMR 벡터 검색 병렬 처리)
    def _bm25_search():
        t = time.time()
        # BM25 리트리버를 통해 키워드 검색
        docs = bm25_retriever.invoke(query_text)
        return docs, time.time() - t

    def _vector_search():
        t = time.time()
        # MMR(Maximum Marginal Relevance) 검색으로 결과의 다양성 확보
        # fetch_k: 후보군 추출 수, lambda_mult: 다양성 지수 (0에 가까울수록 다양)
        docs = vector_db.max_marginal_relevance_search(
            f"query: {query}", 
            k=k, 
            fetch_k=min(k * 2, 40), 
            lambda_mult=0.5
        )
        return docs, time.time() - t

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25 = executor.submit(_bm25_search)
        future_vector = executor.submit(_vector_search)
        keyword_docs, metrics["bm25_s"] = future_bm25.result()
        vector_docs, metrics["vector_s"] = future_vector.result()

    # 4. RRF(Reciprocal Rank Fusion) 통합 및 데이터 타입 교정
    t_fusion = time.time()
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def _process_results(docs, weight: float) -> None:
        for rank, doc in enumerate(docs):
            if isinstance(doc, Document):
                content = doc.page_content
                doc_obj = doc
            elif isinstance(doc, list):
                content = " ".join(doc)
                doc_obj = Document(page_content=content, metadata={"source": "BM25_Index"})
            else:
                content = str(doc)
                doc_obj = Document(page_content=content, metadata={"source": "Unknown"})

            if content not in doc_scores:
                doc_map[content] = doc_obj
                doc_scores[content] = 0.0
            doc_scores[content] += weight * (1 / (rank + 60))

    _process_results(keyword_docs, weight_bm25)
    _process_results(vector_docs, weight_vector)

    # 5. 최종 결과 정렬 및 k개 반환
    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    results = [doc_map[content] for content, score in sorted_items[:k]]
    metrics["fusion_s"] = time.time() - t_fusion
    metrics["search_total_s"] = time.time() - t0
    if return_metrics:
        return results, metrics
    return results


# ──────────────────────────────────────────────────────────────────────
# CrossEncoder 리랭킹 — GPU fp16 고속 처리
# ──────────────────────────────────────────────────────────────────────

def rerank_docs(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_k: int = 5,
    batch_size: int = 64,
    return_metrics: bool = False,
) -> list[tuple[float, Document]] | tuple[list[tuple[float, Document]], dict[str, float]]:
    """
    CrossEncoder로 앙상블 결과를 정밀 리랭킹합니다.
    GPU fp16 모드에서 batch_size=64로 처리량 최대화.

    Args:
        query:      사용자 원본 질문
        docs:       앙상블 검색 결과 Document 리스트
        reranker:   로드된 CrossEncoder 인스턴스
        top_k:      최종 반환 문서 수
        batch_size: CrossEncoder 배치 사이즈 (RTX 2070: 64 권장)

    Returns:
        리랭킹된 상위 top_k개 (score, Document) 튜플 리스트
    """
    from processor import clean_json_to_text

    t0 = time.time()
    # (쿼리, 정제된 문서텍스트) 쌍 구성
    t_prepare = time.time()
    pairs = [[query, clean_json_to_text(doc.page_content)] for doc in docs]
    prepare_s = time.time() - t_prepare

    # GPU fp16 AMP 적용하여 예측
    t_infer = time.time()
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        scores = reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    infer_s = time.time() - t_infer

    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_items = list(reranked[:top_k])
    if return_metrics:
        return top_items, {
            "rerank_prepare_s": prepare_s,
            "rerank_infer_s": infer_s,
            "rerank_total_s": time.time() - t0,
            "rerank_batch_size": float(batch_size),
        }
    return top_items
