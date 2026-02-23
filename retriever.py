"""
retriever.py
BGE-M3 임베딩, ChromaDB, BM25, CrossEncoder 로드 및 앙상블 검색 함수.

모든 무거운 리소스는 @st.cache_resource로 캐싱하여
Streamlit 재실행 시 재로드를 막습니다.
"""

import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from processor import (
    clear_gpu,
    get_kiwi_tokenizer,
    tokenize_corpus,
    tokenize_query,
)
from generator import _retry_api_call
from external_pharma_api import fetch_external_pharma_docs

logger = logging.getLogger(__name__)


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


def load_reranker() -> CrossEncoder:
    """
    BAAI/bge-reranker-v2-m3 리랭킹 모델을 로드합니다.
    GPU: fp16 + max_length 768.
    """
    device = _get_device()
    reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    reranker = CrossEncoder(
        reranker_model_name,
        device=device,
        max_length=768,          # 512 → 768: 더 많은 컨텍스트 반영
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


def build_bm25_retriever(db_path: str) -> tuple[BM25Retriever, object]:
    """
    ChromaDB에서 전체 문서를 가져와 Kiwi 형태소 분석 후 BM25 인덱스를 구축합니다.

    Args:
        db_path: ChromaDB 경로

    Returns:
        (bm25_retriever, kiwi) 튜플
    """
    t0 = time.time()
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
    use_external_api: bool = False,
    external_provider: str = "openfda",
    external_top_k: int = 4,
    external_timeout_sec: float = 8.0,
    weight_external: float = 0.2,
) -> list[Document]:
    """
    콜랩 원본 코드를 참고한 지능형 앙상블 검색 실행.
    - MMR 검색으로 다양성 확보
    - 특정 태그(N, X, S, VA) 기반 키워드 필터링
    """
    # 1. Query Expansion (Query Expansion)
    if search_keywords is None:
        if query_optimizer is not None:
            optimize_prompt = (
                f"다음 질문에서 약학 검색에 필요한 핵심 성분명, 증상, 질환 키워드만 뽑아 공백으로 나열해줘: {query}"
            )
            try:
                search_keywords = _retry_api_call(query_optimizer.invoke, optimize_prompt)
            except Exception:
                search_keywords = query
        else:
            search_keywords = query

    # 2. 키워드 토큰화 (Colab 원본 코드 기준 태그 필터링)
    query_tokens = [
        t.form
        for t in kiwi.tokenize(search_keywords)
        if t.tag.startswith(('N', 'X', 'S', 'VA'))
    ]
    query_text = " ".join(query_tokens)

    # 3. 각 검색 수행 (BM25 + MMR 벡터 검색 + 외부 API 병렬 처리)
    def _bm25_search():
        # BM25 리트리버를 통해 키워드 검색
        return bm25_retriever.invoke(query_text)

    def _vector_search():
        # MMR(Maximum Marginal Relevance) 검색으로 결과의 다양성 확보
        # fetch_k: 후보군 추출 수, lambda_mult: 다양성 지수 (0에 가까울수록 다양)
        return vector_db.max_marginal_relevance_search(
            f"query: {query}", 
            k=k, 
            fetch_k=min(k * 2, 40), 
            lambda_mult=0.5
        )

    def _external_search():
        if not use_external_api:
            return []
        try:
            return fetch_external_pharma_docs(
                query=query,
                provider=external_provider,
                top_k=external_top_k,
                timeout_sec=external_timeout_sec,
            )
        except Exception as e:
            logger.warning("[ExternalAPI] 문서 조회 실패: %s", e)
            return []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_bm25 = executor.submit(_bm25_search)
        future_vector = executor.submit(_vector_search)
        future_external = executor.submit(_external_search)
        keyword_docs = future_bm25.result()
        vector_docs = future_vector.result()
        external_docs = future_external.result()

    # 4. RRF(Reciprocal Rank Fusion) 통합 및 데이터 타입 교정
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
    _process_results(external_docs, max(0.0, float(weight_external)))

    if use_external_api:
        logger.info(
            "[ExternalAPI] provider=%s, fetched=%d, weight=%.2f",
            external_provider,
            len(external_docs),
            weight_external,
        )

    # 5. 최종 결과 정렬 및 k개 반환
    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[content] for content, score in sorted_items[:k]]


# ──────────────────────────────────────────────────────────────────────
# CrossEncoder 리랭킹 — GPU fp16 고속 처리
# ──────────────────────────────────────────────────────────────────────

def rerank_docs(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_k: int = 5,
    batch_size: int = 64,   # 32 → 64: RTX 2070 8GB 기준 최적값
) -> list[tuple[float, Document]]:
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

    # (쿼리, 정제된 문서텍스트) 쌍 구성
    pairs = [[query, clean_json_to_text(doc.page_content)] for doc in docs]

    # GPU fp16 AMP 적용하여 예측
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        scores = reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return list(reranked[:top_k])
