"""
api.py
FastAPI 백엔드 – 기존 RAG 파이프라인을 REST/SSE 엔드포인트로 노출합니다.

실행: python api.py
     (혹은 uvicorn api:app --host 0.0.0.0 --port 8000 --reload)
"""

import os
import time
import json
import math
import asyncio
import logging
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv

# ── CUDA 최적화 플래그 ────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

from retriever import (
    load_embeddings,
    load_vector_db,
    load_reranker,
    build_bm25_retriever,
    get_ensemble_results,
    rerank_docs,
)
from external_pharma_api import fetch_external_pharma_docs, get_external_api_status
from generator import (
    build_context,
    generate_answer,
    verify_answer,
    self_correction_loop,
    get_query_optimizer,
    evaluate_with_ragas,
)
from processor import clear_gpu, get_gpu_status

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 백그라운드에서 리소스 로드 및 종료 시 정리."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    
    # 리소스 비동기 로드 시작
    await loop.run_in_executor(None, _load_all_resources)
    
    yield
    
    # 종료 시 필요하다면 정리 로직 추가 가능
    _resources.clear()

# ─────────────────────────────────────────────────────────────────────
# App 설정
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="약사 AI 챗봇 API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 상수 ─────────────────────────────────────────────────────────────
DB_PATH = "./chroma_db_combined_1771477980"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OOS_GUARD_ENABLED = os.getenv("OOS_GUARD_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
OOS_MIN_RELEVANCE = float(os.getenv("OOS_MIN_RELEVANCE", "0.55"))
OOS_MIN_TOP_SCORE = float(os.getenv("OOS_MIN_TOP_SCORE", "0.002"))
USE_QUERY_OPTIMIZER = os.getenv("USE_QUERY_OPTIMIZER", "false").lower() in {"1", "true", "yes", "on"}
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "gpt-5.2")
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "gpt-5.2")
USE_EXTERNAL_API = os.getenv("USE_EXTERNAL_API", "false").lower() in {"1", "true", "yes", "on"}
EXTERNAL_API_PROVIDER = os.getenv("EXTERNAL_API_PROVIDER", "openfda")
EXTERNAL_TOP_K = int(os.getenv("EXTERNAL_TOP_K", "4"))
EXTERNAL_TIMEOUT_SEC = float(os.getenv("EXTERNAL_TIMEOUT_SEC", "8"))
WEIGHT_EXTERNAL = float(os.getenv("WEIGHT_EXTERNAL", "0.2"))
EXTERNAL_FALLBACK_ONLY = os.getenv("EXTERNAL_FALLBACK_ONLY", "true").lower() in {"1", "true", "yes", "on"}
EXTERNAL_TRIGGER_MIN_RELEVANCE = float(os.getenv("EXTERNAL_TRIGGER_MIN_RELEVANCE", "0.55"))
EXTERNAL_TRIGGER_MIN_TOP_SCORE = float(os.getenv("EXTERNAL_TRIGGER_MIN_TOP_SCORE", "0.002"))
EXTERNAL_TRIGGER_MODE = os.getenv("EXTERNAL_TRIGGER_MODE", "or").strip().lower()  # or / and

# ── 전역 상태 ────────────────────────────────────────────────────────
_resources: dict = {}
_init_done = False
_init_logs: list[str] = []

def _log_init(msg: str):
    logger.info(msg)
    _init_logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

def _load_all_resources():
    global _resources, _init_done
    if _init_done:
        return
    _log_init("리소스 초기화 시작...")
    _resources["embeddings"]     = load_embeddings()
    _log_init("임베딩 모델 로드 완료")
    _resources["vector_db"]      = load_vector_db(DB_PATH)
    _log_init("벡터 데이터베이스 연결 완료")
    _resources["reranker"]       = load_reranker()
    _log_init("리랭커 모델 로드 완료")
    _resources["bm25"], _resources["kiwi"] = build_bm25_retriever(DB_PATH)
    _log_init("BM25 인덱스 생성 완료")
    _resources["query_optimizer"] = get_query_optimizer(OPENAI_API_KEY) if USE_QUERY_OPTIMIZER else None
    _log_init("쿼리 최적화기 준비 완료" if USE_QUERY_OPTIMIZER else "쿼리 최적화기 비활성화(속도 우선)")
    _init_done = True
    _log_init("모든 리소스 준비 완료!")


# ─────────────────────────────────────────────────────────────────────
# 스키마
# ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    model: str = "gpt-5"
    top_k: int = 5
    ensemble_k: int = 20
    weight_bm25: float = 0.8
    use_self_correction: bool = True


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """서버 상태 및 GPU VRAM 정보."""
    gpu_info = {"available": False}
    if torch.cuda.is_available():
        gpu_info = get_gpu_status()
    return {
        "status": "ready" if _init_done else "initializing",
        "init_logs": _init_logs[-5:],
        "gpu": gpu_info,
        "external_api": get_external_api_status(EXTERNAL_API_PROVIDER),
    }


@app.post("/clear-memory")
async def clear_memory():
    """GPU 캐시를 강제로 비웁니다."""
    logger.info("GPU 메모리 정리 요청 수신")
    clear_gpu()
    return await health()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """브라우저의 자동 favicon 요청에 대해 204 No Content로 응답하여 로그를 깔끔하게 유지합니다."""
    return Response(status_code=204)


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """SSE 스트리밍 채팅 엔드포인트."""
    if not _init_done:
        raise HTTPException(503, "리소스 초기화 중입니다. 잠시 후 다시 시도하세요.")
    if not OPENAI_API_KEY:
        raise HTTPException(400, ".env에 OPENAI_API_KEY가 없습니다.")

    async def generate() -> AsyncGenerator[str, None]:
        total_start = time.time()
        verify_task = None
        ragas_task = None
        search_elapsed = 0.0
        rerank_elapsed = 0.0
        gen_elapsed = 0.0
        verify_elapsed = 0.0
        ensemble_docs = []
        final_docs = []
        docs_payload: list[dict] = []
        external_docs_count = 0
        external_fallback_triggered = False
        external_trigger_reason: list[str] = []

        async def _cancel_pending_tasks():
            pending = [t for t in (verify_task, ragas_task) if t is not None and not t.done()]
            if pending:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def _external_diag() -> dict:
            try:
                return get_external_api_status(EXTERNAL_API_PROVIDER)
            except Exception:
                return {
                    "provider": EXTERNAL_API_PROVIDER,
                    "connected": None,
                    "message": "status_probe_failed",
                }

        try:
            # ── 1. 앙상블 검색 ──────────────────────────────
            yield _sse("status", {"step": "검색 중...", "icon": "🔍"})
            search_start = time.time()
            ensemble_docs = await loop.run_in_executor(
                None,
                lambda: get_ensemble_results(
                    query=req.query,
                    kiwi=_resources["kiwi"],
                    bm25_retriever=_resources["bm25"],
                    vector_db=_resources["vector_db"],
                    query_optimizer=_resources.get("query_optimizer"),
                    k=req.ensemble_k,
                    weight_bm25=req.weight_bm25,
                    weight_vector=round(1.0 - req.weight_bm25, 2),
                    use_external_api=bool(USE_EXTERNAL_API and not EXTERNAL_FALLBACK_ONLY),
                    external_provider=EXTERNAL_API_PROVIDER,
                    external_top_k=EXTERNAL_TOP_K,
                    external_timeout_sec=EXTERNAL_TIMEOUT_SEC,
                    weight_external=WEIGHT_EXTERNAL,
                ),
            )
            search_elapsed = time.time() - search_start

            # ── 2. 리랭킹 ───────────────────────────────────
            yield _sse("status", {"step": f"{len(ensemble_docs)}개 문서 리랭킹 중...", "icon": "⚡"})
            rerank_start = time.time()
            ranked_pairs = await loop.run_in_executor(
                None,
                lambda: rerank_docs(
                    query=req.query,
                    docs=ensemble_docs,
                    reranker=_resources["reranker"],
                    top_k=req.top_k,
                    batch_size=64, # RTX 2070 8GB 최적 배치 사이즈 상향
                ),
            )
            rerank_elapsed = time.time() - rerank_start

            rerank_scores = [s for s, _ in ranked_pairs]
            final_docs    = [d for _, d in ranked_pairs]
            max_score     = max(rerank_scores) if rerank_scores else 1.0

            # ── 2.5 외부 API Fallback 동적 참조 ──────────────────────────
            if USE_EXTERNAL_API and EXTERNAL_FALLBACK_ONLY:
                top_score_local = float(rerank_scores[0]) if rerank_scores else -999.0
                top_rel_local = 1.0 / (1.0 + math.exp(-max(min(top_score_local, 30.0), -30.0)))
                low_rel = top_rel_local < EXTERNAL_TRIGGER_MIN_RELEVANCE
                low_score = top_score_local < EXTERNAL_TRIGGER_MIN_TOP_SCORE
                if low_rel:
                    external_trigger_reason.append("low_relevance")
                if low_score:
                    external_trigger_reason.append("low_top_score")

                trigger_mode = EXTERNAL_TRIGGER_MODE if EXTERNAL_TRIGGER_MODE in {"or", "and"} else "or"
                should_call_external = (
                    (low_rel and low_score)
                    if trigger_mode == "and"
                    else (low_rel or low_score)
                )

                if should_call_external:
                    external_fallback_triggered = True
                    yield _sse("status", {"step": "외부 약학 API 보강 검색 중...", "icon": "🌐"})
                    external_docs = await loop.run_in_executor(
                        None,
                        lambda: fetch_external_pharma_docs(
                            query=req.query,
                            provider=EXTERNAL_API_PROVIDER,
                            top_k=EXTERNAL_TOP_K,
                            timeout_sec=EXTERNAL_TIMEOUT_SEC,
                        ),
                    )
                    external_docs_count = len(external_docs)

                    if external_docs_count > 0:
                        # 로컬 검색 결과 + 외부 API 문서를 합쳐 재리랭킹
                        merged_docs = ensemble_docs + external_docs
                        rerank_refine_start = time.time()
                        ranked_pairs = await loop.run_in_executor(
                            None,
                            lambda: rerank_docs(
                                query=req.query,
                                docs=merged_docs,
                                reranker=_resources["reranker"],
                                top_k=req.top_k,
                                batch_size=64,
                            ),
                        )
                        rerank_elapsed += (time.time() - rerank_refine_start)
                        rerank_scores = [s for s, _ in ranked_pairs]
                        final_docs = [d for _, d in ranked_pairs]
                        max_score = max(rerank_scores) if rerank_scores else 1.0
                        logger.info(
                            "[ExternalFallback] triggered=true, fetched=%d, top_score_local=%.4f, top_rel_local=%.4f",
                            external_docs_count,
                            top_score_local,
                            top_rel_local,
                        )
                else:
                    logger.info(
                        "[ExternalFallback] triggered=false, mode=%s, top_score_local=%.4f, top_rel_local=%.4f",
                        trigger_mode,
                        top_score_local,
                        top_rel_local,
                    )

            def _build_docs_payload() -> list[dict]:
                payload = []
                local_max_score = max(rerank_scores) if rerank_scores else 1.0
                for i, (score, doc) in enumerate(zip(rerank_scores, final_docs), 1):
                    pct = min(score / max(local_max_score, 1e-6), 1.0)
                    payload.append({
                        "rank": i,
                        "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                        "score": round(float(score), 4),
                        "pct": round(float(pct) * 100, 1),
                        "preview": doc.page_content.replace("passage: ", "").replace("\n", " ")[:280],
                    })
                return payload

            # ── 3. 참고 문서 정보 선제적 전송 (UI 로딩 체감 개선) ──────
            docs_payload = _build_docs_payload()
            # (UI가 'docs' 이벤트를 개별적으로 처리할 수 있다면 미리 보냄)
            yield _sse("docs", {"docs": docs_payload})

            # ── 3.5 Out-of-scope 가드 ───────────────────────────────
            top_score = float(rerank_scores[0]) if rerank_scores else -999.0
            # CrossEncoder raw score를 0~1 relevance로 매핑 (sigmoid)
            top_relevance = 1.0 / (1.0 + math.exp(-max(min(top_score, 30.0), -30.0)))

            # NOTE:
            # CrossEncoder raw score 범위가 작게 나오는 데이터셋에서는
            # sigmoid(top_score) 기반 relevance만으로 in-scope 질문이 과차단될 수 있음.
            # 따라서 "relevance + raw top_score"를 함께 만족할 때만 OOS 차단.
            should_oos_block = (
                OOS_GUARD_ENABLED
                and (top_relevance < OOS_MIN_RELEVANCE)
                and (top_score < OOS_MIN_TOP_SCORE)
            )

            # 최종 OOS 직전, fallback 모드라면 외부 API를 1회 강제 보강 시도
            if (
                should_oos_block
                and USE_EXTERNAL_API
                and EXTERNAL_FALLBACK_ONLY
                and external_docs_count == 0
            ):
                yield _sse("status", {"step": "외부 약학 API 재보강 검색 중...", "icon": "🌐"})
                external_fallback_triggered = True
                external_trigger_reason.append("pre_oos_force_fallback")
                external_docs = await loop.run_in_executor(
                    None,
                    lambda: fetch_external_pharma_docs(
                        query=req.query,
                        provider=EXTERNAL_API_PROVIDER,
                        top_k=EXTERNAL_TOP_K,
                        timeout_sec=EXTERNAL_TIMEOUT_SEC,
                    ),
                )
                external_docs_count = len(external_docs)

                if external_docs_count > 0:
                    merged_docs = ensemble_docs + external_docs
                    rerank_refine_start = time.time()
                    ranked_pairs = await loop.run_in_executor(
                        None,
                        lambda: rerank_docs(
                            query=req.query,
                            docs=merged_docs,
                            reranker=_resources["reranker"],
                            top_k=req.top_k,
                            batch_size=64,
                        ),
                    )
                    rerank_elapsed += (time.time() - rerank_refine_start)
                    rerank_scores = [s for s, _ in ranked_pairs]
                    final_docs = [d for _, d in ranked_pairs]
                    top_score = float(rerank_scores[0]) if rerank_scores else -999.0
                    top_relevance = 1.0 / (1.0 + math.exp(-max(min(top_score, 30.0), -30.0)))
                    should_oos_block = (
                        OOS_GUARD_ENABLED
                        and (top_relevance < OOS_MIN_RELEVANCE)
                        and (top_score < OOS_MIN_TOP_SCORE)
                    )
                    docs_payload = _build_docs_payload()
                    yield _sse("docs", {"docs": docs_payload})

            if should_oos_block:
                oos_answer = (
                    "제공된 문서에 해당 정보가 없습니다. "
                    "현재 보유한 근거 범위 밖 질문으로 판단되어 추측 답변을 생략합니다. "
                    "관련 의약품/증상 키워드를 더 구체적으로 알려주시면 다시 확인해드리겠습니다."
                )
                total_elapsed_oos = time.time() - total_start
                yield _sse("done", {
                    "answer": oos_answer,
                    "is_pass": True,
                    "correction_rounds": 0,
                    "correction_logs": [],
                    "verify_result": (
                        f"OOS_GUARD (top_relevance={top_relevance:.3f}, rel_threshold={OOS_MIN_RELEVANCE:.3f}, "
                        f"top_score={top_score:.4f}, score_threshold={OOS_MIN_TOP_SCORE:.4f})"
                    ),
                    "metrics_pending": False,
                    "ragas": {"faithfulness": 0.0, "answer_relevancy": 0.0},
                    "docs": docs_payload,
                    "metrics": {
                        "search_s": round(search_elapsed, 3),
                        "rerank_s": round(rerank_elapsed, 3),
                        "gen_s": 0.0,
                        "verify_s": 0.0,
                        "total_s": round(total_elapsed_oos, 3),
                        "ensemble_n": len(ensemble_docs),
                        "final_n": len(final_docs),
                        "top_score": round(top_score, 4),
                        "top_relevance": round(top_relevance, 4),
                        "oos_min_top_score": round(OOS_MIN_TOP_SCORE, 4),
                        "external_fallback_enabled": bool(USE_EXTERNAL_API and EXTERNAL_FALLBACK_ONLY),
                        "external_fallback_triggered": external_fallback_triggered,
                        "external_docs_count": external_docs_count,
                        "external_trigger_reason": external_trigger_reason,
                        "external_api_status": _external_diag(),
                        "oos_guard": True,
                    },
                })
                return

            # ── 4. 답변 생성 (스트리밍) ─────────────────────────
            context_text = build_context(final_docs)
            yield _sse("status", {"step": "답변 생성 중...", "icon": "✍️"})
            gen_start = time.time()
            
            initial_answer = ""
            # 비동기 스트리밍을 위해 async_mode=True 사용 및 async for 루프 적용
            async_stream = await generate_answer(
                query=req.query,
                context_text=context_text,
                openai_api_key=OPENAI_API_KEY,
                model=req.model,
                stream=True,
                async_mode=True,
            )
            
            async for chunk in async_stream:
                if chunk:
                    initial_answer += chunk
                    yield _sse("token", {"text": chunk})
            
            gen_elapsed = time.time() - gen_start

            # ── 5. 검증 및 RAGAS 평가 (병렬 처리로 지연 시간 단축) ───────────
            yield _sse("status", {"step": "품질 검증 및 지표 분석 중...", "icon": "⚡"})
            
            # 병렬 실행을 위한 래퍼 함수들
            async def _run_verify():
                return await loop.run_in_executor(
                    None,
                    lambda: verify_answer(
                        query=req.query, context_text=context_text,
                        answer=initial_answer, openai_api_key=OPENAI_API_KEY, model=VERIFY_MODEL
                    )
                )

            async def _run_ragas():
                if req.model == "debug":
                    return {"faithfulness": 0.0, "answer_relevancy": 0.0}
                return await loop.run_in_executor(
                    None,
                    lambda: evaluate_with_ragas(
                        query=req.query,
                        answer=initial_answer,
                        final_docs=final_docs,
                        embeddings=_resources["embeddings"],
                        openai_api_key=OPENAI_API_KEY,
                        eval_model=RAGAS_MODEL,
                    )
                )

            # 검증과 RAGAS를 동시에 시작
            verify_task = asyncio.create_task(_run_verify())
            ragas_task = asyncio.create_task(_run_ragas())

            # 검증 결과는 교정 루프 진입 여부를 결정하므로 먼저 기다림 (하지만 RAGAS는 계속 돌아감)
            verify_result = await verify_task
            verify_elapsed = time.time() - gen_start - gen_elapsed # 대략적인 시간

            final_answer = initial_answer
            correction_rounds = 0
            correction_logs: list[dict] = []
            
            # 검증 결과 즉시 전송 (UI 반영용)
            yield _sse("verdict", {
                "is_pass": "PASS" in verify_result.upper(),
                "verify_result": verify_result
            })

            if req.use_self_correction and "FAIL" in verify_result.upper():
                yield _sse("status", {"step": "자동 프롬프트 최적화 시작...", "icon": "🤖"})
                
                # 교정 루프 진입 전, 이전 RAGAS 태스크가 있다면 취소하거나 무시 (새로운 답변으로 평가할 것이므로)
                if ragas_task and not ragas_task.done():
                    ragas_task.cancel()

                # 비동기 제너레이터로 교정 루프 실행
                async for event_type, value in self_correction_loop(
                    query=req.query,
                    context_text=context_text,
                    initial_answer=initial_answer,
                    initial_verify_result=verify_result,
                    openai_api_key=OPENAI_API_KEY,
                    gen_model=req.model,
                    max_rounds=1, # 1회 권장
                    initial_ragas_result=None, # 지표 생략
                    embeddings=_resources["embeddings"],
                    final_docs=final_docs,
                ):
                    if event_type == "status":
                        yield _sse("status", value)
                    elif event_type == "token":
                        yield _sse("token", {"text": value})
                    elif event_type == "done_loop":
                        final_answer = value["answer"]
                        verify_result = value["verify_result"]
                        correction_rounds = value["rounds"]
                        correction_logs = value["logs"]
                
                # 교정 후 즉시 'done' 전송 (RAGAS는 백그라운드 시작)
                yield _sse("status", {"step": "교정 완료!", "icon": "✅"})
                total_elapsed = time.time() - total_start
                
                # 교정된 답변에 대해 RAGAS 다시 시작
                ragas_task = asyncio.create_task(_run_ragas_for_answer(final_answer))

                yield _sse("done", {
                    "answer": final_answer,
                    "is_pass": "PASS" in verify_result.upper(),
                    "correction_rounds": correction_rounds,
                    "correction_logs": correction_logs,
                    "verify_result": verify_result,
                    "metrics_pending": True,
                    "ragas": {"faithfulness": 0.0, "answer_relevancy": 0.0},
                    "docs": docs_payload,
                    "metrics": {
                        "search_s": round(search_elapsed, 3),
                        "rerank_s": round(rerank_elapsed, 3),
                        "gen_s": round(gen_elapsed, 3),
                        "verify_s": round(verify_elapsed, 3),
                        "total_s": round(total_elapsed, 3),
                        "ensemble_n": len(ensemble_docs),
                        "final_n": len(final_docs),
                        "external_fallback_enabled": bool(USE_EXTERNAL_API and EXTERNAL_FALLBACK_ONLY),
                        "external_fallback_triggered": external_fallback_triggered,
                        "external_docs_count": external_docs_count,
                        "external_trigger_reason": external_trigger_reason,
                        "external_api_status": _external_diag(),
                    },
                })
            else:
                # PASS인 경우 즉시 'done'을 보내어 사용자 대기 시간 최소화
                yield _sse("status", {"step": "검증 완료!", "icon": "✅"})
                total_elapsed_partial = time.time() - total_start
                yield _sse("done", {
                    "answer": final_answer,
                    "is_pass": True,
                    "correction_rounds": 0,
                    "correction_logs": [],
                    "verify_result": verify_result,
                    "metrics_pending": True,
                    "ragas": {"faithfulness": 0.0, "answer_relevancy": 0.0},
                    "docs": docs_payload,
                    "metrics": {
                        "search_s": round(search_elapsed, 3),
                        "rerank_s": round(rerank_elapsed, 3),
                        "gen_s": round(gen_elapsed, 3),
                        "verify_s": round(verify_elapsed, 3),
                        "total_s": round(total_elapsed_partial, 3),
                        "ensemble_n": len(ensemble_docs),
                        "final_n": len(final_docs),
                        "external_fallback_enabled": bool(USE_EXTERNAL_API and EXTERNAL_FALLBACK_ONLY),
                        "external_fallback_triggered": external_fallback_triggered,
                        "external_docs_count": external_docs_count,
                        "external_trigger_reason": external_trigger_reason,
                        "external_api_status": _external_diag(),
                    },
                })
            
            # ── 6. 공통 RAGAS 업데이트 (백그라운드 완료 대기) ───────────
            if ragas_task:
                try:
                    ragas_results = await ragas_task
                    yield _sse("metrics_update", ragas_results)
                except Exception as ragas_err:
                    logger.warning("RAGAS evaluation failed: %s", ragas_err)

        except asyncio.CancelledError:
            logger.info("Chat stream cancelled by client.")
            await _cancel_pending_tasks()
            return
        except Exception as e:
            logger.exception("Chat error: %s", e)
            total_elapsed_error = time.time() - total_start
            try:
                yield _sse("done", {
                    "answer": "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    "is_pass": False,
                    "correction_rounds": 0,
                    "correction_logs": [],
                    "verify_result": f"ERROR: {str(e)}",
                    "metrics_pending": False,
                    "ragas": {"faithfulness": 0.0, "answer_relevancy": 0.0},
                    "docs": docs_payload,
                    "metrics": {
                        "search_s": round(search_elapsed, 3),
                        "rerank_s": round(rerank_elapsed, 3),
                        "gen_s": round(gen_elapsed, 3),
                        "verify_s": round(verify_elapsed, 3),
                        "total_s": round(total_elapsed_error, 3),
                        "ensemble_n": len(ensemble_docs),
                        "final_n": len(final_docs),
                        "external_fallback_enabled": bool(USE_EXTERNAL_API and EXTERNAL_FALLBACK_ONLY),
                        "external_fallback_triggered": external_fallback_triggered,
                        "external_docs_count": external_docs_count,
                        "external_trigger_reason": external_trigger_reason,
                        "external_api_status": _external_diag(),
                        "error": True,
                    },
                })
            except Exception:
                pass
        finally:
            await _cancel_pending_tasks()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Static files (프론트엔드 서빙)
# ─────────────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web-ui")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ─────────────────────────────────────────────────────────────────────
# 직접 실행
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
