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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

from retriever import (
    load_embeddings,
    load_vector_db_with_embeddings,
    load_reranker,
    build_bm25_retriever,
    get_ensemble_results,
    rerank_docs,
)
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
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_all_resources)
    yield
    _resources.clear()


app = FastAPI(title="약사 AI 챗봇 API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "./chroma_db_combined_1771477980"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OOS_GUARD_ENABLED = os.getenv("OOS_GUARD_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
OOS_MIN_RELEVANCE = float(os.getenv("OOS_MIN_RELEVANCE", "0.55"))
OOS_MIN_TOP_SCORE = float(os.getenv("OOS_MIN_TOP_SCORE", "0.01"))
USE_QUERY_OPTIMIZER = os.getenv("USE_QUERY_OPTIMIZER", "true").lower() in {"1", "true", "yes", "on"}
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "gpt-5.2")
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "gpt-5.2")
RERANK_BATCH_SIZE = max(1, int(os.getenv("RERANK_BATCH_SIZE", "32")))

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
    _resources["embeddings"] = load_embeddings()
    _log_init("임베딩 모델 로드 완료")
    _resources["vector_db"] = load_vector_db_with_embeddings(DB_PATH, _resources["embeddings"])
    _log_init("벡터 데이터베이스 연결 완료")
    _resources["reranker"] = load_reranker()
    _log_init("리랭커 모델 로드 완료")
    _resources["bm25"], _resources["bm25_docs"], _resources["kiwi"] = build_bm25_retriever(vector_db=_resources["vector_db"])
    _log_init("BM25 인덱스 생성 완료")
    _resources["query_optimizer"] = get_query_optimizer(OPENAI_API_KEY) if USE_QUERY_OPTIMIZER else None
    _log_init("쿼리 최적화기 준비 완료" if USE_QUERY_OPTIMIZER else "쿼리 최적화기 비활성화(속도 우선)")
    _init_done = True
    _log_init("모든 리소스 준비 완료!")


class ChatRequest(BaseModel):
    query: str
    model: str = "gpt-5.1"
    top_k: int = 5
    ensemble_k: int = 20
    weight_bm25: float = 0.8
    use_self_correction: bool = True


@app.get("/health")
async def health():
    gpu_info = {"available": False}
    if torch.cuda.is_available():
        gpu_info = get_gpu_status()
    return {
        "status": "ready" if _init_done else "initializing",
        "init_logs": _init_logs[-5:],
        "gpu": gpu_info,
    }


@app.post("/clear-memory")
async def clear_memory():
    logger.info("GPU 메모리 정리 요청 수신")
    clear_gpu()
    return await health()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    if not _init_done:
        raise HTTPException(503, "리소스 초기화 중입니다. 잠시 후 다시 시도하세요.")
    if not OPENAI_API_KEY:
        raise HTTPException(400, ".env에 OPENAI_API_KEY가 없습니다.")

    async def generate() -> AsyncGenerator[str, None]:
        total_start = time.time()
        done_sent = False
        verify_task = None
        ragas_task = None
        search_elapsed = 0.0
        rerank_elapsed = 0.0
        gen_elapsed = 0.0
        verify_elapsed = 0.0
        search_breakdown: dict[str, float] = {}
        rerank_breakdown: dict[str, float] = {}
        ensemble_docs = []
        final_docs = []
        docs_payload: list[dict] = []

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

        import re
        def _is_pass(vr: str) -> bool:
            """검증 결과에서 [최종 판정] 파싱. 분석 코멘트의 PASS/FAIL 혼재를 무시."""
            # [최종 판정]: PASS 또는 FAIL 패턴 우선 탐색
            m = re.search(r'\[최종\s*판정\]\s*[:：]\s*(PASS|FAIL)', vr, re.IGNORECASE)
            if m:
                result = m.group(1).upper() == 'PASS'
                logger.info("[Verdict] Pattern match: '%s' -> %s", m.group(0), result)
                return result
            # 패턴 못 찾으면 마지막 PASS/FAIL 단어 기준
            tokens = re.findall(r'\b(PASS|FAIL)\b', vr, re.IGNORECASE)
            if tokens:
                result = tokens[-1].upper() == 'PASS'
                logger.info("[Verdict] Fallback last token: '%s' -> %s", tokens[-1], result)
                return result
            logger.warning("[Verdict] No PASS/FAIL found in verify_result")
            return False

        try:
            yield _sse("status", {"step": "검색 중...", "icon": "🔍"})
            search_start = time.time()
            ensemble_docs, search_breakdown = await loop.run_in_executor(
                None,
                lambda: get_ensemble_results(
                    query=req.query,
                    kiwi=_resources["kiwi"],
                    bm25_retriever=_resources["bm25"],
                    vector_db=_resources["vector_db"],
                    bm25_docs=_resources["bm25_docs"],
                    query_optimizer=_resources.get("query_optimizer"),
                    k=req.ensemble_k,
                    weight_bm25=req.weight_bm25,
                    weight_vector=round(1.0 - req.weight_bm25, 2),
                    return_metrics=True,
                ),
            )
            search_elapsed = time.time() - search_start

            # 리랭킹 쿼리: 옵티마이저가 있으면 의료 핵심어로 재구성
            rerank_query = req.query
            if USE_QUERY_OPTIMIZER and _resources.get("query_optimizer"):
                try:
                    from langchain_core.output_parsers import StrOutputParser
                    opt = _resources["query_optimizer"]
                    clean_prompt = (
                        "사용자 질문에서 약학/의학 관련 핵심 키워드만 추출하여 검색 쿼리로 변환하세요.\n"
                        "비의학적 표현(뭐먹을까, 어떡해, 괜찮을까 등)은 제거하세요.\n"
                        "출력: 핵심 검색 쿼리만 (한 줄)\n"
                        f"입력: {req.query}"
                    )
                    rerank_query = opt.invoke(clean_prompt).strip().strip('"').strip("'")
                    logger.info("리랭킹 쿼리 최적화: '%s' -> '%s'", req.query, rerank_query)
                except Exception as e:
                    logger.warning("리랭킹 쿼리 최적화 실패: %s", e)

            yield _sse("status", {"step": f"{len(ensemble_docs)}개 문서 리랭킹 중...", "icon": "⚡"})
            rerank_start = time.time()
            ranked_pairs, rerank_breakdown = await loop.run_in_executor(
                None,
                lambda: rerank_docs(
                    query=rerank_query,
                    docs=ensemble_docs,
                    reranker=_resources["reranker"],
                    top_k=req.top_k,
                    batch_size=RERANK_BATCH_SIZE,
                    return_metrics=True,
                ),
            )
            rerank_elapsed = time.time() - rerank_start

            rerank_scores = [s for s, _ in ranked_pairs]
            final_docs = [d for _, d in ranked_pairs]

            # 최소 점수 임계값 적용: 관련성 낮은 문서 제거
            MIN_RERANK_SCORE = 0.005
            filtered = [(s, d) for s, d in zip(rerank_scores, final_docs) if s >= MIN_RERANK_SCORE]
            if filtered:
                rerank_scores, final_docs = zip(*filtered)
                rerank_scores = list(rerank_scores)
                final_docs = list(final_docs)
            else:
                # 모든 문서가 임계값 이하면 상위 1개라도 유지
                rerank_scores = rerank_scores[:1]
                final_docs = final_docs[:1]
            logger.info("문서 필터링: %d/%d개 (임계값 %.3f)", len(final_docs), len(ranked_pairs), MIN_RERANK_SCORE)

            def _build_docs_payload() -> list[dict]:
                payload = []
                local_max_score = max(rerank_scores) if rerank_scores else 1.0
                for i, (score, doc) in enumerate(zip(rerank_scores, final_docs), 1):
                    if score < MIN_RERANK_SCORE:
                        continue
                    pct = min(score / max(local_max_score, 1e-6), 1.0)
                    payload.append({
                        "rank": i,
                        "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                        "score": round(float(score), 4),
                        "pct": round(float(pct) * 100, 1),
                        "preview": doc.page_content.replace("passage: ", "").replace("\n", " ")[:280],
                    })
                return payload

            docs_payload = _build_docs_payload()
            yield _sse("docs", {"docs": docs_payload})

            top_score = float(rerank_scores[0]) if rerank_scores else -999.0
            top_relevance = 1.0 / (1.0 + math.exp(-max(min(top_score, 30.0), -30.0)))
            should_oos_block = (
                OOS_GUARD_ENABLED
                and (top_relevance < OOS_MIN_RELEVANCE)
                and (top_score < OOS_MIN_TOP_SCORE)
            )

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
                        "oos_guard": True,
                        **{k: round(float(v), 3) for k, v in search_breakdown.items()},
                        **{k: round(float(v), 3) for k, v in rerank_breakdown.items()},
                    },
                })
                return

            context_text = build_context(final_docs)
            yield _sse("status", {"step": "답변 생성 중...", "icon": "✍️"})
            gen_start = time.time()

            initial_answer = ""
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
            yield _sse("status", {"step": "품질 검증 및 지표 분석 중...", "icon": "⚡"})

            async def _run_verify():
                return await loop.run_in_executor(
                    None,
                    lambda: verify_answer(
                        query=req.query,
                        context_text=context_text,
                        answer=initial_answer,
                        openai_api_key=OPENAI_API_KEY,
                        model=VERIFY_MODEL,
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

            verify_task = asyncio.create_task(_run_verify())
            ragas_task = asyncio.create_task(_run_ragas())

            verify_result = await verify_task
            verify_elapsed = time.time() - gen_start - gen_elapsed

            final_answer = initial_answer
            correction_rounds = 0
            correction_logs: list[dict] = []

            is_pass = _is_pass(verify_result)
            yield _sse("verdict", {
                "is_pass": is_pass,
                "verify_result": verify_result
            })

            if req.use_self_correction and not is_pass:
                yield _sse("status", {"step": "자동 프롬프트 최적화 시작...", "icon": "🤖"})
                if ragas_task and not ragas_task.done():
                    ragas_task.cancel()

                async for event_type, value in self_correction_loop(
                    query=req.query,
                    context_text=context_text,
                    initial_answer=initial_answer,
                    initial_verify_result=verify_result,
                    openai_api_key=OPENAI_API_KEY,
                    gen_model=req.model,
                    max_rounds=2,
                    initial_ragas_result=None,
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

                yield _sse("status", {"step": "교정 완료!", "icon": "✅"})

            # ── done 이벤트 전송 ──
            total_elapsed_done = time.time() - total_start
            done_sent = True
            yield _sse("done", {
                "answer": final_answer,
                "is_pass": _is_pass(verify_result),
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
                    "total_s": round(total_elapsed_done, 3),
                    "ensemble_n": len(ensemble_docs),
                    "final_n": len(final_docs),
                    **{k: round(float(v), 3) for k, v in search_breakdown.items()},
                    **{k: round(float(v), 3) for k, v in rerank_breakdown.items()},
                },
            })

            # ── RAGAS 결과 수신 ──
            if ragas_task:
                try:
                    ragas_results = await ragas_task
                    yield _sse("metrics_update", ragas_results)
                except BaseException as ragas_err:
                    logger.warning("RAGAS evaluation failed or cancelled: %s", ragas_err)
                    yield _sse("metrics_update", {"faithfulness": 0.0, "answer_relevancy": 0.0})

        except asyncio.CancelledError:
            logger.info("Chat stream cancelled by client.")
            await _cancel_pending_tasks()
            return
        except Exception as e:
            logger.exception("Chat error: %s", e)
            total_elapsed_error = time.time() - total_start
            try:
                if done_sent:
                    yield _sse("error", {"message": f"후속 처리 중 오류: {str(e)}"})
                else:
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
                            "error": True,
                            **{k: round(float(v), 3) for k, v in search_breakdown.items()},
                            **{k: round(float(v), 3) for k, v in rerank_breakdown.items()},
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


STATIC_DIR = os.path.join(os.path.dirname(__file__), "web-ui")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
