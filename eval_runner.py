"""
eval_runner.py
프로젝트 발표용 평가/개선 로그를 자동 생성하는 배치 실행 스크립트.

생성 산출물:
- outputs/eval_runs/<timestamp>/round_logs.jsonl
- outputs/eval_runs/<timestamp>/case_summary.csv
- outputs/eval_runs/<timestamp>/summary.md

실행 예시:
  python eval_runner.py --cases eval_cases.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from retriever import (
    build_bm25_retriever,
    get_ensemble_results,
    load_embeddings,
    load_reranker,
    load_vector_db_with_embeddings,
    rerank_docs,
)
from generator import (
    build_context,
    evaluate_with_ragas,
    generate_answer,
    get_query_optimizer,
    self_correction_loop,
    verify_answer,
)
from processor import clean_json_to_text


@dataclass
class RoundInternal:
    final_docs: list
    context_text: str
    answer: str
    verify_result: str
    ragas: dict[str, float]
    docs_payload: list[dict[str, Any]]
    top_score: float
    top_relevance: float
    is_oos_guarded: bool


def _sigmoid(x: float) -> float:
    x = max(min(x, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-x))


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe(v: Any, d: float = 0.0) -> float:
    try:
        n = float(v)
        if math.isnan(n) or math.isinf(n):
            return d
        return n
    except Exception:
        return d


def _doc_preview(doc) -> str:
    return clean_json_to_text(doc.page_content).replace("\n", " ")[:240]


def _build_docs_payload(ranked_pairs: list[tuple[float, Any]]) -> list[dict[str, Any]]:
    scores = [float(s) for s, _ in ranked_pairs]
    max_score = max(scores) if scores else 1.0
    payload = []
    for idx, (score, doc) in enumerate(ranked_pairs, 1):
        pct = min(float(score) / max(max_score, 1e-6), 1.0)
        payload.append(
            {
                "rank": idx,
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "score": round(float(score), 4),
                "pct": round(pct * 100, 2),
                "preview": _doc_preview(doc),
            }
        )
    return payload


async def _run_correction_once(
    *,
    query: str,
    context_text: str,
    initial_answer: str,
    initial_verify_result: str,
    openai_api_key: str,
    gen_model: str,
    initial_ragas_result: dict[str, float],
    embeddings,
    final_docs,
) -> dict[str, Any] | None:
    done_payload = None
    async for event_type, value in self_correction_loop(
        query=query,
        context_text=context_text,
        initial_answer=initial_answer,
        initial_verify_result=initial_verify_result,
        openai_api_key=openai_api_key,
        gen_model=gen_model,
        max_rounds=1,
        initial_ragas_result=initial_ragas_result,
        embeddings=embeddings,
        final_docs=final_docs,
    ):
        if event_type == "done_loop":
            done_payload = value
    return done_payload


def _run_round_1(
    *,
    case: dict[str, Any],
    resources: dict[str, Any],
    openai_api_key: str,
    model: str,
    eval_model: str,
    top_k: int,
    ensemble_k: int,
    weight_bm25: float,
    rerank_batch_size: int,
    run_ragas: bool,
    oos_guard_enabled: bool,
    oos_min_relevance: float,
    oos_min_top_score: float,
) -> tuple[dict[str, Any], RoundInternal]:
    query = case["query"]

    t0 = time.time()
    ensemble_docs, search_breakdown = get_ensemble_results(
        query=query,
        kiwi=resources["kiwi"],
        bm25_retriever=resources["bm25"],
        vector_db=resources["vector_db"],
        query_optimizer=resources["query_optimizer"],
        k=ensemble_k,
        weight_bm25=weight_bm25,
        weight_vector=round(1.0 - weight_bm25, 2),
        return_metrics=True,
    )
    search_s = time.time() - t0

    t1 = time.time()
    ranked_pairs, rerank_breakdown = rerank_docs(
        query=query,
        docs=ensemble_docs,
        reranker=resources["reranker"],
        top_k=top_k,
        batch_size=rerank_batch_size,
        return_metrics=True,
    )
    rerank_s = time.time() - t1

    rerank_scores = [float(s) for s, _ in ranked_pairs]
    final_docs = [d for _, d in ranked_pairs]
    top_score = rerank_scores[0] if rerank_scores else -999.0
    top_relevance = _sigmoid(top_score)
    docs_payload = _build_docs_payload(ranked_pairs)

    retrieval_results = [
        {
            "rank": i + 1,
            "source": os.path.basename(doc.metadata.get("source", "Unknown")),
            "preview": _doc_preview(doc),
        }
        for i, doc in enumerate(ensemble_docs[: min(len(ensemble_docs), 20)])
    ]

    rerank_results = docs_payload

    is_oos_guarded = bool(
        oos_guard_enabled
        and top_relevance < oos_min_relevance
        and top_score < oos_min_top_score
    )
    if is_oos_guarded:
        answer = (
            "제공된 문서에 해당 정보가 없습니다. "
            "현재 보유한 근거 범위 밖 질문으로 판단되어 추측 답변을 생략합니다."
        )
        verify_result = (
            f"OOS_GUARD (top_relevance={top_relevance:.3f}, rel_threshold={oos_min_relevance:.3f}, "
            f"top_score={top_score:.4f}, score_threshold={oos_min_top_score:.4f})"
        )
        ragas = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        gen_s = 0.0
        verify_s = 0.0
        context_text = build_context(final_docs)
    else:
        context_text = build_context(final_docs)

        t2 = time.time()
        answer = asyncio.run(
            generate_answer(
                query=query,
                context_text=context_text,
                openai_api_key=openai_api_key,
                model=model,
                stream=False,
            )
        )
        gen_s = time.time() - t2

        t3 = time.time()
        verify_result = verify_answer(
            query=query,
            context_text=context_text,
            answer=answer,
            openai_api_key=openai_api_key,
            model=eval_model,
        )
        verify_s = time.time() - t3

        if run_ragas:
            ragas = evaluate_with_ragas(
                query=query,
                answer=answer,
                final_docs=final_docs,
                embeddings=resources["embeddings"],
                openai_api_key=openai_api_key,
                eval_model=eval_model,
            )
        else:
            ragas = {"faithfulness": 0.0, "answer_relevancy": 0.0}

    total_s = search_s + rerank_s + gen_s + verify_s
    is_pass = "PASS" in verify_result.upper() or is_oos_guarded

    round_log = {
        "case_id": case["id"],
        "round": 1,
        "query": query,
        "expected_scope": case.get("expected_scope", "in_scope"),
        "retrieval_results": retrieval_results,
        "rerank_results": rerank_results,
        "answer": answer,
        "verify_result": verify_result,
        "is_pass": is_pass,
        "ragas": ragas,
        "metrics": {
            "search_s": round(search_s, 3),
            "rerank_s": round(rerank_s, 3),
            "gen_s": round(gen_s, 3),
            "verify_s": round(verify_s, 3),
            "total_s": round(total_s, 3),
            "ensemble_n": len(ensemble_docs),
            "final_n": len(final_docs),
            "top_score": round(top_score, 4),
            "top_relevance": round(top_relevance, 4),
            "oos_min_top_score": round(oos_min_top_score, 4),
            "oos_guard": is_oos_guarded,
            **{k: round(float(v), 3) for k, v in search_breakdown.items()},
            **{k: round(float(v), 3) for k, v in rerank_breakdown.items()},
        },
    }

    internal = RoundInternal(
        final_docs=final_docs,
        context_text=context_text,
        answer=answer,
        verify_result=verify_result,
        ragas=ragas,
        docs_payload=docs_payload,
        top_score=top_score,
        top_relevance=top_relevance,
        is_oos_guarded=is_oos_guarded,
    )
    return round_log, internal


def _run_round_2_if_needed(
    *,
    case: dict[str, Any],
    r1_log: dict[str, Any],
    r1_internal: RoundInternal,
    resources: dict[str, Any],
    openai_api_key: str,
    model: str,
    enable_improve: bool,
) -> dict[str, Any] | None:
    if not enable_improve:
        return None
    if r1_internal.is_oos_guarded:
        return None
    if "FAIL" not in r1_internal.verify_result.upper():
        return None

    done_payload = asyncio.run(
        _run_correction_once(
            query=case["query"],
            context_text=r1_internal.context_text,
            initial_answer=r1_internal.answer,
            initial_verify_result=r1_internal.verify_result,
            openai_api_key=openai_api_key,
            gen_model=model,
            initial_ragas_result=r1_internal.ragas,
            embeddings=resources["embeddings"],
            final_docs=r1_internal.final_docs,
        )
    )

    if not done_payload:
        return None

    answer2 = done_payload.get("answer", "")
    verify2 = done_payload.get("verify_result", "")
    ragas2 = done_payload.get("ragas", {"faithfulness": 0.0, "answer_relevancy": 0.0})
    is_pass2 = "PASS" in verify2.upper()

    return {
        "case_id": case["id"],
        "round": 2,
        "query": case["query"],
        "expected_scope": case.get("expected_scope", "in_scope"),
        "retrieval_results": r1_log["retrieval_results"],
        "rerank_results": r1_log["rerank_results"],
        "answer": answer2,
        "verify_result": verify2,
        "is_pass": is_pass2,
        "ragas": ragas2,
        "metrics": {
            **r1_log["metrics"],
            "correction_rounds": done_payload.get("rounds", 1),
        },
        "correction_logs": done_payload.get("logs", []),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "expected_scope",
        "round1_pass",
        "round2_pass",
        "improved",
        "round1_faithfulness",
        "round1_relevancy",
        "round2_faithfulness",
        "round2_relevancy",
        "top_relevance",
        "oos_guard",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_summary_md(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    total = len(summary_rows)
    r1_pass = sum(1 for r in summary_rows if r["round1_pass"])
    r2_pass = sum(1 for r in summary_rows if r["round2_pass"])
    improved = sum(1 for r in summary_rows if r["improved"])

    lines = [
        "# Eval Summary",
        "",
        f"- total_cases: **{total}**",
        f"- round1_pass: **{r1_pass}**",
        f"- round2_pass: **{r2_pass}**",
        f"- improved_cases: **{improved}**",
        "",
        "## Case Table",
        "",
        "| case_id | scope | R1 | R2 | improved | faith(R1→R2) | relev(R1→R2) | oos_guard |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summary_rows:
        lines.append(
            "| {case_id} | {expected_scope} | {r1} | {r2} | {imp} | {f1:.3f}→{f2:.3f} | {a1:.3f}→{a2:.3f} | {oos} |".format(
                case_id=r["case_id"],
                expected_scope=r["expected_scope"],
                r1="PASS" if r["round1_pass"] else "FAIL",
                r2="PASS" if r["round2_pass"] else "-",
                imp="Y" if r["improved"] else "N",
                f1=_safe(r["round1_faithfulness"]),
                f2=_safe(r["round2_faithfulness"]),
                a1=_safe(r["round1_relevancy"]),
                a2=_safe(r["round2_relevancy"]),
                oos="Y" if r["oos_guard"] else "N",
            )
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 프로젝트 평가/개선 배치 실행")
    parser.add_argument("--cases", default="eval_cases.json", help="평가 케이스 JSON 경로")
    parser.add_argument("--output-dir", default="outputs/eval_runs", help="출력 상위 디렉터리")
    parser.add_argument("--db-path", default="./chroma_db_combined_1771477980")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--eval-model", default="gpt-5.2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ensemble-k", type=int, default=20)
    parser.add_argument("--weight-bm25", type=float, default=0.8)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=0, help="0이면 전체")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-improve", action="store_true")
    parser.add_argument("--oos-guard", action="store_true")
    parser.add_argument("--oos-min-relevance", type=float, default=0.55)
    parser.add_argument("--oos-min-top-score", type=float, default=0.002)
    args = parser.parse_args()

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"케이스 파일이 없습니다: {cases_path}")

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    run_dir = Path(args.output_dir) / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[INIT] 리소스 로딩 중...")
    embeddings = load_embeddings()
    vector_db = load_vector_db_with_embeddings(args.db_path, embeddings)
    reranker = load_reranker()
    bm25, kiwi = build_bm25_retriever(vector_db=vector_db)
    query_optimizer = get_query_optimizer(openai_api_key)
    resources = {
        "embeddings": embeddings,
        "vector_db": vector_db,
        "reranker": reranker,
        "bm25": bm25,
        "kiwi": kiwi,
        "query_optimizer": query_optimizer,
    }

    logs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        print(f"[CASE {idx}/{len(cases)}] {case['id']} - {case['query']}")
        round1, internal = _run_round_1(
            case=case,
            resources=resources,
            openai_api_key=openai_api_key,
            model=args.model,
            eval_model=args.eval_model,
            top_k=args.top_k,
            ensemble_k=args.ensemble_k,
            weight_bm25=args.weight_bm25,
            rerank_batch_size=args.rerank_batch_size,
            run_ragas=not args.skip_ragas,
            oos_guard_enabled=args.oos_guard,
            oos_min_relevance=args.oos_min_relevance,
            oos_min_top_score=args.oos_min_top_score,
        )
        logs.append(round1)

        round2 = _run_round_2_if_needed(
            case=case,
            r1_log=round1,
            r1_internal=internal,
            resources=resources,
            openai_api_key=openai_api_key,
            model=args.model,
            enable_improve=not args.skip_improve,
        )
        if round2 is not None:
            logs.append(round2)

        r1_pass = bool(round1.get("is_pass", False))
        r2_pass = bool(round2.get("is_pass", False)) if round2 else False
        improved = bool((not r1_pass) and r2_pass)

        r1_ragas = round1.get("ragas", {})
        r2_ragas = (round2 or {}).get("ragas", {})
        summary_rows.append(
            {
                "case_id": case["id"],
                "expected_scope": case.get("expected_scope", "in_scope"),
                "round1_pass": r1_pass,
                "round2_pass": r2_pass,
                "improved": improved,
                "round1_faithfulness": _safe(r1_ragas.get("faithfulness", 0.0)),
                "round1_relevancy": _safe(r1_ragas.get("answer_relevancy", 0.0)),
                "round2_faithfulness": _safe(r2_ragas.get("faithfulness", 0.0)),
                "round2_relevancy": _safe(r2_ragas.get("answer_relevancy", 0.0)),
                "top_relevance": _safe(round1.get("metrics", {}).get("top_relevance", 0.0)),
                "oos_guard": bool(round1.get("metrics", {}).get("oos_guard", False)),
            }
        )

    _write_jsonl(run_dir / "round_logs.jsonl", logs)
    _write_summary_csv(run_dir / "case_summary.csv", summary_rows)
    _write_summary_md(run_dir / "summary.md", summary_rows)

    print("\n[DONE] 산출물 생성 완료")
    print(f"- {run_dir / 'round_logs.jsonl'}")
    print(f"- {run_dir / 'case_summary.csv'}")
    print(f"- {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
