"""
generator.py
GPT 답변 생성, 자기 검증(Verifier), RAGAS 평가 함수.
"""

import os
import json
import time
import asyncio
import logging
from datasets import Dataset

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from processor import clean_json_to_text, get_clean_doc_text


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 프롬프트 템플릿 (전역 상수)
# ──────────────────────────────────────────────────────────────────────

_ANSWER_PROMPT_TEMPLATE = """\
당신은 약법 및 전문 지침을 준수하며 오직 공식 데이터에만 근거하여 답변하는 **전문 약사 AI**입니다.
말투는 환자 입장에서 이해하기 쉬운 **친절하고 공감적인 한국어**를 사용하되, 사실/근거는 엄격히 지키십시오.

━━━ 🚨 엄격 준수 규칙 (절대적) ━━━
1. **데이터 중심 답변**: 반드시 아래 [검색된 문서]에 명시된 내용만 사용하여 답변하십시오. 문서에 없는 내용은 "제공된 문서에 해당 정보가 없습니다."를 포함해 명확히 알리고, 이어서 **약효/성분 추측 없이** 일반적 관리 팁 또는 추가 질문 유도 문장을 1~3줄 덧붙일 수 있습니다.
2. **환자 안전 최우선**: 문서에 부작용이나 주의사항이 있다면 반드시 포함하십시오. 일반적인 상식(예: "미지근한 물과 복용")은 조언으로 덧붙일 수 있으나, 약효나 성분에 대한 추측은 절대 금지입니다.
3. **출처 표기 (필수)**: 정보를 가져온 문장 끝에 반드시 **[문서 N]** 표기를 붙이십시오 (예: ...입니다. [문서 1]).
4. **허구 인용 금지**: 문서에 없는 내용을 적으면서 허위로 [문서 N] 표기를 붙이는 행위는 허용되지 않습니다.
━━━━━━━━━━━━━━━━━━━━━━━

[검색된 문서]
{context}

질문: {question}
답변:"""

_VERIFY_PROMPT_TEMPLATE = """\
당신은 약학 데이터의 무결성을 검증하는 **관대한 감사관**입니다.
[검증 대상 답변]이 [검색된 문서]의 핵심 내용을 왜곡 없이 반영했는지 평가하십시오.

[검색된 문서 (Ground Truth)]
{context}

[질문]
{question}

[검증 대상 답변]
{answer}

━━━ 🚨 FAIL 판정 기준 (치명적 오류만 FAIL) ━━━
F1. **심각한 성분/용량 오류**: 문서에 없는 약물 성분을 언급하거나, 권장 용량을 임의로 변경한 경우.
F2. **완전한 허구 인용**: 문서에 전혀 없는 정보를 언급하며 **[문서 N]** 표기를 붙인 경우.
F3. **검색 문서와 정반대되는 정보**: 문서 내용상 불가한 것을 가능하다고 하는 등 사실 관계 왜곡.

━━━ ✅ PASS 허용 기준 (이런 경우엔 PASS) ━━━
P1. 환자 안전을 위한 기본 권고(충분한 물, 전문가 상담 등)가 포함된 경우.
P2. 문맥을 위해 문서의 표현을 소폭 다듬은 경우.
P3. 핵심 정보의 출처가 명확히 기재된 경우.

[출력 형식]
- [분석 코멘트]: (무엇이 틀렸고 어떻게 고쳐야 하는지 구체적으로 기술)
- [최종 판정]: PASS 또는 FAIL"""

_CORRECTION_PROMPT_TEMPLATE = """\
당신은 검증 피드백을 바탕으로 답변을 수정하는 **전문 약사 AI**입니다.

[사용자 질문]: {question}
[검색된 문서]: {context}
[이전 답변]: {answer}
[검증 피드백]: {verify_result}

위 피드백을 반영하여 오류를 바로잡고, 다시 최선의 답변을 작성하십시오.
반드시 [검색된 문서]의 내용에만 집중하고, 수정된 답변만 출력하십시오."""

_OPTIMIZER_PROMPT_TEMPLATE = """\
당신은 'RAG 시스템 프롬프트 엔지니어링 전문가'입니다.
이전 라운드에서 생성된 답변이 검증 실패 판정을 받았습니다.

[사용자 질문]: {question}
[검증 결과]: {verify_result}
[RAGAS 지표]: {ragas_result}
[기존 프롬프트 템플릿]: {original_template}

위의 실패 원인과 지표를 분석하여, 다음 라운드에서 더 정확한 답변을 생성할 수 있도록 수정된 프롬프트 템플릿을 만드십시오.
- [검색된 문서]의 데이터를 더 정확하게 인용하고 추측을 배제하도록 지시를 강화하세요.
- 필요하다면 출력 형식이나 주의사항을 구체적으로 조정하세요.
- 반드시 {context}와 {question} 변수를 포함한 전체 프롬프트 전문만 출력하세요."""


# ──────────────────────────────────────────────────────────────────────
# 컨텍스트 구성
# ──────────────────────────────────────────────────────────────────────

def build_context(final_docs: list[Document], max_chars: int = 1500) -> str:
    """
    최종 선택 문서들을 컨텍스트 문자열로 변환합니다.

    Args:
        final_docs: 리랭킹된 Document 리스트
        max_chars:  문서 당 최대 문자 수 (기본 1500 – 정확도 향상)

    Returns:
        포맷된 컨텍스트 문자열
    """
    parts = []
    for i, doc in enumerate(final_docs, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        content = get_clean_doc_text(doc)
        parts.append(f"[문서 {i}] (출처: {source})\n{content[:max_chars]}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# 답변 생성
# ──────────────────────────────────────────────────────────────────────

def _get_llm(
    model: str,
    temperature: float,
    api_key: str,
    streaming: bool = False,
) -> ChatOpenAI:
    """
    LLM 인스턴스를 요청 단위로 생성합니다.

    전역 캐시 재사용 시 간헐적으로 "Event loop is closed"가 전파되는 사례가 있어,
    루프/요청 경계 간 객체 공유를 피하도록 안전 모드로 운용합니다.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        api_key=api_key,
    )



def _is_rate_limit_error(exc: Exception) -> bool:
    normalized = str(exc).lower()
    return any(keyword in normalized for keyword in ("429", "rate limit", "quota"))


def _is_event_loop_closed_error(exc: Exception) -> bool:
    return "event loop is closed" in str(exc).lower()


def _retry_api_call(callable_obj, payload):
    max_attempts = 5
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            return callable_obj(payload)
        except Exception as exc:
            if _is_event_loop_closed_error(exc) and attempt < max_attempts:
                logger.warning(
                    "Detected closed event loop. Rebuilding request chain and retrying... (%d/%d)",
                    attempt,
                    max_attempts,
                )
                time.sleep(0.2)
                continue
            if not _is_rate_limit_error(exc) or attempt == max_attempts:
                logger.error(f"API Call failed (Attempt {attempt}/{max_attempts}): {exc}")
                raise
            logger.warning(f"Rate limit hit. Retrying in {backoff}s... (Attempt {attempt}/{max_attempts})")
            time.sleep(backoff)
            backoff *= 2


async def _async_retry_api_call(callable_obj, payload):
    max_attempts = 5
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(callable_obj):
                return await callable_obj(payload)
            else:
                # callable_obj might be a method like chain.ainvoke
                res = callable_obj(payload)
                if hasattr(res, "__await__"):
                    return await res
                return res
        except Exception as exc:
            if _is_event_loop_closed_error(exc) and attempt < max_attempts:
                logger.warning(
                    "Detected closed event loop in async call. Retrying... (%d/%d)",
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(0.2)
                continue
            if not _is_rate_limit_error(exc) or attempt == max_attempts:
                logger.error(f"Async API Call failed (Attempt {attempt}/{max_attempts}): {exc}")
                raise
            logger.warning(f"Async Rate limit hit. Retrying in {backoff}s... (Attempt {attempt}/{max_attempts})")
            await asyncio.sleep(backoff)
            backoff *= 2


async def generate_answer(
    query: str,
    context_text: str,
    openai_api_key: str,
    model: str = "gpt-5",
    temperature: float = 0.1,
    prompt_template_str: str = _ANSWER_PROMPT_TEMPLATE,
    stream: bool = False,
    async_mode: bool = False,
):
    """
    GPT 모델로 최종 약사 답변을 생성합니다.

    Args:
        stream: True이면 제너레이터(청크 이터레이터)를 반환합니다.
        async_mode: True이고 stream=True이면 비동기 제너레이터(astream)를 반환합니다.

    Returns:
        stream=False → 완성된 답변 문자열
        stream=True, async_mode=False → 문자열 청크 이터레이터
        stream=True, async_mode=True → 비동기 문자열 청크 이터레이터
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key

    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = _get_llm(
        model=model,
        temperature=temperature,
        api_key=openai_api_key,
        streaming=bool(stream or async_mode),
    )
    chain = prompt | llm | StrOutputParser()

    call = (
        chain.astream if stream and async_mode else
        chain.stream if stream else
        chain.invoke
    )
    
    if async_mode:
        if stream:
            # Note: For stream=True, this only retries the initial connection/iterator creation.
            return await _async_retry_api_call(chain.astream, {"context": context_text, "question": query})
        else:
            return await _async_retry_api_call(chain.ainvoke, {"context": context_text, "question": query})
    
    return _retry_api_call(call, {"context": context_text, "question": query})


def get_query_optimizer(openai_api_key: str, model: str = "gpt-5.2"):
    """
    쿼리 확장(Query Expansion)용 경량 GPT 체인을 반환합니다.
    안정성을 위해 호출 시 새로운 인스턴스를 생성합니다.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return ChatOpenAI(model=model, temperature=0, api_key=openai_api_key) | StrOutputParser()


# ──────────────────────────────────────────────────────────────────────
# 자기 검증 (Verifier)
# ──────────────────────────────────────────────────────────────────────

def verify_answer(
    query: str,
    context_text: str,
    answer: str,
    openai_api_key: str,
    model: str = "gpt-5.2",
) -> str:
    """
    GPT 검증관이 답변의 논리적 타당성을 평가합니다. (사용자 요청: gpt-5.2 사용)

    Returns:
        검증 결과 문자열 (PASS / FAIL 포함)
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key

    prompt = PromptTemplate.from_template(_VERIFY_PROMPT_TEMPLATE)
    verifier_llm = _get_llm(
        model=model,
        temperature=0.0,
        api_key=openai_api_key,
        streaming=False,
    )
    chain = prompt | verifier_llm | StrOutputParser()

    return _retry_api_call(chain.invoke, {
        "context": context_text,
        "question": query,
        "answer": answer,
    })


# ──────────────────────────────────────────────────────────────────────
# 자기 교정 루프 (Self-Correction Loop)
# ──────────────────────────────────────────────────────────────────────

async def self_correction_loop(
    query: str,
    context_text: str,
    initial_answer: str,
    initial_verify_result: str,
    openai_api_key: str,
    gen_model: str = "gpt-5",
    max_rounds: int = 2,
    initial_ragas_result: dict = None,
    embeddings = None,
    final_docs = None,
):
    """
    FAIL 판정 시 검증 결과 및 RAGAS 지표를 분석하여 프롬프트 템플릿을 자동 최적화하고 재생성합니다.
    (비동기 제너레이터로 전환하여 토큰 스트리밍 지원)
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key

    current_answer = initial_answer
    last_verify_result = initial_verify_result
    last_ragas_result = initial_ragas_result or {"faithfulness": 0.0, "answer_relevancy": 0.0}
    current_template = _ANSWER_PROMPT_TEMPLATE

    # Round 0 = 최초 시도 로그
    correction_logs: list[dict] = [
        {
            "round": 0,
            "answer": initial_answer,
            "verify_result": initial_verify_result,
            "ragas_result": last_ragas_result,
            "prompt_template": current_template,
        }
    ]

    for round_num in range(1, max_rounds + 1):
        # "FAIL"이 없고 "PASS"만 있거나, [최종 판정]이 PASS이면 종료
        u_verify = last_verify_result.upper()
        if "FAIL" not in u_verify or "[최종 판정]: PASS" in u_verify:
            break

        # 1. 프롬프트 최적화 (GPT-5.2 사용)
        optimizer_llm = _get_llm(
            model="gpt-5.2",
            temperature=0.0,
            api_key=openai_api_key,
            streaming=False,
        )
        optimizer_prompt = PromptTemplate.from_template(_OPTIMIZER_PROMPT_TEMPLATE)
        optimizer_chain = optimizer_prompt | optimizer_llm | StrOutputParser()

        yield ("status", {"step": f"프롬프트 최적화 중 (Round {round_num})...", "icon": "⚙️"})
        
        # Optimizer는 텍스트를 바로 반환하므로 invoke 사용
        new_template = await _async_retry_api_call(optimizer_chain.ainvoke, {
            "question": query,
            "verify_result": last_verify_result,
            "ragas_result": json.dumps(last_ragas_result, ensure_ascii=False),
            "original_template": current_template,
        })
        current_template = new_template

        # 2. 새 프롬프트로 답변 재생성 (스트리밍)
        yield ("status", {"step": f"최적화된 프롬프트로 재생성 중...", "icon": "✍️"})
        yield ("token", f"\n\n---\n🔄 **자동 최적화된 답변 ({round_num}회차):**\n\n")
        
        current_answer = ""
        async_stream = await generate_answer(
            query=query,
            context_text=context_text,
            openai_api_key=openai_api_key,
            model=gen_model,
            prompt_template_str=current_template,
            stream=True,
            async_mode=True
        )

        async for chunk in async_stream:
            if chunk:
                current_answer += chunk
                yield ("token", chunk)

        # 3. 재검증 (GPT-5.2)
        yield ("status", {"step": f"교정 답변 검증 중...", "icon": "🧐"})
        last_verify_result = verify_answer(
            query, context_text, current_answer, openai_api_key, model="gpt-5.2"
        )

        # 4. RAGAS 지표 생략 (속도를 위해 교정 루프 중에는 측정하지 않음)
        # 최종 결과에서만 1회 측정하도록 api.py에서 제어 권장
        last_ragas_result = {"faithfulness": 0.0, "answer_relevancy": 0.0}

        # 라운드 로그 기록
        correction_logs.append({
            "round": round_num,
            "answer": current_answer,
            "verify_result": last_verify_result,
            "ragas_result": last_ragas_result,
            "prompt_template": current_template,
        })

    # 교정 실제 수행 횟수 계산
    # FAIL로 시작해서 PASS로 끝났으면 loop가 돌았음.
    actual_rounds = round_num - 1
    if "PASS" in last_verify_result.upper() and actual_rounds == 0 and "FAIL" in initial_verify_result.upper():
        # 이 경우는 이론상 1회는 돌아야 함 (최소 1회 진입 후 PASS가 되었으므로)
        actual_rounds = 1
    
    # 더 정확한 계산: logs 길이를 활용 (초기 로그 1개 + 라운드별 1개)
    actual_rounds = len(correction_logs) - 1

    yield ("done_loop", {
        "answer": current_answer,
        "verify_result": last_verify_result,
        "rounds": actual_rounds,
        "logs": correction_logs,
        "ragas": last_ragas_result
    })


# ──────────────────────────────────────────────────────────────────────
# RAGAS 평가
# ──────────────────────────────────────────────────────────────────────

def evaluate_with_ragas(
    query: str,
    answer: str,
    final_docs: list[Document],
    embeddings,
    openai_api_key: str,
    eval_model: str = "gpt-5.2",
) -> dict[str, float]:
    """
    RAGAS로 RAG 파이프라인의 faithfulness와 answer_relevancy를 평가합니다.
    (사용자 요청에 따라 GPT-5.2 사용 및 Temperature 설정 해제)
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # Ragas 0.4.x 이상 데이터셋 규격 준수 (question, answer, contexts)
    # user_input 대신 question, response 대신 answer, retrieved_contexts 대신 contexts 사용 가능성 확인
    ragas_data = {
        "question": [query],
        "answer": [answer],
        "contexts": [
            [d.page_content.replace("passage: ", "")[:1500] for d in final_docs]
        ],
    }
    dataset = Dataset.from_dict(ragas_data)
    
    # GPT-5.2 계열은 temperature 설정을 허용하지 않는 경우가 많아 기본값을 사용하게 합니다.
    eval_llm = ChatOpenAI(model=eval_model)

    try:
        results = _retry_api_call(
            lambda p: evaluate(
                dataset=p["dataset"],
                metrics=p["metrics"],
                llm=p["llm"],
                embeddings=p["embeddings"],
            ),
            {
                "dataset": dataset,
                "metrics": [faithfulness, answer_relevancy],
                "llm": eval_llm,
                "embeddings": embeddings,
            }
        )
        df = results.to_pandas()
        logger.info("[RAGAS] Evaluation successful. Columns: %s", df.columns.tolist())
        
        # 컬럼명 유연하게 찾기 (버전에 따라 'faithfulness' 또는 'faithfulness.score' 등일 수 있음)
        def _get_metric_val(keywords):
            for col in df.columns:
                if any(k.lower() in col.lower() for k in keywords):
                    return df.iloc[0][col]
            return 0.0

        f_val = _get_metric_val(["faithfulness"])
        r_val = _get_metric_val(["relevancy", "relevance"])

        import math
        def _safe(val: float) -> float:
            try:
                v = float(val)
                return 0.0 if (math.isnan(v) or math.isinf(v)) else max(0.0, min(1.0, v))
            except Exception:
                return 0.0

        return {
            "faithfulness": _safe(f_val),
            "answer_relevancy": _safe(r_val),
        }
    except Exception as e:
        logger.error("[RAGAS] Evaluation Error: %s", e)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
