"""
external_pharma_api.py
외부 약학 API(OpenFDA)에서 문서를 가져와 RAG 검색 후보로 변환합니다.

기본 동작:
- OPENFDA Drug Label API 조회
- 응답을 LangChain Document 리스트로 정규화
- 실패 시 빈 리스트 반환(기존 파이프라인 영향 최소화)
"""

from __future__ import annotations

import json
import logging
import os
import re
import html
import time
from urllib.parse import quote_plus, unquote_plus
from copy import deepcopy
from typing import Any

import requests
from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_translation_client: OpenAI | None = None
_translation_cache: dict[str, str] = {}
_llm_router_cache: dict[str, list[str]] = {}
# _external_provider_status: dict[str, dict[str, Any]] = {
#     "openfda": {
#         "provider": "openfda",
#         "connected": None,
#         "message": "not_checked",
#         "http_status": None,
#         "last_checked": None,
#     },
# }
_external_provider_status: dict[str, dict[str, Any]] = {}  # 모든 외부 제공자 제거됨


def _set_provider_status(
    provider: str,
    connected: bool | None,
    message: str,
    http_status: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "provider": provider,
        "connected": connected,
        "message": message,
        "http_status": http_status,
        "last_checked": int(time.time()),
    }
    if extra:
        payload.update(extra)
    _external_provider_status[provider] = payload


def get_external_api_status(provider: str = "") -> dict[str, Any]:
    """외부 API 연결/응답 상태 스냅샷을 반환합니다. (현재 모두 비활성화)"""
    return {
        "provider": provider or "external",
        "connected": False,
        "message": "all_external_apis_disabled",
        "http_status": None,
        "last_checked": int(time.time()),
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _translation_settings() -> dict[str, Any]:
    return {
        "enabled": _env_bool("EXTERNAL_AUTO_TRANSLATE", False),
        "model": os.getenv("EXTERNAL_TRANSLATE_MODEL", "gpt-5-nano"),
        "max_chars": _env_int("EXTERNAL_TRANSLATE_MAX_CHARS", 1800),
    }


def _llm_router_settings() -> dict[str, Any]:
    return {
        "enabled": _env_bool("EXTERNAL_LLM_QUERY_ROUTER", False),
        "model": os.getenv("EXTERNAL_LLM_QUERY_ROUTER_MODEL", "gpt-5-nano"),
        "max_keywords": _env_int("EXTERNAL_LLM_QUERY_ROUTER_MAX_KEYWORDS", 6),
    }


# 한국어 질의 대응을 위한 최소 매핑(데모/발표용)
_KO_EN_KEYWORD_MAP: dict[str, list[str]] = {
    "타이레놀": ["acetaminophen", "paracetamol", "tylenol"],
    "아세트아미노펜": ["acetaminophen", "paracetamol"],
    "이부프로펜": ["ibuprofen"],
    "루테인": ["lutein"],
    "오메가3": ["omega-3", "fish oil"],
    "비타민d": ["vitamin d", "cholecalciferol"],
    "건조": ["dry eye", "xerophthalmia"],
    "눈": ["eye", "ocular"],
    "충혈": ["red eye", "hyperemia"],
    "알레르기": ["allergy", "allergic"],
    "소염진통제": ["nsaid", "nonsteroidal anti-inflammatory"],
}

# 한국 공공 API(itemName 중심 검색) 보강을 위한 증상→의약품 키워드 힌트
_KO_KR_DRUG_HINTS: dict[str, list[str]] = {
    "눈": ["인공눈물", "히알루론산", "카르복시메틸셀룰로오스"],
    "건조": ["인공눈물", "히알루론산", "카르복시메틸셀룰로오스"],
    "충혈": ["나파졸린", "테트라히드로졸린"],
    "알레르기": ["항히스타민", "세티리진", "로라타딘"],
    "두통": ["아세트아미노펜", "이부프로펜"],
    "통증": ["아세트아미노펜", "이부프로펜"],
    "소염": ["이부프로펜", "나프록센"],
    "소화": ["판토프라졸", "파모티딘"],
}


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v if x is not None)
    return str(v)


def _clean_text(v: Any) -> str:
    raw = _to_text(v)
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_doc(item: dict[str, Any], idx: int) -> Document:
    openfda = item.get("openfda", {}) if isinstance(item.get("openfda"), dict) else {}

    payload = {
        "api_provider": "openfda",
        "brand_name": _to_text(openfda.get("brand_name")),
        "generic_name": _to_text(openfda.get("generic_name")),
        "manufacturer_name": _to_text(openfda.get("manufacturer_name")),
        "substance_name": _to_text(openfda.get("substance_name")),
        "route": _to_text(openfda.get("route")),
        "product_type": _to_text(openfda.get("product_type")),
        "indications_and_usage": _to_text(item.get("indications_and_usage")),
        "dosage_and_administration": _to_text(item.get("dosage_and_administration")),
        "warnings": _to_text(item.get("warnings")),
        "adverse_reactions": _to_text(item.get("adverse_reactions")),
    }

    settings = _translation_settings()
    if settings["enabled"]:
        src = _build_translation_source(payload, max_chars=settings["max_chars"])
        ko_summary = _translate_to_korean_summary(src, model=settings["model"])
        if ko_summary:
            payload["korean_summary"] = ko_summary
            payload["translation_applied"] = True
        else:
            payload["translation_applied"] = False

    # processor.clean_json_to_text와 호환되도록 "passage: {dict}" 포맷 유지
    page_content = f"passage: {payload}"
    return Document(
        page_content=page_content,
        metadata={
            "source": f"OPENFDA_DRUG_LABEL_{idx}",
            "provider": "openfda",
            "is_external": True,
        },
    )


def _contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def _build_translation_source(payload: dict[str, Any], max_chars: int) -> str:
    lines = [
        f"Brand: {payload.get('brand_name', '')}",
        f"Generic: {payload.get('generic_name', '')}",
        f"Substance: {payload.get('substance_name', '')}",
        f"Route: {payload.get('route', '')}",
        f"Indications: {payload.get('indications_and_usage', '')}",
        f"Dosage: {payload.get('dosage_and_administration', '')}",
        f"Warnings: {payload.get('warnings', '')}",
        f"Adverse reactions: {payload.get('adverse_reactions', '')}",
    ]
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _get_translation_client() -> OpenAI | None:
    global _translation_client
    if _translation_client is not None:
        return _translation_client
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        _translation_client = OpenAI(api_key=api_key)
        return _translation_client
    except Exception as e:
        logger.warning("OpenAI translation client init failed: %s", e)
        return None


def _translate_to_korean_summary(text: str, model: str) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    if _contains_korean(src):
        return src
    cache_key = f"{model}::{src}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    client = _get_translation_client()
    if client is None:
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 의약품 문서를 한국어로 번역·요약하는 약학 어시스턴트입니다. "
                "핵심 정보(효능, 복용/용법, 주의사항, 이상반응)를 과장 없이 한국어로 간결히 정리하세요."
            ),
        },
        {
            "role": "user",
            "content": (
                "다음 의약품 라벨 정보를 한국어로 6~10줄 요약 번역해줘.\n"
                "용량 수치/금기/주의사항은 가능한 원문 의미를 보존해.\n\n"
                f"{src}"
            ),
        },
    ]

    try:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
            )
        except Exception as e:
            msg = str(e).lower()
            if "temperature" in msg and "default" in msg:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            else:
                raise
        out = (resp.choices[0].message.content or "").strip()
        if out:
            _translation_cache[cache_key] = out
        return out
    except Exception as e:
        logger.warning("OpenFDA summary translation failed: %s", e)
        return ""


def _expand_query_candidates(query: str, max_terms: int = 8) -> list[str]:
    """한국어 질의를 OpenFDA 검색용 영어 키워드로 보강합니다."""
    q = (query or "").strip().lower()
    if not q:
        return []

    candidates: list[str] = [query.strip()]

    if _contains_korean(q):
        for ko, en_list in _KO_EN_KEYWORD_MAP.items():
            if ko in q:
                candidates.extend(en_list)

    # 영문/숫자/기호 기반 토큰도 살림 (예: ibuprofen 200mg)
    ascii_terms = re.findall(r"[a-zA-Z0-9\-\+\.]{3,}", query)
    candidates.extend(ascii_terms)

    # 순서 유지 중복 제거
    dedup: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        t = term.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
        if len(dedup) >= max_terms:
            break
    return dedup


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _expand_with_llm_router(query: str, base_candidates: list[str], target: str) -> list[str]:
    settings = _llm_router_settings()
    if not settings["enabled"]:
        return base_candidates

    q = (query or "").strip()
    if not q:
        return base_candidates

    cache_key = f"{target}::{settings['model']}::{q}"
    if cache_key in _llm_router_cache:
        llm_terms = _llm_router_cache[cache_key]
    else:
        client = _get_translation_client()
        if client is None:
            return base_candidates

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "너는 약학 검색 키워드 라우터다. "
                        "질문을 API 검색용 키워드로 변환해라. "
                        "반드시 JSON만 출력한다."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"질문: {q}\n"
                        f"검색 대상: {target}\n"
                        "출력 형식(JSON): {\"keywords\": [\"...\"]}\n"
                        "규칙: 브랜드명/성분명/동의어를 한국어+영어 혼합으로 최대한 짧게, "
                        f"최대 {settings['max_keywords']}개"
                    ),
                },
            ]

            try:
                resp = client.chat.completions.create(
                    model=settings["model"],
                    temperature=0,
                    messages=messages,
                )
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg and "default" in msg:
                    resp = client.chat.completions.create(
                        model=settings["model"],
                        messages=messages,
                    )
                else:
                    raise

            content = (resp.choices[0].message.content or "").strip()
            obj = _extract_json_object(content) or {}
            raw_keywords = obj.get("keywords", []) if isinstance(obj, dict) else []
            llm_terms = []
            if isinstance(raw_keywords, list):
                for x in raw_keywords:
                    if isinstance(x, str) and x.strip():
                        llm_terms.append(x.strip())
                        if len(llm_terms) >= settings["max_keywords"]:
                            break
            _llm_router_cache[cache_key] = llm_terms
            logger.info("[LLM-Router] target=%s query='%s' keywords=%s", target, q, llm_terms)
        except Exception as e:
            logger.warning("LLM query router failed (target=%s): %s", target, e)
            llm_terms = []

    merged: list[str] = []
    seen: set[str] = set()
    for term in base_candidates + llm_terms:
        t = (term or "").strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        merged.append(t)

    return merged


def _expand_korean_api_candidates(query: str, max_terms: int = 8) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []

    candidates: list[str] = [q]
    q_norm = q.lower()

    # 공백 분해 토큰(너무 짧은 토큰 제외)
    for token in re.split(r"\s+", q):
        t = token.strip()
        if len(t) >= 2:
            candidates.append(t)

    # 증상 기반 힌트 확장
    for key, hints in _KO_KR_DRUG_HINTS.items():
        if key in q_norm:
            candidates.extend(hints)

    dedup: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        t = term.strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(t)
        if len(dedup) >= max_terms:
            break
    return dedup


def fetch_openfda_docs(query: str, top_k: int = 4, timeout_sec: float = 8.0) -> list[Document]:
    """
    OpenFDA Drug Label API에서 query 관련 문서를 조회합니다.

    참고: OpenFDA는 영어 중심 데이터셋이라 한국어 질의는 매핑 보강이 필요합니다.
    """
    if not query.strip():
        return []

    base_url = "https://api.fda.gov/drug/label.json"
    api_key = os.getenv("OPENFDA_API_KEY", "").strip()
    candidates: list[dict[str, Any]] = []
    had_success_response = False
    last_http_status: int | None = None
    last_error: str = ""

    query_candidates = _expand_query_candidates(query)
    query_candidates = _expand_with_llm_router(query, query_candidates, target="openfda")
    for term in query_candidates:
        field_query = (
            f'openfda.generic_name:"{term}"'
            f'+OR+openfda.brand_name:"{term}"'
            f'+OR+openfda.substance_name:"{term}"'
            f'+OR+indications_and_usage:"{term}"'
        )

        for search_query in (field_query, term):
            params = {
                "search": search_query,
                "limit": max(1, min(int(top_k), 10)),
            }
            if api_key:
                params["api_key"] = api_key

            try:
                resp = requests.get(base_url, params=params, timeout=timeout_sec)
                last_http_status = resp.status_code
                if resp.status_code != 200:
                    continue
                had_success_response = True
                data = resp.json()
                results = data.get("results", [])
                if isinstance(results, list) and results:
                    candidates = results
                    break
            except Exception as e:
                last_error = str(e)
                logger.warning("OpenFDA fetch failed: %s", e)

        if candidates:
            logger.info(
                "[OpenFDA] query='%s' mapped_term='%s' fetched=%d",
                query,
                term,
                len(candidates),
            )
            break

    docs: list[Document] = []
    for i, item in enumerate(candidates[:top_k], 1):
        if isinstance(item, dict):
            docs.append(_make_doc(item, i))

    if had_success_response:
        _set_provider_status(
            provider="openfda",
            connected=True,
            message="ok" if docs else "connected_no_results",
            http_status=last_http_status,
            extra={"fetched": len(docs)},
        )
    else:
        _set_provider_status(
            provider="openfda",
            connected=False,
            message=(f"http_{last_http_status}" if last_http_status else (last_error or "request_failed")),
            http_status=last_http_status,
            extra={"fetched": len(docs)},
        )

    return docs


def _make_mfds_doc(item: dict[str, Any], idx: int) -> Document:
    item_name = _clean_text(item.get("itemName"))
    entp_name = _clean_text(item.get("entpName"))

    payload = {
        "api_provider": "mfds_ezdrug",
        "item_name": item_name,
        "company_name": entp_name,
        "efficacy": _clean_text(item.get("efcyQesitm")),
        "usage": _clean_text(item.get("useMethodQesitm")),
        "caution": _clean_text(item.get("atpnQesitm")),
        "interaction": _clean_text(item.get("intrcQesitm")),
        "side_effect": _clean_text(item.get("seQesitm")),
        "storage": _clean_text(item.get("depositMethodQesitm")),
    }
    page_content = f"passage: {payload}"
    return Document(
        page_content=page_content,
        metadata={
            "source": f"MFDS_EASY_DRUG_{idx}_{item_name or 'UNKNOWN'}",
            "provider": "mfds_ezdrug",
            "is_external": True,
            "lang": "ko",
        },
    )


def _extract_items_from_mfds_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    # 응답 형식 변동 대응
    paths = [
        data.get("body", {}).get("items"),
        data.get("response", {}).get("body", {}).get("items"),
    ]
    for p in paths:
        if isinstance(p, list):
            return [x for x in p if isinstance(x, dict)]
        if isinstance(p, dict):
            # 일부 API는 items.item 형태
            item = p.get("item")
            if isinstance(item, list):
                return [x for x in item if isinstance(x, dict)]
            if isinstance(item, dict):
                return [item]
    return []


def fetch_mfds_ezdrug_docs(query: str, top_k: int = 4, timeout_sec: float = 8.0) -> list[Document]:
    """식약처 공공데이터(의약품개요정보 e약은요) 조회."""
    raw_key = os.getenv("MFDS_API_KEY", "").strip()
    encoded_key = os.getenv("MFDS_API_KEY_ENCODED", "").strip()
    if not raw_key and not encoded_key:
        _set_provider_status(
            provider="mfds_ezdrug",
            connected=False,
            message="key_missing",
            http_status=None,
            extra={"fetched": 0},
        )
        logger.warning("MFDS_API_KEY is not set. Skip mfds_ezdrug provider.")
        return []

    key_candidates: list[str] = []
    for k in (raw_key, encoded_key):
        if not k:
            continue
        key_candidates.append(k)
        try:
            decoded = unquote_plus(k)
            if decoded and decoded not in key_candidates:
                key_candidates.append(decoded)
        except Exception:
            pass
        try:
            encoded = quote_plus(k, safe="")
            if encoded and encoded not in key_candidates:
                key_candidates.append(encoded)
        except Exception:
            pass

    base_url = os.getenv(
        "MFDS_EZDRUG_API_URL",
        "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList",
    ).strip()

    candidates = _expand_korean_api_candidates(query, max_terms=8)
    candidates = _expand_with_llm_router(query, candidates, target="mfds_ezdrug")
    merged: list[Document] = []
    seen: set[str] = set()
    had_success_response = False
    had_forbidden = False
    last_http_status: int | None = None

    for term in candidates:
        for service_key in key_candidates:
            params = {
                "serviceKey": service_key,
                "pageNo": 1,
                "numOfRows": max(3, min(top_k * 2, 15)),
                "itemName": term,
                "type": "json",
            }
            try:
                resp = requests.get(base_url, params=params, timeout=timeout_sec)
                last_http_status = resp.status_code
                if resp.status_code == 403:
                    # 키 인코딩/서비스 신청 상태 이슈 가능성 → 다른 키 포맷 시도
                    had_forbidden = True
                    continue
                if resp.status_code != 200:
                    continue
                had_success_response = True
                data = resp.json()
                items = _extract_items_from_mfds_json(data)
                if not items:
                    continue

                for it in items:
                    doc = _make_mfds_doc(it, len(merged) + 1)
                    key = f"{doc.metadata.get('source','')}::{doc.page_content[:180]}"
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(doc)
                    if len(merged) >= max(top_k, 1) * 3:
                        break
                if len(merged) >= max(top_k, 1) * 3:
                    break
            except Exception as e:
                logger.warning("MFDS ezdrug fetch failed (term=%s): %s", term, e)

        if len(merged) >= max(top_k, 1) * 3:
            break

    logger.info("[MFDS-EZDRUG] query='%s' fetched=%d", query, len(merged))

    if had_success_response:
        _set_provider_status(
            provider="mfds_ezdrug",
            connected=True,
            message="ok" if merged else "connected_no_results",
            http_status=last_http_status,
            extra={"fetched": len(merged)},
        )
    else:
        _set_provider_status(
            provider="mfds_ezdrug",
            connected=False,
            message="forbidden_403" if had_forbidden else (f"http_{last_http_status}" if last_http_status else "request_failed"),
            http_status=last_http_status,
            extra={"fetched": len(merged)},
        )

    return merged


def fetch_external_pharma_docs(
    query: str,
    provider: str = "openfda",
    top_k: int = 4,
    timeout_sec: float = 8.0,
) -> list[Document]:
    """
    외부 약학 API 문서를 조회합니다. (비활성화 상태 - 항상 빈 리스트 반환)
    """
    return []
