"""
processor.py
JSON 전처리 로직 및 Kiwi 형태소 분석 함수.
"""

import ast
import gc
import torch
from kiwipiepy import Kiwi
from langchain_core.documents import Document


# ──────────────────────────────────────────
# GPU 메모리 관리
# ──────────────────────────────────────────

def clear_gpu() -> None:
    """GPU 캐시를 비워 메모리 누수를 방지합니다."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_status(device_index: int | None = None) -> dict:
    """현재 CUDA 디바이스 상태를 반환합니다.

    Returns:
        {
            "available": bool,
            "index": int,
            "name": str,
            "total_gb": float,
            "used_gb": float,
            "reserved_gb": float,
            "used_pct": float,
            "pct": float,  # reserved 기준(기존 호환)
        }
    """
    if not torch.cuda.is_available():
        return {"available": False}

    idx = torch.cuda.current_device() if device_index is None else device_index
    props = torch.cuda.get_device_properties(idx)

    total = props.total_memory / 1024**3
    used = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3

    return {
        "available": True,
        "index": idx,
        "name": props.name,
        "total_gb": round(total, 2),
        "used_gb": round(used, 2),
        "reserved_gb": round(reserved, 2),
        "used_pct": round((used / total) * 100, 1) if total > 0 else 0.0,
        # 작업관리자 표시값과의 체감 일치를 위해 reserved 비율 유지
        "pct": round((reserved / total) * 100, 1) if total > 0 else 0.0,
    }


# ──────────────────────────────────────────
# JSON → 사람이 읽는 텍스트 변환
# ──────────────────────────────────────────

def clean_json_to_text(raw_content: str) -> str:
    """
    벡터 DB에 저장된 JSON 형식 문자열을 사람이 읽기 쉬운 텍스트로 변환합니다.
    모든 키-값 쌍을 유지하여 정보 손실을 방지합니다.

    Args:
        raw_content: page_content 원문 (예: "passage: {'일반명': '루테인', ...}")

    Returns:
        정제된 텍스트 문자열
    """
    content = raw_content.replace("passage: ", "")
    try:
        # 안전한 Dict 변환
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            # 모든 키-값을 "Key: Value" 형태로 변환
            items = []
            for k, v in data.items():
                if v: # 값이 있는 경우만
                    if isinstance(v, list):
                        v_str = ", ".join(map(str, v))
                    else:
                        v_str = str(v)
                    items.append(f"{k}: {v_str}")
            return ", ".join(items)
        return content
    except Exception:
        return content


def get_clean_doc_text(doc: Document) -> str:
    """
    Document의 정제된 텍스트를 캐시하여 중복 전처리를 피합니다.

    Args:
        doc: LangChain Document 객체

    Returns:
        클린 텍스트 문자열
    """
    metadata = doc.metadata or {}
    clean_text = metadata.get("_clean_text")
    if clean_text is None:
        clean_text = clean_json_to_text(doc.page_content)
        metadata = {**metadata, "_clean_text": clean_text}
        doc.metadata = metadata
    return clean_text


# ──────────────────────────────────────────
# Kiwi 형태소 분석 및 BM25 코퍼스 구축
# ──────────────────────────────────────────

def get_kiwi_tokenizer() -> Kiwi:
    """Kiwi 멀티코어 형태소 분석기를 초기화합니다.
    
    num_workers=4: GPU 작업과 CPU 코어 경쟁을 줄이면서 병렬성 확보.
    -1(전 코어) 사용 시 GPU 스레드와 경합하여 오히려 느릴 수 있음.
    """
    return Kiwi(num_workers=4)


def tokenize_corpus(docs: list[Document], kiwi: Kiwi) -> list[list[str]]:
    """
    Document 리스트를 BM25용 토큰 코퍼스로 변환합니다.
    Kiwi의 내장 배치 처리(tokenize 리스트 입력)로 멀티코어 활용.

    Args:
        docs: langchain Document 객체 리스트
        kiwi:  초기화된 Kiwi 인스턴스

    Returns:
        토큰 리스트의 리스트 (각 문서에 대한 형태소 목록)
    """
    # 검색에 유효한 품사만 필터링 (명사/동사/어근/숫자/형용사)
    TARGET_TAGS = ("N", "V", "X", "S", "VA")

    clean_texts = [d.page_content.replace("passage: ", "") for d in docs]
    tokenized_corpus: list[list[str]] = []

    # Kiwi.tokenize()에 리스트를 전달하면 배치 멀티코어 처리
    for tokens in kiwi.tokenize(clean_texts):
        tokenized_corpus.append(
            [t.form for t in tokens if t.tag.startswith(TARGET_TAGS)]
        )

    return tokenized_corpus


def tokenize_query(query: str, kiwi: Kiwi) -> str:
    """
    검색 쿼리를 Kiwi로 형태소 분석하여 BM25 검색용 텍스트로 변환합니다.

    Args:
        query: 사용자 질문 (또는 Query Expansion된 키워드 텍스트)
        kiwi:  초기화된 Kiwi 인스턴스

    Returns:
        공백으로 연결된 형태소 문자열
    """
    # Tag 일관성 유지 (V 추가)
    TARGET_TAGS = ("N", "V", "X", "S", "VA")
    tokens = [
        t.form
        for t in kiwi.tokenize(query)
        if t.tag.startswith(TARGET_TAGS)
    ]
    return " ".join(tokens)
