"""
app.py
약사 챗봇 RAG - Streamlit 메인 앱.

실행: streamlit run app.py
"""

import os
import time
import asyncio

import streamlit as st
import torch
from dotenv import load_dotenv

# ── CUDA 최적화 플래그 (로드 전 설정) ───────────────────────────────
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
from generator import (
    build_context,
    generate_answer,
    verify_answer,
    self_correction_loop,
    evaluate_with_ragas,
    get_query_optimizer,
)
from processor import clear_gpu, get_clean_doc_text, get_gpu_status

# ──────────────────────────────────────────────────────────────────────
# 환경 변수 로드
# ──────────────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💊 약사 AI 챗봇",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# CSS 테마 (프리미엄 의료 다크 스타일)
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── 전체 배경 ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a1220 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── 사이드바 ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2e 0%, #091424 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #60a5fa;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* ── 메인 텍스트 색상 ── */
    .stMarkdown p, .stMarkdown li { color: #cbd5e1; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e2e8f0; }

    /* ── 채팅 메시지 ── */
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.12);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(8px);
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(30, 58, 96, 0.5) !important;
        border-color: rgba(59, 130, 246, 0.25);
    }

    /* ── 히어로 헤더 ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(37,99,235,0.15) 0%, rgba(16,185,129,0.08) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.3);
        color: #34d399;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 20px;
        padding: 0.15rem 0.65rem;
        margin-right: 0.4rem;
        margin-top: 0.6rem;
    }
    .badge-blue {
        background: rgba(59,130,246,0.1);
        border-color: rgba(59,130,246,0.3);
        color: #60a5fa;
    }
    .badge-purple {
        background: rgba(139,92,246,0.1);
        border-color: rgba(139,92,246,0.3);
        color: #a78bfa;
    }

    /* ── PASS / FAIL 배지 ── */
    .verdict-pass {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16,185,129,0.12);
        border: 1.5px solid rgba(16,185,129,0.4);
        color: #34d399;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.4rem 1.1rem;
        margin-top: 0.6rem;
        letter-spacing: 0.05em;
    }
    .verdict-fail {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(239,68,68,0.1);
        border: 1.5px solid rgba(239,68,68,0.35);
        color: #f87171;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.4rem 1.1rem;
        margin-top: 0.6rem;
        letter-spacing: 0.05em;
    }
    .verdict-corrected {
        font-size: 0.75rem;
        font-weight: 500;
        opacity: 0.8;
        margin-left: 0.3rem;
    }

    /* ── 소스 배지 ── */
    .source-badge {
        font-size: 0.7rem;
        color: #60a5fa;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 4px;
        padding: 2px 7px;
        margin-right: 5px;
        font-weight: 500;
    }
    .rank-badge {
        font-size: 0.7rem;
        color: #a78bfa;
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 4px;
        padding: 2px 7px;
        font-weight: 600;
    }

    /* ── 예시 질문 카드 ── */
    .example-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ── 교정 로그 카드 ── */
    .correction-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .round-label-pass { color: #34d399; font-size: 0.8rem; font-weight: 700; }
    .round-label-fail { color: #f87171; font-size: 0.8rem; font-weight: 700; }

    /* ── Expander 스타일 ── */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 10px;
    }

    /* ── 구분선 ── */
    hr { border-color: rgba(59,130,246,0.15) !important; }

    /* ── 탭 ── */
    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #60a5fa;
        border-bottom-color: #60a5fa;
    }

    /* ── VRAM 고정 위젯 ── */
    .vram-widget {
        position: fixed;
        top: 56px;
        right: 16px;
        z-index: 9999;
        background: rgba(13, 27, 46, 0.92);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 0.45rem 0.85rem;
        min-width: 210px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .vram-title {
        font-size: 0.65rem;
        font-weight: 600;
        color: #60a5fa;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .vram-bar-bg {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.2rem 0;
    }
    .vram-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    .vram-text {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# 환경 변수
# ──────────────────────────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────────────────
# ── 하드코딩된 DB 경로 ──
db_path = "./chroma_db_combined_1771477980"

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0.5rem 0 1rem;'>"
        "<span style='font-size:2rem;'>💊</span>"
        "<div style='color:#60a5fa;font-size:1.1rem;font-weight:700;margin-top:0.25rem;'>약사 AI</div>"
        "<div style='color:#475569;font-size:0.72rem;'>Pharmacist RAG System</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── 모델 선택 ──
    st.markdown("### 🤖 GPT 모델")
    gen_model = st.selectbox(
        "답변 생성 모델",
        options=["gpt-5", "gpt-4o"],
        index=0,
        help="답변 생성에 사용할 GPT 모델 (기본: gpt-5)",
        label_visibility="collapsed",
    )
    use_query_expansion = True  # 항상 활성화

    st.divider()

    # ── 검색 가중치 ──
    st.markdown("### 🔍 앙상블 가중치")
    weight_bm25 = st.slider("BM25 (키워드)", 0.0, 1.0, 0.8, 0.05, label_visibility="visible")
    weight_vector = round(1.0 - weight_bm25, 2)
    st.caption(f"벡터 검색: **{weight_vector}** (자동)")

    top_k    = st.slider("Top-K (최종 문서 수)", 3, 10, 5)
    ensemble_k = st.slider("앙상블 후보 수", 10, 50, 20)

    st.divider()

    # ── 고급 옵션 ──
    st.markdown("### 🧪 고급 옵션")
    use_self_correction = st.checkbox("🔄 자기 교정 루프", value=True, help="FAIL 시 최대 3회 재시도")
    use_ragas           = st.checkbox("📊 RAGAS 평가", value=False, help="추가 API 비용 발생")

    st.divider()

    # ── 예시 질문 ──
    st.markdown("### 💡 예시 질문")
    EXAMPLE_QUESTIONS = [
        "눈이 침침한데 뭐 먹으면 될까요?",
        "타이레놀과 이부프로펜 같이 먹어도 되나요?",
        "루테인 하루 복용량이 어떻게 되나요?",
        "임산부가 먹어도 되는 영양제가 있나요?",
        "간에 좋은 약은 어떤 게 있나요?",
    ]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    for eq in EXAMPLE_QUESTIONS:
        if st.button(eq, key=f"ex_{eq}", use_container_width=True):
            st.session_state.pending_question = eq

    st.divider()

    # ── 대화 초기화 ──
    if st.button("🗑️ 대화 초기화", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pending_question = ""
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
# VRAM 고정 위젯 (우측 상단)
# ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    _gpu = get_gpu_status()
    _gpu_name = _gpu["name"]
    _vram_total = _gpu["total_gb"]
    _vram_used = _gpu["used_gb"]
    _vram_rsrvd = _gpu["reserved_gb"]
    _pct = min((_gpu["used_pct"] / 100.0), 1.0)
    _bar_color = "#34d399" if _pct < 0.6 else ("#facc15" if _pct < 0.85 else "#f87171")
    st.markdown(
        f"""
        <div class="vram-widget">
            <div class="vram-title">💻 GPU · {_gpu_name}</div>
            <div class="vram-bar-bg">
                <div class="vram-bar-fill" style="width:{_pct*100:.1f}%;background:{_bar_color};"></div>
            </div>
            <div class="vram-text">VRAM {_vram_used:.1f} / {_vram_total:.1f} GB &nbsp;·&nbsp; 예약 {_vram_rsrvd:.1f} GB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class=\"vram-widget\"><div class=\"vram-title\">⚠️ GPU 미인식 – CPU 모드</div></div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# 메인 히어로 헤더
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
      <div class="hero-title">💊 약사 AI 챗봇</div>
      <div class="hero-subtitle">
        외부 의약품 데이터베이스 기반 RAG 시스템 · 근거 중심 답변 생성
      </div>
      <div style="margin-top:0.6rem;">
        <span class="status-badge">✦ BGE-M3-ko 임베딩</span>
        <span class="status-badge badge-blue">✦ BM25 + 벡터 앙상블</span>
        <span class="status-badge badge-purple">✦ CrossEncoder 리랭킹</span>
        <span class="status-badge">✦ GPT 자기검증</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# DB 경로 유효성 확인
# ──────────────────────────────────────────────────────────────────────
if not os.path.exists(db_path):
    st.error(
        f"❌ ChromaDB 경로를 찾을 수 없습니다: `{db_path}`\n\n"
        "사이드바에서 올바른 ChromaDB 폴더 경로를 입력해 주세요.",
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# 리소스 초기화 (최초 1회)
# ──────────────────────────────────────────────────────────────────────
if "_initialized" not in st.session_state:
    with st.status("⏳ 리소스 초기화 중...", expanded=True) as status:
        t_init = time.time()

        st.write("🧬 BGE-M3 임베딩 모델 로드 중...")
        embeddings = load_embeddings()
        st.write(f"✔️ 임베딩 모델 로드 완료 ({time.time() - t_init:.1f}s)")

        t_db = time.time()
        st.write("📦 ChromaDB 로드 중...")
        vector_db = load_vector_db(db_path)
        st.write(f"✔️ ChromaDB 로드 완료 ({time.time() - t_db:.1f}s)")

        t_re = time.time()
        st.write("⚡ CrossEncoder 리랭커 로드 중...")
        reranker = load_reranker()
        st.write(f"✔️ CrossEncoder 로드 완료 ({time.time() - t_re:.1f}s)")

        t_bm = time.time()
        st.write("🚀 Kiwi + BM25 인덱스 구축 중 (최초 1회)...")
        bm25_retriever, kiwi = build_bm25_retriever(db_path)
        st.write(f"✔️ BM25 인덱스 구축 완료 ({time.time() - t_bm:.1f}s)")

        status.update(
            label=f"✅ 리소스 준비 완료! (전체 {time.time() - t_init:.1f}s)",
            state="complete",
            expanded=False,
        )
    st.session_state._initialized = True
else:
    embeddings     = load_embeddings()
    vector_db      = load_vector_db(db_path)
    reranker       = load_reranker()
    bm25_retriever, kiwi = build_bm25_retriever(db_path)

# ──────────────────────────────────────────────────────────────────────
# API 키 검증
# ──────────────────────────────────────────────────────────────────────
if not openai_api_key:
    st.error("❌ .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.", icon="🔑")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# 채팅 히스토리 초기화
# ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "verdict" in msg:
            is_pass   = msg["verdict"]
            cor_rounds = msg.get("correction_rounds", 0)
            cor_txt   = f"<span class='verdict-corrected'>(교정 {cor_rounds}회 후)</span>" if cor_rounds else ""
            badge_cls = "verdict-pass" if is_pass else "verdict-fail"
            icon      = "✅" if is_pass else "⚠️"
            label     = "PASS" if is_pass else "FAIL"
            st.markdown(
                f"<span class='{badge_cls}'>{icon} 검증 {label}{cor_txt}</span>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────
# 사용자 입력 처리
# ──────────────────────────────────────────────────────────────────────
# 예시 질문 버튼 클릭 시 pending_question을 기본값으로 사용
default_input = st.session_state.pop("pending_question", "") if "pending_question" in st.session_state else ""

user_query = st.chat_input(
    "약에 대해 궁금한 것을 물어보세요 (예: 눈이 침침한데 뭐 먹으면 될까요?)",
    key="chat_input",
) or default_input

if user_query:
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        total_start = time.time()

        # ── 1. Query Expansion ─────────────────────────
        query_optimizer = None
        if use_query_expansion:
            with st.spinner("🪄 쿼리 확장 중..."):
                if "query_optimizer" not in st.session_state:
                    st.session_state.query_optimizer = get_query_optimizer(openai_api_key)
                if "search_keywords_cache" not in st.session_state:
                    st.session_state.search_keywords_cache = {}
                query_optimizer = st.session_state.query_optimizer
                cached_keywords = st.session_state.search_keywords_cache.get(user_query)
                cache_hit = cached_keywords is not None
                if cache_hit:
                    search_keywords = cached_keywords
                else:
                    optimize_prompt = (
                        f"다음 질문에서 약학 검색에 필요한 핵심 성분명, 증상, 질환 키워드만 뽑아 공백으로 나열해줘: {user_query}"
                    )
                    try:
                        search_keywords = query_optimizer.invoke(optimize_prompt)
                    except Exception:
                        search_keywords = user_query
                    st.session_state.search_keywords_cache[user_query] = search_keywords
        else:
            search_keywords = None

        # ── 2. 앙상블 검색 ─────────────────────────────
        with st.spinner(f"🔍 앙상블 검색 중 (BM25 {weight_bm25} / 벡터 {weight_vector})..."):
            search_start = time.time()
            ensemble_docs = get_ensemble_results(
                query=user_query,
                kiwi=kiwi,
                bm25_retriever=bm25_retriever,
                vector_db=vector_db,
                query_optimizer=query_optimizer,
                search_keywords=search_keywords,
                k=ensemble_k,
                weight_bm25=weight_bm25,
                weight_vector=weight_vector,
            )
            search_elapsed = time.time() - search_start

        # ── 3. CrossEncoder 리랭킹 ──────────────────────
        with st.spinner(f"⚡ {len(ensemble_docs)}개 문서 리랭킹 중..."):
            rerank_start = time.time()
            ranked_pairs: list[tuple[float, object]] = rerank_docs(
                query=user_query,
                docs=ensemble_docs,
                reranker=reranker,
                top_k=top_k,
                batch_size=32,
            )
            rerank_elapsed = time.time() - rerank_start

        # 점수·문서 분리
        rerank_scores = [s for s, _ in ranked_pairs]
        final_docs    = [d for _, d in ranked_pairs]
        max_score     = max(rerank_scores) if rerank_scores else 1.0

        # ── 4. 컨텍스트 구성 + 답변 생성 (스트리밍) ──────────────
        context_text = build_context(final_docs)

        gen_start = time.time()
        stream_iter = asyncio.run(generate_answer(
            query=user_query,
            context_text=context_text,
            openai_api_key=openai_api_key,
            model=gen_model,
            stream=True,
        ))
        # 토큰 실시간 렌더링
        final_answer = st.write_stream(stream_iter)
        gen_elapsed = time.time() - gen_start

        # ── 5. 검증 ────────────────────────────────────
        with st.spinner("🧐 답변 검증 중..."):
            verify_start = time.time()
            verify_result = verify_answer(
                query=user_query,
                context_text=context_text,
                answer=final_answer,
                openai_api_key=openai_api_key,
                model="gpt-5.2",
            )
            verify_elapsed = time.time() - verify_start

        # ── 6. 자기 교정 루프 ───────────────────────────
        correction_rounds = 0
        correction_logs: list[dict] = []
        if use_self_correction and "FAIL" in verify_result.upper():
            with st.spinner("🔄 자기 교정 루프 실행 중 (최대 3회)..."):
                async def _run_correction():
                    res = {}
                    async for etype, val in self_correction_loop(
                        query=user_query,
                        context_text=context_text,
                        initial_answer=final_answer,
                        initial_verify_result=verify_result,
                        openai_api_key=openai_api_key,
                        gen_model=gen_model,
                        max_rounds=3,
                    ):
                        if etype == "done_loop":
                            res = val
                    return res
                
                loop_res = asyncio.run(_run_correction())
                if loop_res:
                    final_answer = loop_res["answer"]
                    verify_result = loop_res["verify_result"]
                    correction_rounds = loop_res["rounds"]
                    correction_logs = loop_res["logs"]

        # ── 답변 표시 ───────────────────────────────────
        st.markdown(final_answer)

        # PASS / FAIL 배지
        is_passed   = "PASS" in verify_result.upper()
        badge_cls   = "verdict-pass" if is_passed else "verdict-fail"
        icon        = "✅" if is_passed else "⚠️"
        label       = "PASS" if is_passed else "FAIL"
        cor_txt     = (
            f"<span class='verdict-corrected'>(교정 {correction_rounds}회 후)</span>"
            if correction_rounds > 0 else ""
        )
        st.markdown(
            f"<span class='{badge_cls}'>{icon} 검증 {label}{cor_txt}</span>",
            unsafe_allow_html=True,
        )

        total_elapsed = time.time() - total_start

        # ── 탭 UI ───────────────────────────────────────
        tab_docs, tab_perf, tab_log, tab_ragas = st.tabs([
            f"📄 참고 문서 ({len(final_docs)})",
            "📊 성능 지표",
            f"🔄 교정 로그 ({max(correction_rounds, len(correction_logs))})",
            "🧪 RAGAS",
        ])

        # ── 탭1: 참고 문서 ──────────────────────────────
        with tab_docs:
            for i, (score, doc) in enumerate(zip(rerank_scores, final_docs), 1):
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                content_preview = doc.page_content.replace("passage: ", "").replace("\n", " ")
                pct_score = min(score / max(max_score, 1e-6), 1.0) if max_score > 0 else 0.0
                pct_score = max(pct_score, 0.0)

                st.markdown(
                    f"**[{i}]** "
                    f"<span class='source-badge'>{source}</span>"
                    f"<span class='rank-badge'>점수 {score:.3f}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(float(pct_score), text=f"관련도 {pct_score*100:.1f}%")
                st.caption(content_preview[:280] + "..." if len(content_preview) > 280 else content_preview)
                if i < len(final_docs):
                    st.divider()

        # ── 탭2: 성능 지표 ──────────────────────────────
        with tab_perf:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔍 검색", f"{search_elapsed:.2f}s",
                      help=f"앙상블 {len(ensemble_docs)}개 → {len(final_docs)}개")
            c2.metric("⚡ 리랭킹", f"{rerank_elapsed:.2f}s",
                      help=f"CrossEncoder {len(ensemble_docs)}개 처리")
            c3.metric("✍️ 생성", f"{gen_elapsed:.2f}s")
            c4.metric("🧐 검증", f"{verify_elapsed:.2f}s")
            st.metric("⏱️ 전체", f"{total_elapsed:.2f}s")

            st.divider()
            st.markdown("**검증 상세:**")
            st.text(verify_result)

        # ── 탭3: 교정 로그 ──────────────────────────────
        with tab_log:
            if not correction_logs:
                st.info("✅ 자기 교정 없이 PASS 판정을 받았습니다." if is_passed
                        else "교정 루프를 실행하지 않았습니다. (사이드바에서 활성화)")
            else:
                for log in correction_logs:
                    rno = log["round"]
                    r_pass = "PASS" in log["verify_result"].upper()
                    r_icon = "✅" if r_pass else "🔁" if rno > 0 else "1️⃣"
                    lbl_class = "round-label-pass" if r_pass else "round-label-fail"
                    with st.expander(
                        f"{'ROUND ' + str(rno) if rno > 0 else 'ROUND 0 (초기)'}  ·  {'PASS' if r_pass else 'FAIL'}",
                        expanded=(rno == correction_rounds),
                    ):
                        st.markdown("**답변:**")
                        st.markdown(log["answer"])
                        st.markdown("**검증 결과:**")
                        st.text(log["verify_result"])

        # ── 탭4: RAGAS ──────────────────────────────────
        with tab_ragas:
            if not use_ragas:
                st.info("📊 사이드바에서 **RAGAS 평가**를 켜면 faithfulness·answer_relevancy 점수를 확인할 수 있습니다.")
            else:
                with st.spinner("🧪 RAGAS 평가 중 (추가 API 호출)..."):
                    ragas_scores = evaluate_with_ragas(
                        query=user_query,
                        answer=final_answer,
                        final_docs=final_docs,
                        embeddings=embeddings,
                        openai_api_key=openai_api_key,
                        eval_model=gen_model,
                    )
                if "error" in ragas_scores:
                    st.error(f"평가 오류: {ragas_scores['error']}")
                else:
                    r1, r2 = st.columns(2)
                    faith = ragas_scores.get("faithfulness", 0.0)
                    relev = ragas_scores.get("answer_relevancy", 0.0)
                    r1.metric("✅ Faithfulness", f"{faith:.4f}",
                              help="답변이 문서에 얼마나 근거하는지 (1.0 최고)")
                    r2.metric("🎯 Answer Relevancy", f"{relev:.4f}",
                              help="답변이 질문과 얼마나 관련 있는지 (1.0 최고)")
                    st.progress(float(faith), text=f"Faithfulness {faith*100:.1f}%")
                    st.progress(float(relev), text=f"Answer Relevancy {relev*100:.1f}%")

        # GPU 메모리 정리
        clear_gpu()

        # ── 히스토리 저장 ───────────────────────────────
        doc_snapshots = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content.replace("passage: ", ""),
            }
            for doc in final_docs
        ]
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "verdict": is_passed,
            "correction_rounds": correction_rounds,
            "docs": doc_snapshots,
        })
