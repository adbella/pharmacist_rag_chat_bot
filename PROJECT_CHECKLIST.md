# 3차 프로젝트 제출 체크리스트 (실행/발표용)

아래 항목을 순서대로 채우면 평가 기준 대응이 쉬워집니다.

## 1) PRD (1페이지)
- [ ] 문제 정의 / 목표 / 타깃 사용자
- [ ] 핵심 기능 (RAG, 검증, 개선 루프)
- [ ] 성공 기준 (예: 실패 케이스 개선 수, RAGAS 상승)

## 2) 시스템 구조도
- [ ] 데이터 로드 → 청킹/임베딩 → Chroma 저장/로드
- [ ] Retrieval(BM25+Vector/MMR) → 리랭커 → 생성 → 검증 → 개선 루프
- [ ] UI(SSE)에서 근거/지표/로그 표시 흐름

## 3) 데이터 소개
- [ ] 문서 종류/출처/개수 정리 (최소 5개 이상)
- [ ] DB 통계 (총 chunk 수, source 수)

## 4) Retrieval/리랭커 구성 및 근거
- [ ] 임베딩 후보 2개 이상 비교
- [ ] 리랭커 사용/미사용(또는 후보 비교)
- [ ] 선택 기준 2가지 이상 (속도/정확도/비용/도메인 적합성)
- [ ] 전/후 수치 1개 이상 제시

## 5) 피드백 프로세스 (필수)
- [ ] 테스트 질문 세트 준비 (in-scope + out-of-scope 포함)
- [ ] 실패 케이스 3개 선정
- [ ] 각 케이스 ROUND 1/2 로그 확보:
  - [ ] (a) retrieval 결과
  - [ ] (b) rerank 결과
  - [ ] (c) 1차 답변
  - [ ] (d) ROUND1 검증(PASS/FAIL + RAGAS)
  - [ ] (e) ROUND2 답변
  - [ ] (f) ROUND2 검증(PASS/FAIL + RAGAS)

## 6) 재현성
- [ ] .env 사용 (키 하드코딩 금지)
- [ ] Chroma persist/load 확인
- [ ] 실행 명령 문서화

## 7) 현재 코드에서 바로 실행 가능한 명령

### 서버 실행
```bash
python api.py
```

### 평가 로그 자동 생성 (발표용)
```bash
python eval_runner.py --cases eval_cases.json --oos-guard --oos-min-relevance 0.55
```

### 생성 산출물
- `outputs/eval_runs/<timestamp>/round_logs.jsonl`
- `outputs/eval_runs/<timestamp>/case_summary.csv`
- `outputs/eval_runs/<timestamp>/summary.md`
