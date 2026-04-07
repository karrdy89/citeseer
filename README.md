# citeseer

S2(Semantic Scholar) API 기반 탑 학회 논문 수집 + Elasticsearch 하이브리드 검색 + MCP 서버.

## 대상 학회

NeurIPS, ICLR, ICML, ACL, EMNLP, NAACL, CVPR, AAAI, CHI

## 아키텍처

| 서비스 | 역할 | 실행 |
|--------|------|------|
| **ES** | 데이터 저장/검색 | `elasticsearch:8.17.0` |
| **SPECTER2** | 임베딩 모델 서빙 (`/encode`) | `specter/` FastAPI |
| **App** | 수집 + 검색 + MCP 도구 | `app/` 패키지 |

```
app/
├── core/          # config (pydantic-settings), logging
├── es/            # mapping, client (bulk upsert/scroll), search (BM25+knn 하이브리드)
├── specter/       # SPECTER2 임베딩 클라이언트
├── s2/            # S2 API fetcher (retry+prefetch), models, PDF 다운로드
├── ingest/        # 수집 파이프라인 (체크포인트, 배치 윈도우, 스케줄링)
└── mcp/           # FastMCP search_papers 도구
```

## 설정

`.env` 파일 또는 환경변수로 설정. `.env.example` 참고.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `S2_API_KEY` | (없음) | S2 API 키 (없으면 rate limit 적용) |
| `YEAR_FROM` | `2020` | 수집 시작 (YYYY 또는 YYYY-MM-DD) |
| `YEAR_TO` | (오늘) | 수집 끝 (미설정 시 오늘 날짜) |
| `INGEST_WINDOW_DAYS` | `0` | 1회 수집 날짜 범위 (일). 0=전체 |
| `INGEST_CRON` | (없음) | cron 스케줄 (예: `0 */3 * * *`) |
| `INGEST_INTERVAL` | `0` | 수집 반복 간격 (초). cron 우선 |
| `MAX_RETRIES` | `3` | 재시도 횟수 (S2, PDF, SPECTER2, ES 공통) |
| `S2_TIMEOUT` | `30` | S2 API 타임아웃 (초) |
| `SPECTER_TIMEOUT` | `120` | SPECTER2 타임아웃 (초) |
| `PDF_TIMEOUT` | `60` | PDF 다운로드 타임아웃 (초) |
| `ES_TIMEOUT` | `60` | Elasticsearch 타임아웃 (초) |

## 실행

### Docker Compose (권장)

```bash
docker compose up --build
```

기본 설정에서 앱은 MCP 서버로 실행됩니다.
수집 파이프라인은 별도로 실행:

```bash
docker compose run app python -m app.ingest.pipeline
```

### 로컬 개발

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows/Git Bash
pip install -e ".[dev]"

# 수집
python -m app.ingest.pipeline

# MCP 서버
python -m app.mcp.server

# 테스트
pytest tests/ -m "not integration" -v
```

## 검색

하이브리드 검색: BM25 (boost 0.3) + knn cosine (boost 0.7), 768d SPECTER2 벡터.

MCP 도구 `search_papers_tool`로 접근:
- `query`: 검색 쿼리
- `venues`: 학회 필터 (optional)
- `date_from` / `date_to`: 날짜 범위 (YYYY-MM-DD 또는 YYYY)
- `limit`: 결과 수

## 수집 파이프라인

1. S2 bulk API에서 논문 메타데이터 + 초록 fetch (publicationDate desc)
2. SPECTER2로 임베딩 생성 + OA PDF 병렬 다운로드
3. Elasticsearch에 bulk upsert
4. 페이지 단위 체크포인트 저장 → 중단 시 자동 resume

### 배치 윈도우 + 스케줄링 예시

```env
YEAR_FROM=2020
INGEST_WINDOW_DAYS=365
INGEST_CRON="0 */3 * * *"
```

3시간마다 1년치씩 수집. 현재 날짜 도달 후에는 새 논문만 추가.
