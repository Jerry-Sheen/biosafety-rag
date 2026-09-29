# 🧠 RAG Vibe: 로컬 Ollama 기반 멀티 포맷 문서 대화형 RAG 시스템

> **PDF, Word, Excel, CSV, HWP/HWPX** 등 다양한 사내·개인 문서를 분석하고, 최신 **Docling** 구조화 파서 및 **OCR** 기술을 결합하여 영구 벡터 데이터베이스에 저장한 뒤, **로컬 Ollama 모델(EXAONE 3.5 / BGE-M3)**을 통해 외부 API 키 없이 100% 로컬에서 실시간 스트리밍 대화형 답변을 제공하는 RAG 시스템입니다.
> 
> 본 시스템은 **관리자(`admin.py`)**와 **사용자(`user.py`)**가 **완전히 별도의 독립 파일**로 분리되어 있어 각각 독립적으로 실행 및 배포할 수 있습니다.

---

## 🌟 핵심 특징 및 시스템 구조

### 1. Admin과 User의 완전한 파일 분리 (독립 실행 가능)
- **`admin.py` (관리자 전용 웹 포털)**:
  - 포트: 기본 `8501` (`streamlit run admin.py`)
  - PDF, Word, Excel, CSV, HWP/HWPX 다중 문서 일괄 업로드.
  - Docling 구조화 파싱 및 이미지 PDF OCR 적용.
  - 청크 크기(Chunk Size) 및 중복(Overlap) 조절.
  - **영구 증분 저장(Incremental Ingestion)**: 기존 벡터 데이터를 보존하며 새로운 문서의 임베딩을 계속 추가.
  - 등록 문서 목록 모니터링, 특정 문서 선택 삭제 및 DB 전체 초기화.
- **`user.py` (사용자 대화 질의응답 전용 포털)**:
  - 포트: 기본 `8502` (`streamlit run user.py`)
  - 사용자 자연어 질의 입력 -> 영구 벡터 DB(ChromaDB) 밀집 검색.
  - 로컬 Ollama LLM을 통한 답변 실시간 스트리밍(Streaming) 출력.
  - **멀티턴 대화 기억(Conversational Memory)**: 이전 대화 문맥을 파악하여 후속 질문 재구성.
  - **출처 투명성(Citations)**: 참고한 원문 문서명, 유사도 점수, 실제 발췌 문맥 공개.

### 2. 100% 로컬 무료 모델 (Ollama 기반 / 한국어 최적화)
- **LLM 모델**: **LG EXAONE 3.5 (`exaone3.5`)** 또는 **Qwen 2.5 (`qwen2.5:7b`)**
  - 대규모 한국어 코퍼스로 사전 학습된 최고 수준의 한국어 추론 및 대화 능력.
  - 외부 상용 API 키(Gemini, OpenAI 등) 없이 로컬 GPU/CPU에서 완전 무료 실행.
- **임베딩 모델**: **BAAI BGE-M3 (`bge-m3`)**
  - 한국어를 포함한 다국어 시맨틱 검색 SOTA 임베딩 모델 (1024 차원 고밀도 벡터).
- **영구 벡터 데이터베이스**: **ChromaDB (PersistentClient)**
  - 로컬 디스크 및 컨테이너 볼륨에 저장되어 재기동 후에도 데이터가 영구 보존됩니다.

### 3. 엔터프라이즈 보안 및 격리
- **비밀 정보 보호**: `.env` 파일과 환경 변수를 통해서만 설정값을 주입받으며, [`.gitignore`](file:///C:/Users/USER/Desktop/Rag_vibe/.gitignore)를 통해 Git 유출을 원천 차단.
- **로그 마스킹 필터**: API 토큰, 패스워드 패턴을 자동 감지하여 콘솔 및 로그 파일에 `[REDACTED_SECRET]`으로 마스킹.
- **컨테이너 보안**: `Dockerfile` 내 `appuser` (Non-root) 계정 실행 및 User 서비스의 ChromaDB 읽기 전용(`ro`) 마운트.

---

## 📂 프로젝트 구조

```
Rag_vibe/
├── admin.py                  # 🛠️ [Admin] 관리자 문서 업로드/색인/DB 관리 독립 앱
├── user.py                   # 💬 [User] 사용자 대화형 질의응답 독립 앱
├── app.py                    # 🧭 통합 런처 (admin과 user를 한 화면에서 선택 실행 가능)
├── .env.example              # 환경 변수 설정 템플릿
├── .gitignore                # Git 제외 규칙 (보안 키, DB 데이터, 캐시 격리)
├── .dockerignore             # Docker 빌드 제외 규칙
├── Dockerfile                # 운영 환경용 보안 Dockerfile (Non-root, OCR 포함)
├── docker-compose.yml        # admin-service(8501) & user-service(8502) 독립 컨테이너 구성
├── requirements.txt          # 파이썬 의존성 패키지 목록
├── README.md                 # 프로젝트 상세 가이드
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD 파이프라인
├── src/
│   ├── config.py             # 중앙 설정 (Pydantic 기반)
│   ├── security/
│   │   ├── secrets_manager.py# 환경별 비밀 관리 (.env, Docker Secrets, AWS Secrets)
│   │   └── logger.py         # 보안 로그 마스킹 필터
│   ├── parsers/
│   │   ├── document_parser.py# 통합 마스터 파서 (Docling + HWP + Fallback 자동 라우팅)
│   │   ├── docling_parser.py # IBM Docling 구조화 분석 및 OCR 파서
│   │   ├── hwp_parser.py     # HWP 및 HWPX 전용 파서
│   │   └── fallback_parser.py# PDF, DOCX, XLSX, CSV 복원력 파서
│   ├── rag/
│   │   ├── ollama_client.py  # 로컬 Ollama HTTP 통신 클라이언트 (스트리밍, 임베딩)
│   │   ├── chunker.py        # 한국어 문장/단락 및 표 구조 보존 지능형 청커
│   │   ├── embeddings.py     # Ollama bge-m3 로컬 임베딩 프로바이더
│   │   ├── vector_store.py   # ChromaDB Persistent 스토어 (증분 추가/검색/삭제)
│   │   └── engine.py         # 로컬 대화형 RAG 및 스트리밍 엔진
│   └── ui/
│       ├── admin_view.py     # Admin UI 컴포넌트
│       └── user_view.py      # User UI 컴포넌트
└── tests/
    ├── run_tests.py          # 경량 단위 테스트 실행기
    ├── test_secrets.py       # 보안 및 마스킹 검증 테스트
    ├── test_chunker.py       # 한국어 청킹 검증 테스트
    ├── test_parsers.py       # 문서 파서 검증 테스트
    └── test_vector_store.py  # ChromaDB 증분 저장 검증 테스트
```

---

## 🚀 빠른 시작 가이드

### 1. 로컬 Ollama 준비 (사전 작업)

1. [Ollama 공식 홈페이지](https://ollama.com/)에서 Ollama를 설치합니다.
2. 터미널에서 한국어 추천 모델과 임베딩 모델을 내려받습니다:
   ```bash
   # 한국어 추천 LLM 다운로드
   ollama pull exaone3.5
   # 한국어 고성능 임베딩 모델 다운로드
   ollama pull bge-m3

   # Ollama 서비스 실행 (백그라운드 미실행 시)
   ollama serve
   ```

---

### 2. 환경 설정 및 패키지 설치

1. `.env.example`을 복사하여 `.env` 파일을 생성합니다:
   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env
   # Linux / macOS
   cp .env.example .env
   ```

2. 가상환경 생성 및 의존성 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

3. 기본 단위 테스트를 실행하여 정상 동작을 확인합니다:
   ```bash
   python tests/run_tests.py
   ```

---

### 3. 애플리케이션 실행

#### 🛠️ 관리자 포털 실행 (`admin.py`)
```bash
streamlit run admin.py --server.port=8501
```
- 브라우저 접속: `http://localhost:8501`
- 초기 관리자 PIN 번호: `admin1234` (`.env`의 `ADMIN_ACCESS_PIN`에서 변경 가능)
- 문서(PDF, DOCX, XLSX, CSV, HWP, HWPX)를 업로드하여 영구 벡터 DB에 저장합니다.

#### 💬 사용자 대화 포털 실행 (`user.py`)
```bash
streamlit run user.py --server.port=8502
```
- 브라우저 접속: `http://localhost:8502`
- 사용자는 `admin.py`에서 저장한 문서에 대해 실시간 스트리밍 대화를 나눌 수 있습니다.

> 💡 두 화면을 하나의 런처에서 탭으로 전환하여 보고 싶다면 `streamlit run app.py`를 실행할 수도 있습니다.

---

### 4. Docker Compose로 일괄 배포

`docker-compose.yml`에는 관리자(`admin-service`, 포트 8501)와 사용자(`user-service`, 포트 8502) 컨테이너가 분리되어 구성되어 있습니다.

```bash
# 컨테이너 빌드 및 백그라운드 시작
docker compose up --build -d

# 실행 상태 확인
docker compose ps
```

- **관리자 웹**: `http://localhost:8501`
- **사용자 웹**: `http://localhost:8502`
- 두 컨테이너는 공통의 `chroma_data` 네임드 볼륨을 공유하므로, 관리자가 문서를 등록하는 즉시 사용자가 질문할 수 있습니다.

---

## 🔄 CI/CD 파이프라인 (GitHub Actions)

`.github/workflows/ci.yml` 파일에 구성된 자동화 워크플로우:
1. **Lint & Code Quality**: `ruff`를 통해 정적 코드 분석 수행.
2. **Automated Unit Testing**: 청커, 파서, 시크릿 매니저, 벡터 스토어 테스트 자동 실행.
3. **Docker Build Verification**: Docker 이미지 빌드 무결성 자동 검증.
