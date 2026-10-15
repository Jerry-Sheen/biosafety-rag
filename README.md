# 🧬 Biosafety RAG: 병원체 규제 및 생물안전 법령 안내 대화형 RAG 시스템

> **국내 병원체 및 생물작용제 관련 사내·연구 규정 문서**(PDF, HWP/HWPX, Word, Excel, CSV, JSON 등)와 **법제처 국가법령정보센터(law.go.kr)의 최신 법령 데이터**를 결합하여, 특정 병원균에 적용되는 국내외 법령 규제 및 **8대 법정 의무 신고·허가 절차**를 체계적으로 안내하는 AI 기반 듀얼 검색 RAG 시스템입니다.
> 
> 본 시스템은 **관리자(`admin.py`)**와 **사용자(`user.py`)**가 완전히 독립된 포털로 분리되어 있어 각각 독립적으로 실행 및 운영할 수 있습니다.

---

## 🌟 핵심 특징 및 시스템 구조

### 1. 병원체 규제 분류 및 법제처 듀얼 검색 (Dual Retrieval)
- **1단계: 영구 벡터 DB(ChromaDB) 기반 규제 분류 제시**
  - 사용자가 특정 병원균(예: *콜레라균, 탄저균, 페스트균* 등)을 질의하면, 사내 영구 벡터 DB에 구축된 병원체 분류 데이터를 검색하여 주요 법률별 해당 여부를 직관적으로 표기합니다.
  - 가독성을 극대화하여 **`해당`** 또는 **`해당사항 없음`**으로 명확히 구분하여 제시합니다.
    - *감염병예방법(고위험병원체)*
    - *감염병예방법(생물테러감염병병원체)*
    - *유전자변형생물체법(국가관리대상병원체)*
    - *화학생물무기금지법(생물작용제)*
    - *가축전염병예방법(제1종/제2종/제3종)*
    - *수의생명자원관리규정(특별관리병원체)*
    - *생물체위험군 분류(제1~4위험군)*
- **2단계: 국가법령정보센터(law.go.kr) 최신 법령 실시간 검색 및 8대 의무 항목 매핑**
  - 법제처 Open DRF API 연동 모듈(`src/rag/law_search.py`)을 통해 적용 법령의 최신 조항을 실시간 검색·대조합니다.
  - 해당 특정 병원균이 적용되는 각 법률별로 아래 **8대 법정 의무 신고·허가 항목**을 빠짐없이 구조화하여 안내합니다:
    1. **분리신고**
    2. **이동(분양)신고**
    3. **보유·폐기신고**
    4. **제조신고 및 개발실험승인**
    5. **수입(반입)신고**
    6. **인수신고**
    7. **수출신고**
    8. **인도신고**
    *(※ 해당 법률에 규정이 없는 항목은 근거와 함께 '해당없음'으로 명시)*
- **3단계: 실무 가이드 및 온라인 전자민원 포털 안내**
  - 질병관리청(질병보건통합관리시스템), 산업통상자원부(전략물자관리시스템 등), 농림축산검역본부 등 소관 부처별 온라인 신고 포털 URL 및 단계별 실무 절차(Step 1~Step 3)를 구체적으로 제공합니다.

---

### 2. Admin과 User의 완전한 파일 분리 (독립 포털)
- **`admin.py` (관리자 전용 데이터 관리 포털)**:
  - 기본 포트: `8501` (`streamlit run admin.py`)
  - PDF, HWP, HWPX, DOCX, XLSX, CSV, JSON 등 다양한 형식의 규정 문서 일괄 업로드.
  - **Docling 구조화 파서 및 RapidOCR/Tesseract**를 통한 스캔 문서 광학 문자 인식.
  - 청크 크기(Chunk Size) 및 중복(Overlap) 조절 지원.
  - **영구 증분 저장(Incremental Ingestion)**: 기존 벡터 데이터를 보존하며 신규 규정 추가 색인.
  - 등록 문서 목록 모니터링, 특정 문서 선택 삭제 및 DB 전체 초기화 기능.
- **`user.py` (사용자 규제 안내 및 대화형 포털)**:
  - 기본 포트: `8502` (`streamlit run user.py`)
  - 병원균 검색 및 규제 안내 실시간 스트리밍(Streaming) 답변 출력.
  - **멀티 세션 대화 관리(Chat Session Management)**: 사이드바에서 이전 대화 목록 저장, 불러오기, 완전 삭제 지원.
  - **중복 세션 생성 방지**: 페이지 새로고침 시 무분별하게 세션이 추가되지 않고 기존 활성 세션을 유지.
  - **출처 투명성(Citations)**: 참고한 법제처 법령 링크, 내부 규정 문서명 및 페이지/청크 정보 제공.

---

### 3. 엔터프라이즈 환경 및 네트워크 호환성
- **Windows 사내망 SSL 자동 호환**: 사내 프록시나 자체 서명 인증서 환경에서 발생하는 SSL 핸드셰이크 오류를 방지하기 위해 `truststore`를 시스템 SSL에 자동 주입합니다.
- **유연한 LLM 및 임베딩 모델 지원**:
  - **LLM**: Google Gemini API (`gemini-2.5-flash`, `gemini-1.5-flash`), OpenAI API (`gpt-4o-mini` 등)
  - **임베딩**: 고성능 다국어 임베딩 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **보안 및 개인정보 보호**:
  - API 키 및 시스템 설정은 `.env` 파일을 통해 안전하게 주입받으며, `.gitignore`를 통해 Git 유출을 원천 방지합니다.
  - 사용자별 대화 세션 기록(`data/chat_sessions/`) 및 로컬 벡터 데이터(`data/chroma_db/`)는 버전 관리에서 자동 격리됩니다.

---

## 📂 프로젝트 구조

```
Rag_vibe/
├── admin.py                  # 🛠️ [Admin] 관리자 문서 업로드/파싱/색인/DB 관리 독립 앱
├── user.py                   # 💬 [User] 사용자 대화형 병원체 규제 안내 독립 앱
├── config.py                 # ⚙️ 중앙 애플리케이션 설정 (Pydantic BaseSettings)
├── .env.example              # 환경 변수 설정 템플릿
├── .gitignore                # Git 제외 규칙 (보안 키, DB 데이터, 대화 기록 격리)
├── Dockerfile                # 운영 환경용 컨테이너 빌드 파일
├── docker-compose.yml        # admin-service(8501) & user-service(8502) 서비스 구성
├── requirements.txt          # 파이썬 의존성 패키지 목록
├── README.md                 # 프로젝트 상세 가이드
├── parsers/                  # 모듈형 문서 파서 패키지 (OCR, PDF, HWP, Markdown 등)
├── pipeline/                 # 문서 수집, 중복 제거, 품질 평가 및 청킹 파이프라인
├── vectorstore/              # 벡터 스토어 통합 인터페이스
├── src/
│   ├── config.py             # 설정 호환성 브릿지
│   ├── security/
│   │   ├── secrets_manager.py# 환경별 비밀 관리 (.env 및 환경변수)
│   │   └── logger.py         # 보안 로그 마스킹 필터
│   ├── parsers/
│   │   ├── document_parser.py# 통합 마스터 파서 (Docling + HWP + Fallback 라우팅)
│   │   ├── docling_parser.py # IBM Docling 구조화 파서 및 OCR 연동
│   │   ├── hwp_parser.py     # HWP / HWPX 전용 파서
│   │   ├── csv_parser.py     # CSV 전용 파서
│   │   └── fallback_parser.py# 일반 텍스트 및 기본 문서 복원력 파서
│   ├── rag/
│   │   ├── law_search.py     # ⚖️ 법제처(law.go.kr) Open DRF API 연동 및 8대 의무 항목 매핑
│   │   ├── engine.py         # 🧠 2단계 듀얼 검색 대화형 RAG 스트리밍 엔진
│   │   ├── vector_store.py   # ChromaDB 영구 저장소 (증분 추가/검색/삭제)
│   │   ├── embeddings.py     # Sentence-Transformers 다국어 임베딩
│   │   ├── gemini_client.py  # Google Gemini 클라우드 API 클라이언트
│   │   └── openai_client.py  # OpenAI API 클라이언트
│   └── ui/
│       ├── admin_view.py     # Admin UI 컴포넌트
│       └── user_view.py      # User UI 컴포넌트
├── tests/
│   ├── run_tests.py          # 경량 단위 테스트 실행기
│   ├── test_vector_store.py  # ChromaDB 증분 저장 및 검색 테스트
│   ├── test_chunker.py       # 구조 보존형 청킹 테스트
│   ├── test_parsers.py       # 다중 포맷 파서 테스트
│   ├── test_secrets.py       # 보안 및 마스킹 검증 테스트
│   └── test_pipeline_refactor.py # 파이프라인 무결성 테스트
└── data/
    └── input/
        └── sample_biosafety_guide.md # 테스트용 샘플 생물안전 가이드라인
```

---

## 🚀 빠른 시작 가이드

### 1. 환경 설정 및 패키지 설치

1. 저장소를 클론하고 프로젝트 디렉토리로 이동합니다:
   ```bash
   git clone https://github.com/Jerry-Sheen/biosafety-rag.git
   cd biosafety-rag
   ```

2. `.env.example`을 복사하여 `.env` 파일을 생성하고 필요한 API 키를 입력합니다:
   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env
   # Linux / macOS
   cp .env.example .env
   ```

3. `.env` 파일 내 API 키 및 환경 변수를 설정합니다:
   ```ini
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ADMIN_ACCESS_PIN=admin1234
   ```

4. 파이썬 가상환경을 활성화하고 의존성 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

5. 단위 테스트를 실행하여 환경 무결성을 검증합니다:
   ```bash
   python tests/run_tests.py
   ```

---

### 2. 애플리케이션 실행

#### 🛠️ 관리자 포털 실행 (`admin.py`)
```bash
streamlit run admin.py --server.port=8501
```
- 브라우저 접속: `http://localhost:8501`
- 기본 관리자 PIN: `admin1234` (`.env`의 `ADMIN_ACCESS_PIN`에서 수정 가능)
- 생물안전 규정 문서(PDF, HWP, HWPX, JSON 등)를 업로드하여 영구 벡터 DB에 색인합니다.

#### 💬 사용자 규제 안내 포털 실행 (`user.py`)
```bash
streamlit run user.py --server.port=8502
```
- 브라우저 접속: `http://localhost:8502`
- 검색창에 병원균명(예: `콜레라균`, `탄저균`)을 입력하면:
  1. 영구 벡터 DB 매핑 결과 (`해당` / `해당사항 없음`)
  2. 적용 법률별 8대 의무 항목 규정 상세 (분리, 이동, 보유·폐기, 제조, 수입, 인수, 수출, 인도)
  3. 부처별 온라인 신고 포털 및 실무 단계별 절차가 실시간 스트리밍으로 출력됩니다.

---

### 3. Docker 배포

`docker-compose.yml`을 이용하여 관리자(`admin-service`, 포트 8501)와 사용자(`user-service`, 포트 8502) 컨테이너를 한 번에 기동할 수 있습니다.

```bash
# 컨테이너 빌드 및 백그라운드 실행
docker compose up --build -d

# 실행 상태 확인
docker compose ps
```

- **관리자 웹**: `http://localhost:8501`
- **사용자 웹**: `http://localhost:8502`
- 두 서비스는 공유 볼륨을 통해 영구 벡터 DB 데이터를 실시간으로 동기화합니다.

---

## 📄 License

**Copyright (c) 2026 신용환 (Yonghwan Shin). All rights reserved.**

본 프로젝트의 소스코드 및 관련 모든 문서는 **독점 소프트웨어(Proprietary Software)**입니다.  
오직 단순 열람(보기) 및 참조 목적으로만 제공되며, 저작권자의 사전 서면 허가 없이 무단 복제, 수정, 편집, 배포, 2차적 저작물 작성 및 상업적 이용을 엄격히 금지합니다. 자세한 내용은 [`LICENSE`](LICENSE) 파일을 참조하십시오.

