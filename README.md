# 🧠 RAG Vibe: 멀티 포맷 문서 기반 지능형 대화 RAG 시스템

> **PDF, Word, Excel, CSV, HWP/HWPX** 등 다양한 사내·개인 문서를 분석하고, 최신 **Docling** 구조화 파서 및 **OCR** 기술을 결합하여 영구 벡터 데이터베이스에 저장한 뒤, **Google Gemini API**(무료 티어 / 한국어 최고 성능)를 통해 실시간 스트리밍 대화형 답변을 제공하는 프로덕션 레디 RAG(Retrieval-Augmented Generation) 시스템입니다.

---

## 🌟 주요 핵심 기능

### 1. 다양한 문서 포맷 일괄 업로드 및 지능형 파싱
- **다중 파일 동시 업로드**: PDF, Word(`.docx`), Excel(`.xlsx`, `.xls`), CSV, 한글 문서(`.hwp`, `.hwpx`), Markdown, TXT를 한 번에 업로드 가능.
- **스캔 PDF 이미지 OCR 지원**: 텍스트 레이어가 없는 스캔 문서나 이미지형 PDF의 경우 OCR(Tesseract / Docling OCR)을 적용하여 텍스트를 정밀 추출.
- **IBM Docling 연동**: 단순 텍스트 추출을 넘어 문서의 헤더 구조, 표(Table), 캡션을 구조화된 마크다운 데이터로 변환.
- **국내 전용 HWP/HWPX 완벽 지원**:
  - `.hwpx`: XML 섹션 압축 해제를 통한 원문 및 표 텍스트 추출.
  - `.hwp`: OLE 복합 문서 스트림 압축 해제(Deflate) 및 UTF-16LE 디코딩 지원.
- **한국어 CSV/Excel 인코딩 자동 보정**: UTF-8, CP949, EUC-KR 인코딩 자동 감지.

### 2. 모델 및 벡터 데이터베이스 (무료 티어 및 한국어 최적화)
- **LLM**: Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash` 등)
  - 구글 AI 스튜디오에서 완전 무료 API 키 발급 가능.
  - 한국어 이해도 및 문맥 생성 능력이 세계 최고 수준.
  - Streamlit 환경 및 Docker 컨테이너에서 가볍게 동작.
- **임베딩 모델**: Google `text-embedding-004` (768 차원 고밀도 벡터, 다국어 및 한국어 최적화).
- **벡터 데이터베이스**: **ChromaDB (Persistent Mode)**
  - 오픈소스 무료 벡터 데이터베이스.
  - 컨테이너 및 로컬 디스크에 영구 보존(Persistent Volume 마운트 가능).

### 3. 시스템 구조 (Admin / User 완벽 분리)
- **🛠️ 관리자(Admin) 영역**:
  - 문서 업로드 및 Docling / OCR 파싱 옵션 설정.
  - 청크 크기(Chunk Size) 및 중복 크기(Overlap) 동적 제어.
  - **증분 저장(Incremental Ingestion)**: 기존 데이터를 덮어쓰지 않고 새로운 문서 임베딩을 지속적으로 추가.
  - 실시간 통계 모니터링: 총 청크 수, 문서별 청크 현황, 특정 문서 삭제 및 전체 DB 초기화 기능.
  - 관리자 PIN 인증을 통한 보안 접근 제어.
- **💬 사용자(User) 영역**:
  - 질문 입력 시 영구 벡터 DB에서 관련 문맥(Context) 밀집 검색.
  - **실시간 스트리밍(Streaming)** 출력으로 사용자 대기 시간을 최소화하고 부드러운 텍스트 렌더링 제공.
  - **멀티턴 대화형 RAG(Conversational Memory)**: 이전 질문과 답변 히스토리를 기억하고 질문을 문맥에 맞게 재구성(Query Contextualization).
  - **참고 출처 투명성(Citations)**: 답변 생성에 사용된 문서명, 유사도 점수, 실제 원문 청크를 접이식(Expander) 영역에 투명하게 공개.

### 4. 강력한 엔터프라이즈급 보안
- **비밀 정보 격리**: API 키 및 패스워드는 코드에 절대 하드코딩되지 않으며, `.env` 및 환경 변수로 관리.
- **Git 유출 방지**: `.gitignore`에 `.env`, `data/`, `chroma_db/`, 키 파일 등을 등록하여 외부 저장소 유출 원천 차단.
- **운영 환경 비밀 관리(Secrets Management)**:
  - 개발/테스트: `.env` 파일 로드.
  - 운영(Production): **Docker Secrets** (`/run/secrets/*`) 및 **AWS Secrets Manager** (`boto3` IAM Role) 지원.
- **최소 권한 원칙(Principle of Least Privilege)**: IAM Role 및 Docker 비루트(Non-root `appuser`) 계정으로 컨테이너 실행.
- **로그 마스킹 필터**: API 키 패턴(`AIza...`, `Bearer ...` 등)이 로그 파일이나 콘솔에 노출되지 않도록 자동 마스킹(`[REDACTED_SECRET]`).
- **HTTPS 지원**: Streamlit SSL 설정 및 리버스 프록시(Nginx, ALB)를 통한 TLS 암호화 통신 지원.

### 5. 배포 및 유지보수
- **Docker & Docker Compose**: 멀티 스테이지 빌드 및 볼륨 영구 보존 설정.
- **CI/CD 파이프라인**: GitHub Actions를 통한 린트(Ruff), 단위 테스트(Pytest), 빌드 무결성 자동 검증.

---

## 📂 프로젝트 구조

```
Rag_vibe/
├── .env.example              # 환경 변수 설정 템플릿
├── .gitignore                # Git 제외 규칙 (비밀키, DB 데이터, 캐시 격리)
├── .dockerignore             # Docker 빌드 제외 규칙
├── Dockerfile                # 운영 환경용 보안 Dockerfile (Non-root, OCR 포함)
├── docker-compose.yml        # 볼륨 영구 저장 및 리소스 제한 Docker Compose
├── requirements.txt          # 파이썬 의존성 패키지 목록
├── README.md                 # 프로젝트 문서
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD 파이프라인
├── src/
│   ├── config.py             # 중앙 설정 (Pydantic 검증)
│   ├── security/
│   │   ├── secrets_manager.py# 환경별 비밀 관리 (Docker Secrets, AWS Secrets Manager)
│   │   └── logger.py         # 민감정보 자동 마스킹 로거
│   ├── parsers/
│   │   ├── document_parser.py# 통합 마스터 파서 (포맷 감지 및 자동 폴백)
│   │   ├── docling_parser.py # Docling 레이아웃 분석 및 OCR 파서
│   │   ├── hwp_parser.py     # HWP / HWPX 전용 파서
│   │   └── fallback_parser.py# PDF, DOCX, XLSX, CSV 고속 복원력 파서
│   ├── rag/
│   │   ├── chunker.py        # 한국어 문장/단락 보존 지능형 청커
│   │   ├── embeddings.py     # Gemini text-embedding-004 프로바이더
│   │   ├── vector_store.py   # ChromaDB Persistent 스토어 (증분 추가/검색/삭제)
│   │   └── engine.py         # 멀티턴 대화형 RAG 및 스트리밍 엔진
│   └── ui/
│       ├── admin_view.py     # 관리자 업로드/DB 관리 뷰
│       └── user_view.py      # 사용자 채팅/스트리밍/출처 뷰
├── app.py                    # Streamlit 실행 메인 파일
└── tests/
    ├── run_tests.py          # 경량 단위 테스트 러너
    ├── test_secrets.py       # 보안 및 마스킹 검증 테스트
    ├── test_chunker.py       # 한국어 청킹 검증 테스트
    ├── test_parsers.py       # 문서 파서 검증 테스트
    └── test_vector_store.py  # ChromaDB 증분 저장 검증 테스트
```

---

## 🚀 빠른 시작 가이드

### 1. 환경 설정

1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/your-username/Rag_vibe.git
   cd Rag_vibe
   ```

2. `.env.example`을 복사하여 `.env` 파일을 생성합니다:
   ```bash
   cp .env.example .env
   ```

3. `.env` 파일을 열고 본인의 Google Gemini API 키를 입력합니다:
   ```ini
   APP_ENV=development
   GEMINI_API_KEY=AIzaSy...실제키입력...
   LLM_MODEL=gemini-2.5-flash
   EMBEDDING_MODEL=text-embedding-004
   ADMIN_ACCESS_PIN=admin1234
   ```
   > 💡 **Gemini API 키 발급**: [Google AI Studio](https://aistudio.google.com/)에서 무료로 발급받을 수 있습니다.

---

### 2. 로컬 실행 방법

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 단위 테스트 실행
python tests/run_tests.py

# 4. Streamlit 애플리케이션 시작
streamlit run app.py
```
브라우저에서 `http://localhost:8501`로 접속합니다.

---

### 3. Docker 및 Docker Compose 배포

운영 환경에서는 Docker를 통해 일관된 환경과 볼륨 영구 저장을 보장합니다.

```bash
# 1. 도커 컨테이너 빌드 및 실행
docker compose up --build -d

# 2. 컨테이너 상태 및 로그 확인
docker compose ps
docker compose logs -f

# 3. 컨테이너 중지
docker compose down
```

> **볼륨 영구 보존**: `chroma_data` 및 `upload_data` 네임드 볼륨이 설정되어 있어, 컨테이너를 재시작하거나 재빌드해도 기존에 임베딩된 벡터 DB 데이터가 영구 보존됩니다.

---

## 🔒 엔터프라이즈 보안 아키텍처

### 1. 운영 환경 비밀 관리 서비스 (AWS Secrets Manager 연동)
운영 환경(`APP_ENV=production`)에서는 `.env` 파일 대신 클라우드 비밀 관리 서비스를 사용할 수 있습니다.
- AWS ECS/EKS 환경에서는 인스턴스/파드에 연결된 **IAM Role**을 통해 액세스 키 노출 없이 `boto3`가 자동으로 Secrets Manager에서 API 키를 조회합니다.
- `.env`에 설정:
  ```ini
  APP_ENV=production
  AWS_REGION=ap-northeast-2
  AWS_SECRET_NAME=prod/rag_vibe/api_keys
  ```

### 2. 최소 권한 원칙 (Least Privilege)
- **AWS IAM Role 정책 예시**:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "secretsmanager:GetSecretValue"
        ],
        "Resource": "arn:aws:secretsmanager:ap-northeast-2:*:secret:prod/rag_vibe/*"
      }
    ]
  }
  ```
- **Docker 보안**: `Dockerfile` 내 `appuser` (UID: 1000) 비루트 계정을 생성하여 컨테이너 탈취 공격을 방어합니다.

### 3. HTTPS / TLS 통신 설정
Streamlit 자체 SSL 활성화:
```bash
streamlit run app.py \
  --server.sslCertFile=/etc/ssl/certs/fullchain.pem \
  --server.sslKeyFile=/etc/ssl/private/privkey.pem
```
또는 Nginx / AWS ALB(Application Load Balancer) 뒷단에 위치시켜 HTTPS를 종단(SSL Termination)하는 구성을 권장합니다.

---

## 🔄 CI/CD 파이프라인 (GitHub Actions)

`.github/workflows/ci.yml` 파일에 구성된 자동화 파이프라인은 다음과 같은 단계로 코드 품질과 무결성을 점검합니다:

1. **Lint & Quality**: `ruff`를 통해 PEP 8 스타일 가이드 및 코드 오류를 정적 분석.
2. **Automated Testing**: `pytest`를 통해 파서, 청커, 시크릿 매니저, 벡터 DB의 단위 테스트 실행.
3. **Docker Build Validation**: Docker 이미지가 오류 없이 안전하게 빌드되는지 사전 검증.

---

## 📄 라이선스
본 프로젝트는 MIT 라이선스에 따라 자유롭게 수정 및 배포할 수 있습니다.
