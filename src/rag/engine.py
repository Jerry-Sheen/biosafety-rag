"""
Conversational RAG Engine supporting Google Gemini (Cloud API) and OpenAI.
Ollama support has been completely removed.
Includes dynamic model discovery, fault-tolerant fallbacks, and real-time streaming.
"""

from typing import List, Dict, Any, Generator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from config import config
from src.rag.vector_store import PersistentVectorStore
from src.rag.openai_client import OpenAIClient
from src.rag.gemini_client import GeminiModelResolver, classify_gemini_error
from src.rag.law_search import LawSearchClient
from src.security.logger import get_secure_logger
import os

logger = get_secure_logger("rag_engine")

SYSTEM_PROMPT = """당신은 법령 검색 및 규제 안내 전문가입니다.  
사용자가 특정 병원균 이름을 입력하면 반드시 다음 두 단계의 구조로 답변을 체계적으로 작성하세요:

[1단계: 영구 벡터 DB 기반 적용 법령 및 규제 대상 현황 우선 제시]
- 영구 벡터 DB 검색 결과를 바탕으로 해당 병원균에 적용되는 법령과 규제 대상 여부(생물체위험군 등급, 감염병예방법 고위험병원체, 생화학무기금지법 생물작용제, 유전자변형생물체법 국가관리대상병원체, 가축전염병예방법 특별관리병원체 등)를 표 또는 글머리 기호로 가장 먼저 명확히 제시하세요.
- ★ 가독성 표기 원칙: 법령 적용 여부를 제시할 때, DB의 'True'는 반드시 '해당', 'False'는 반드시 '해당사항 없음'으로 명확하게 변환하여 표기하세요. (예: `감염병예방법(고위험병원체): 해당`, `수산생물질병관리법: 해당사항 없음`)

[2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차 작성]
- 법제처 웹사이트(https://www.law.go.kr) 및 국가법령정보센터 검색 결과에 기반하여, 해당 특정 병원균이 적용되는 법률별로(예: 감염병예방법, 화학생물무기금지법, 유전자변형생물체법, 가축전염병예방법 등) 각각 독립된 하위 섹션(### [법률명])을 나누어 작성하세요.
- 적용되는 각 법률마다 아래의 8대 항목을 하나도 빠짐없이 순서대로 설명하세요:
  1. 분리신고
  2. 이동(분양)신고
  3. 보유·폐기신고
  4. 제조신고 및 개발실험승인
  5. 수입(반입)신고
  6. 인수신고
  7. 수출신고
  8. 인도신고
  (※ 해당 법률에서 특정 항목에 대한 의무나 규정이 없는 경우 반드시 '해당없음'이라고 명시하고 그 이유를 간략히 설명하세요.)
- 각 법률의 8대 항목 설명 직후, 해당 법률에 따른 **'실무 가이드 절차'**를 구체적으로 작성하세요 (소관 부처, 신청/신고 시스템 포털명 및 URL, 단계별 절차 Step 1~3, 구비서류, 주의사항 등).

[작성 주의사항]
- 법제처 웹사이트의 최신 법령(조항 번호, 시행일자)을 반영하여 핵심 내용을 알기 쉽게 요약하세요. 법령 원문을 그대로 복사하지 마세요.
- 적용되는 법률들을 하나로 뭉뚱그려 설명하지 말고, 반드시 법률별로 각각 8대 항목과 실무 가이드 절차를 명확하게 분리하여 작성해야 합니다.
"""


class ConversationalRAGEngine:
    """End-to-end conversational RAG pipeline using Google Gemini API or OpenAI."""

    def __init__(
        self,
        vector_store: Optional[PersistentVectorStore] = None,
        llm_provider: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None
    ):
        self.vector_store = vector_store or PersistentVectorStore()
        self.llm_provider = (llm_provider or config.llm_provider).lower()
        if self.llm_provider not in ("gemini", "openai"):
            self.llm_provider = "gemini"
        # Gemini setup
        self.gemini_api_key = gemini_api_key or config.gemini_api_key
        self.gemini_model_resolver = GeminiModelResolver(api_key=self.gemini_api_key)
        initial_gemini_model = gemini_model or config.gemini_model or "gemini-2.5-flash"
        self.gemini_model, _ = self.gemini_model_resolver.resolve_model(initial_gemini_model)
        # OpenAI setup
        self.openai_api_key = openai_api_key or config.openai_api_key
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.openai_client = OpenAIClient(api_key=self.openai_api_key, model=self.openai_model)
        # Law Search client (law.go.kr)
        self.law_search_client = LawSearchClient()

    def check_status(self) -> Dict[str, Any]:
        """Checks connection status depending on the configured provider."""
        if self.llm_provider == "gemini":
            has_key = bool(self.gemini_api_key and self.gemini_api_key.strip())
            resolved, warn = self.gemini_model_resolver.resolve_model(self.gemini_model)
            self.gemini_model = resolved
            return {
                "provider": "gemini",
                "connected": has_key,
                "model": self.gemini_model,
                "warning": warn,
                "error": None if has_key else "Gemini API 키가 설정되지 않았습니다."
            }
        is_configured = getattr(self.openai_client, "is_configured", False)
        return {
            "provider": "openai",
            "connected": is_configured,
            "model": self.openai_model,
            "warning": None,
            "error": None if is_configured else "OpenAI API 키가 설정되지 않았습니다."
        }

    def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates follow-up question into a standalone question if dialogue history exists.
        """
        if not history or len(history) <= 1:
            return query

        recent_history = history[-10:]  # Use up to last 10 turns for better context retention
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        prompt = (
            f"다음 대화 내용을 바탕으로, 사용자의 마지막 질문을 이전 맥락이 포함된 독립적인 단독 질문(Standalone Search Query)으로 재작성해주세요.\n"
            f"질문에 대한 답변은 일절 하지 말고, 오직 검색용 단독 질문 문장 1개만 출력하세요.\n\n"
            f"[대화 내역]\n{history_str}\n\n"
            f"[사용자의 마지막 질문]\n{query}\n\n"
            f"재작성된 질문:"
        )

        try:
            if self.llm_provider == "gemini" and self.gemini_api_key:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                resolved_model, _ = self.gemini_model_resolver.resolve_model(self.gemini_model)
                resp = client.models.generate_content(
                    model=resolved_model,
                    contents=prompt
                )
                rewritten = (resp.text or "").strip().replace('"', '').replace("'", "")
                if rewritten:
                    logger.info(f"Contextualized query via Gemini ({resolved_model}): '{query}' -> '{rewritten}'")
                    return rewritten
        except Exception as e:
            logger.warning(f"Query contextualization error ({e}); using original query.")

        return query

    def generate_streaming_response(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Retrieves context and yields progress status, retrieved sources, and streaming tokens.
        """
        # Step 1: Progress indicator - Pathogen Classification Retrieval
        yield {
            "type": "status",
            "stage": "searching",
            "message": "영구 벡터 데이터베이스에서 병원체 규제 분류 및 데이터 검색 중..."
        }

        search_query = self.contextualize_query(query, chat_history)

        # Step 2: Retrieve relevant chunks from persistent ChromaDB
        from concurrent.futures import ThreadPoolExecutor

        def _search():
            return self.vector_store.search(
                query=search_query,
                top_k=max(top_k, 6),
                score_threshold=score_threshold,
            )

        retrieved_chunks = []
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_search)
            retrieved_chunks = future.result(timeout=30)
        except Exception as e:
            logger.error(f"Vector store search timeout or error: {e}")
            retrieved_chunks = []
        finally:
            executor.shutdown(wait=False)

        # Step 3: Analyze pathogen classification to identify applicable laws
        full_text_corpus = " ".join([c.get("text", "") for c in retrieved_chunks]) + " " + search_query
        
        # Determine applicable laws based on attributes found in ChromaDB chunks
        is_bioweapon = any(k in full_text_corpus for k in ["생화학무기", "생물작용제", "화학무기"]) or ("생화학무기금지법(생물작용제)\tTrue" in full_text_corpus)
        is_high_risk = any(k in full_text_corpus for k in ["고위험병원체", "감염병"]) or ("감염병예방법(고위험병원체)\tTrue" in full_text_corpus)
        is_lmo = any(k in full_text_corpus for k in ["유전자변형", "LMO"]) or ("유전자변형생물체법" in full_text_corpus and "True" in full_text_corpus)
        is_veterinary = any(k in full_text_corpus for k in ["가축전염병", "수의생명자원"]) or ("특별관리병원체)\tTrue" in full_text_corpus)

        # Step 4: Law.go.kr real-time search & statutory provision mapping
        yield {
            "type": "status",
            "stage": "law_search",
            "message": "법제처(law.go.kr)에서 최신 법령 조문 및 8대 의무 항목 검색 중..."
        }
        law_data = self.law_search_client.search_applicable_laws_for_pathogen(
            pathogen_name=query,
            detected_categories={
                "is_bioweapon": is_bioweapon,
                "is_high_risk": is_high_risk,
                "is_lmo": is_lmo,
                "is_veterinary": is_veterinary,
            }
        )
        law_context_str = self.law_search_client.format_law_context_for_prompt(law_data)

        # Step 5: Retrieve secondary chunks for detailed procedural guidance (8 items)
        yield {
            "type": "status",
            "stage": "procedure_search",
            "message": "8대 신고·허가 세부 실무 절차 가이드라인 검색 및 결합 중..."
        }
        procedure_chunks = []
        try:
            procedure_chunks = self.vector_store.search(
                "제조신고 보유신고 수입허가 인수신고 인도신고 양도신고 폐기신고 절차 서류",
                top_k=3
            )
        except Exception as e:
            logger.warning(f"Procedure chunks search warning: {e}")

        # Combine all sources for UI display (deduplicated by source & text hash)
        seen_texts = set()
        all_sources = []
        for ch in (retrieved_chunks + procedure_chunks):
            t_key = ch.get("text", "")[:80]
            if t_key not in seen_texts:
                seen_texts.add(t_key)
                all_sources.append(ch)

        yield {"type": "sources", "content": all_sources}

        if not retrieved_chunks and not procedure_chunks:
            yield {
                "type": "status",
                "stage": "done",
                "message": "검색 완료 (관련 문서 없음)"
            }
            yield {
                "type": "token",
                "content": "업로드된 문서에서 질문과 관련된 내용을 찾을 수 없습니다. 관리자 포털에서 문서를 먼저 업로드하거나 검색 조건을 조정해보세요."
            }
            return

        # Step 6: Format context blocks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", chunk["metadata"].get("page_start", 1))
            score = chunk.get("similarity", 0.0)
            context_blocks.append(
                f"[문서 {i+1} | 출처: {source} (페이지 {page}, 유사도: {score:.2f})]\n{chunk['text']}"
            )
        formatted_primary_context = "\n\n--------------------\n\n".join(context_blocks)

        proc_blocks = []
        for i, chunk in enumerate(procedure_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", chunk["metadata"].get("page_start", 1))
            proc_blocks.append(
                f"[실무 가이드 {i+1} | {source} (페이지 {page})]\n{chunk['text']}"
            )
        formatted_proc_context = "\n\n--------------------\n\n".join(proc_blocks)

        user_prompt_with_context = (
            f"[검색 대상 병원체]\n{query}\n\n"
            f"[1. 영구 벡터 DB 검색 결과 - 병원체 분류 및 규제 대상 현황 데이터]\n{formatted_primary_context}\n\n"
            f"[2. 법제처(국가법령정보센터) 검색 결과 - 적용 법률별 8대 항목 및 실무 절차 데이터]\n{law_context_str}\n\n"
            f"[3. 실무 지침서 및 상세 신고·허가 가이드 데이터]\n{formatted_proc_context}\n\n"
            f"[사용자 요청]\n{query}\n\n"
            f"위 검색 결과를 바탕으로 반드시 다음 2개 파트로 나누어 체계적으로 답변을 작성하세요:\n\n"
            f"■ [1단계: 영구 벡터 DB 기반 적용 법령 및 규제 대상 현황 우선 제시]\n"
            f"- '영구 벡터 DB' 검색 결과에 나타난 {query}의 생물체위험군 등급 및 법적 규제 속성(감염병예방법 고위험병원체, 화학생물무기금지법 생물작용제, 유전자변형생물체법 국가관리대상병원체, 가축전염병예방법 특별관리병원체, 수산생물질병관리법 등)을 표 또는 글머리 기호로 가장 먼저 명확하게 제시하세요.\n"
            f"- ★ 가독성 표기 원칙: DB 상의 속성값 중 'True'는 반드시 '해당', 'False'는 반드시 '해당사항 없음'으로 변환하여 표기하세요. (예: 감염병예방법(고위험병원체): 해당, 수산생물질병관리법: 해당사항 없음)\n\n"
            f"■ [2단계: 법제처(국가법령정보센터) 검색 기반 법률별 8대 항목 및 실무 가이드 절차]\n"
            f"- 법제처 웹사이트/국가법령정보센터 검색 결과에 기반하여, 해당 병원체({query})가 적용되는 법률별로(예: 《감염병의 예방 및 관리에 관한 법률》, 《화학무기ㆍ생물무기의 금지와 특정화학물질ㆍ생물작용제 등의 제조ㆍ수출입 규제 등에 관한 법률》 등) 각각 독립된 하위 섹션(### [법률명])을 나누어 작성하세요.\n"
            f"- 적용되는 각 법률마다 아래 8대 항목의 규제 내용(해당 없을 시 반드시 '해당없음'과 사유 명시)을 빠짐없이 순서대로 정리하세요:\n"
            f"  1. 분리신고\n"
            f"  2. 이동(분양)신고\n"
            f"  3. 보유·폐기신고\n"
            f"  4. 제조신고 및 개발실험승인\n"
            f"  5. 수입(반입)신고\n"
            f"  6. 인수신고\n"
            f"  7. 수출신고\n"
            f"  8. 인도신고\n"
            f"- 그리고 각 법률별 8대 항목 설명 바로 아래에, 해당 법률에 따른 **'실무 가이드 절차'**(신고/접수 포털 시스템명 및 링크, 단계별 실무 프로세스 Step 1~3, 구비서류 및 주의사항)를 구체적으로 작성하세요.\n"
            f"(※ 주의: 법률들을 하나로 합쳐서 설명하지 말고, 반드시 법률별로 각각 8대 항목과 실무 가이드를 별도로 분리하여 작성해야 합니다.)"
        )

        # Step 7: Generation
        active_model = self.gemini_model if self.llm_provider == "gemini" else self.openai_model
        yield {
            "type": "status",
            "stage": "generating",
            "message": f"{active_model} 모델로 최신 법령 기반 종합 안내 생성 중..."
        }

        def _format_citations() -> str:
            citation_lines = ["\n\n---\n**📚 참고 법령 및 공식 출처**"]
            for law in law_data.get("applicable_laws", []):
                citation_lines.append(f"- 법제처: [{law['full_name']} ({law['effective_date']} 시행)]({law['url']})")
            for i, ch in enumerate(retrieved_chunks[:3], 1):
                src = ch["metadata"].get("source", "Unknown")
                page = ch["metadata"].get("page", ch["metadata"].get("page_start", 1))
                citation_lines.append(f"- 내부 규제 DB {i}: {src} (p.{page})")
            return "\n".join(citation_lines)

        # Step 8-A: Google Gemini Cloud API Streaming
        if self.llm_provider == "gemini":
            if not self.gemini_api_key or not self.gemini_api_key.strip():
                yield {
                    "type": "error",
                    "content": (
                        "⚠️ **Google Gemini API 키가 설정되지 않았습니다.**\n\n"
                        "다수 사용자에게 빠르고 안정적인 실시간 스트리밍 답변을 제공하려면 Gemini API 키가 필요합니다.\n\n"
                        "1. 사이드바의 **'🔑 Gemini API 키'** 입력창에 API 키를 입력하거나,\n"
                        "2. 프로젝트의 `.env` 파일에 `GEMINI_API_KEY=your_key_here`를 등록해주세요.\n\n"
                        "👉 **[Google AI Studio에서 무료 API 키 즉시 발급받기](https://aistudio.google.com/app/apikey)**"
                    )
                }
                return
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.gemini_api_key)

                # Dynamically resolve and verify model
                resolved_model, warn = self.gemini_model_resolver.resolve_model(self.gemini_model)
                self.gemini_model = resolved_model
                if warn:
                    logger.info(f"Model resolution notice: {warn}")

                # Format conversation turns for multi-turn Gemini chat
                contents = []
                for m in chat_history[-4:]:
                    role = "model" if m["role"] == "assistant" else "user"
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=m["content"])]
                    ))
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt_with_context)]
                ))

                response_stream = client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        top_p=0.95
                    )
                )

                first_token = True
                for chunk in response_stream:
                    if chunk.text:
                        if first_token:
                            first_token = False
                            yield {"type": "status", "stage": "streaming", "message": "답변 스트리밍 중..."}
                        yield {"type": "token", "content": chunk.text}

                # 스트리밍 완료 후 법제처 및 문서 출처 추가 출력
                try:
                    yield {"type": "token", "content": _format_citations()}
                except Exception as cite_err:
                    logger.warning(f"Failed to format citations: {cite_err}")

                yield {"type": "status", "stage": "done", "message": "답변 완료"}

            except Exception as e:
                diag = classify_gemini_error(e)
                logger.error(f"Gemini streaming error ({diag['error_type']}): {e}")
                yield {
                    "type": "error",
                    "content": f"{diag['user_message']}\n\n*상세 오류: {str(e)}*"
                }
            return

        # Step 8-B: OpenAI Streaming
        if self.llm_provider == "openai":
            if not getattr(self.openai_client, "is_configured", False):
                yield {
                    "type": "error",
                    "content": "⚠️ **OpenAI API 키가 설정되지 않았습니다.**\n\n"
                               "OpenAI 모델을 사용하려면 사이드바에서 API 키를 입력하거나 .env 파일에 OPENAI_API_KEY를 설정해주세요."
                }
                return
            messages = []
            for m in chat_history[-4:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_prompt_with_context})
            try:
                stream = self.openai_client.chat_stream(
                    messages=messages,
                    system=SYSTEM_PROMPT
                )
                first_token = True
                for token in stream:
                    if first_token:
                        first_token = False
                        yield {"type": "status", "stage": "streaming", "message": "답변 스트리밍 중..."}
                    yield {"type": "token", "content": token}
                try:
                    yield {"type": "token", "content": _format_citations()}
                except Exception as cite_err:
                    logger.warning(f"Failed to format citations: {cite_err}")
                yield {"type": "status", "stage": "done", "message": "답변 완료"}
            except Exception as e:
                logger.error(f"Error during OpenAI streaming: {e}")
                user_msg = (
                    "⚠️ **OpenAI API 연결에 실패했습니다.**\n\n"
                    "1) API 키가 올바르게 설정되었는지 확인하세요.\n"
                    "2) 네트워크가 정상인지 확인하고 방화벽 등에서 차단되지 않았는지 확인하십시오.\n"
                    "3) `.env` 파일에 `OPENAI_API_KEY=your_key` 를 추가하거나, 사이드바에 키를 입력하십시오."
                )
                yield {"type": "error", "content": f"{user_msg}\n\n*상세 오류: {str(e)}*"}
                return

