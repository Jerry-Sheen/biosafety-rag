"""
Conversational RAG Engine supporting both Google Gemini (Cloud API) and local Ollama.
Eliminates local Ollama installation requirement for seamless multi-user deployment,
while preserving optional local Ollama capability.
Includes real-time streaming, query contextualization, and step-by-step progress tracking.
"""
from typing import List, Dict, Any, Generator, Optional
from src.config import config
from src.rag.vector_store import PersistentVectorStore
from src.rag.ollama_client import OllamaClient
from src.security.logger import get_secure_logger

logger = get_secure_logger("rag_engine")

SYSTEM_PROMPT = """당신은 신뢰할 수 있는 전문 지식 어시스턴트입니다.
제공된 [검색된 문서 컨텍스트]를 꼼꼼히 확인하고, 사용자의 질문에 대해 친절하고 정확하며 자연스러운 한국어로 답변하세요.

[답변 원칙]
1. 반드시 제공된 컨텍스트의 내용에 기반하여 사실적으로 답변하세요.
2. 컨텍스트에 없는 내용을 추측하거나 임의로 지어내지(환각/Hallucination) 마세요.
3. 질문과 관련된 내용이 컨텍스트에 없다면, "제공된 문서 내용에서는 해당 정보를 찾을 수 없습니다."라고 정중히 밝히세요.
4. 표(Table)나 수치 데이터가 포함된 경우 마크다운 표나 글머리 기호(bullet points)를 활용하여 보기 쉽게 정리하세요.
5. 답변 마지막에는 참고한 문서의 출처(문서명, 페이지 등)를 표기하세요.
"""


class ConversationalRAGEngine:
    """End-to-end conversational RAG pipeline using Google Gemini API or local Ollama LLM."""

    def __init__(
        self,
        vector_store: Optional[PersistentVectorStore] = None,
        llm_provider: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None
    ):
        self.vector_store = vector_store or PersistentVectorStore()
        self.llm_provider = (llm_provider or config.llm_provider).lower()
        self.gemini_api_key = gemini_api_key or config.gemini_api_key
        self.gemini_model = gemini_model or config.gemini_model
        self.ollama_model = ollama_model or config.ollama_llm_model
        self.ollama_base_url = ollama_base_url or config.ollama_base_url
        self.ollama_client = OllamaClient(base_url=self.ollama_base_url)

    def check_status(self) -> Dict[str, Any]:
        """Checks connection status depending on the configured provider."""
        if self.llm_provider == "gemini":
            has_key = bool(self.gemini_api_key and self.gemini_api_key.strip())
            return {
                "provider": "gemini",
                "connected": has_key,
                "model": self.gemini_model,
                "error": None if has_key else "Gemini API 키가 설정되지 않았습니다."
            }
        else:
            status = self.ollama_client.check_connection()
            status["provider"] = "ollama"
            status["model"] = self.ollama_model
            return status

    def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates follow-up question into a standalone question if dialogue history exists.
        Skips LLM call if history is short or LLM is unreachable to prevent latency spikes.
        """
        if not history or len(history) <= 1:
            return query

        recent_history = history[-4:]
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
                resp = client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                rewritten = (resp.text or "").strip().replace('"', '').replace("'", "")
                if rewritten:
                    logger.info(f"Contextualized query via Gemini: '{query}' -> '{rewritten}'")
                    return rewritten
            elif self.llm_provider == "ollama":
                status = self.ollama_client.check_connection()
                if status["connected"]:
                    rewritten = self.ollama_client.generate(model=self.ollama_model, prompt=prompt)
                    rewritten = rewritten.strip().replace('"', '').replace("'", "")
                    if rewritten:
                        logger.info(f"Contextualized query via Ollama: '{query}' -> '{rewritten}'")
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
        Yields dict: {"type": "status"|"sources"|"token"|"error", "content": ...}
        """
        # Step 1: Progress indicator - Retrieval
        yield {
            "type": "status",
            "stage": "searching",
            "message": "영구 벡터 데이터베이스에서 관련 문서 검색 중..."
        }

        search_query = self.contextualize_query(query, chat_history)

        # Step 2: Retrieve relevant chunks from persistent ChromaDB
        retrieved_chunks = self.vector_store.search(
            query=search_query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # Send retrieved source information
        yield {"type": "sources", "content": retrieved_chunks}

        if not retrieved_chunks:
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

        # Step 3: Format context string
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            score = chunk.get("similarity", 0.0)
            context_blocks.append(
                f"[문서 {i+1} | 출처: {source} (유사도: {score:.2f})]\n{chunk['text']}"
            )
        formatted_context = "\n\n--------------------\n\n".join(context_blocks)

        user_prompt_with_context = (
            f"[검색된 문서 컨텍스트]\n{formatted_context}\n\n"
            f"[사용자 질문]\n{query}"
        )

        # Step 4: Progress indicator - Generation
        active_model = self.gemini_model if self.llm_provider == "gemini" else self.ollama_model
        yield {
            "type": "status",
            "stage": "generating",
            "message": f"{active_model} 모델로 실시간 답변 생성 중..."
        }

        # Step 5-A: Google Gemini Cloud API Streaming
        if self.llm_provider == "gemini":
            if not self.gemini_api_key or not self.gemini_api_key.strip():
                yield {
                    "type": "error",
                    "content": (
                        "⚠️ **Google Gemini API 키가 설정되지 않았습니다.**\n\n"
                        "다수 사용자에게 빠르고 안정적인 실시간 스트리밍 답변을 제공하려면 Gemini API 키가 필요합니다.\n\n"
                        "1. 사이드바의 **'🔑 Gemini API 키'** 입력창에 API 키를 입력하거나,\n"
                        "2. 프로젝트의 `.env` 파일에 `GEMINI_API_KEY=your_key_here`를 등록해주세요.\n\n"
                        "👉 **[Google AI Studio에서 무료 API 키 즉시 발급받기](https://aistudio.google.com/app/apikey)** (1분 소요, 신용카드 불필요)"
                    )
                }
                return

            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.gemini_api_key)

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

                yield {"type": "status", "stage": "done", "message": "답변 완료"}

            except Exception as e:
                logger.error(f"Gemini streaming error: {e}")
                yield {
                    "type": "error",
                    "content": f"Google Gemini API 응답 생성 중 오류: {str(e)}\n\nAPI 키가 유효한지 확인해주세요."
                }
            return

        # Step 5-B: Local Ollama Streaming
        conn_status = self.ollama_client.check_connection()
        if not conn_status["connected"]:
            mock_reply = (
                f"⚠️ **[Ollama 로컬 서버 미연결 안내]**\n\n"
                f"현재 로컬 Ollama 서비스(`{self.ollama_base_url}`)에 연결할 수 없습니다.\n\n"
                f"**해결 방법:**\n"
                f"1. 로컬 설치 없이 배포하려면 사이드바에서 **'Google Gemini'** 모드를 선택해주세요.\n"
                f"2. 또는 터미널에서 `ollama serve` 명령어로 Ollama를 시작하고 모델을 준비해주세요: `ollama pull {self.ollama_model}`\n\n"
                f"---\n"
                f"**검색된 문서 정보 (ChromaDB 정상 검색됨):**\n"
                f"- 출처: {retrieved_chunks[0]['metadata'].get('source')}\n"
                f"> {retrieved_chunks[0]['text'][:300]}..."
            )
            import time
            for word in mock_reply.split(" "):
                time.sleep(0.02)
                yield {"type": "token", "content": word + " "}
            yield {"type": "status", "stage": "done", "message": "완료"}
            return

        messages = []
        for m in chat_history[-4:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_prompt_with_context})

        try:
            stream = self.ollama_client.chat_stream(
                model=self.ollama_model,
                messages=messages,
                system=SYSTEM_PROMPT
            )
            first_token = True
            for token in stream:
                if first_token:
                    first_token = False
                    yield {"type": "status", "stage": "streaming", "message": "답변 스트리밍 중..."}
                yield {"type": "token", "content": token}

            yield {"type": "status", "stage": "done", "message": "답변 완료"}
        except Exception as e:
            logger.error(f"Error during Ollama streaming: {e}")
            yield {
                "type": "error",
                "content": f"Ollama 로컬 모델('{self.ollama_model}') 응답 생성 중 오류: {str(e)}\n\n"
                           f"로컬 사양 부족으로 타임아웃이 발생한 경우, 사이드바에서 **Google Gemini** 모드로 전환하는 것을 강력 추천합니다."
            }
