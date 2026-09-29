"""
Conversational RAG Engine.
Handles multi-turn conversation memory, query reformulation, context retrieval,
and streaming answer generation via Google Gemini API.
"""
from typing import List, Dict, Any, Generator, Optional
from src.config import config
from src.rag.vector_store import PersistentVectorStore
from src.security.logger import get_secure_logger

logger = get_secure_logger("rag_engine")

SYSTEM_PROMPT = """당신은 신뢰할 수 있는 지식 기반 전문 AI 어시스턴트입니다.
제공된 [검색된 문서 컨텍스트]를 꼼꼼히 확인하고, 사용자의 질문에 대해 친절하고 정확하며 논리적인 한국어로 답변하세요.

[답변 원칙]
1. 반드시 제공된 컨텍스트의 내용에 기반하여 사실적으로 답변하세요.
2. 컨텍스트에 없는 내용을 추측하거나 임의로 지어내지(환각/Hallucination) 마세요.
3. 질문과 관련된 내용이 컨텍스트에 없다면, "제공된 문서 내용에서는 해당 정보를 찾을 수 없습니다."라고 정중히 밝히세요.
4. 표(Table)나 수치 데이터가 포함된 경우 마크다운 표나 글머리 기호(bullet points)를 활용하여 보기 쉽게 정리하세요.
5. 답변 마지막에는 답변에 참고한 문서의 출처(문서명, 페이지 등)를 간략히 표기하세요.
"""


class ConversationalRAGEngine:
    """End-to-end conversational RAG pipeline with streaming response."""

    def __init__(
        self,
        vector_store: Optional[PersistentVectorStore] = None,
        llm_model: Optional[str] = None
    ):
        self.vector_store = vector_store or PersistentVectorStore()
        self.llm_model = llm_model or config.llM_model
        self.api_key = config.gemini_api_key
        self._client = None
        self._is_mock = False

        if not self.api_key or self.api_key.startswith("your_gemini") or len(self.api_key) < 10:
            logger.warning("No valid GEMINI_API_KEY. ConversationalRAG running in DEMO/MOCK mode.")
            self._is_mock = True
        else:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini Client for LLM model: {self.llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {e}. Falling back to demo mode.")
                self._is_mock = True

    def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates follow-up question into a standalone question if history exists.
        """
        if not history or len(history) == 0 or self._is_mock or not self._client:
            return query

        # Take last 2-3 turns for query rewriting context
        recent_history = history[-4:]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        prompt = (
            f"다음 대화 내용을 바탕으로, 사용자의 마지막 질문을 이전 맥락이 포함된 독립적인 단독 질문(Standalone Query)으로 재작성해주세요.\n"
            f"질문에 대한 답변을 하지 말고, 오직 검색용 단독 질문 문장 1개만 출력하세요.\n\n"
            f"[대화 내역]\n{history_str}\n\n"
            f"[사용자의 마지막 질문]\n{query}\n\n"
            f"재작성된 질문:"
        )

        try:
            response = self._client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )
            rewritten = response.text.strip()
            logger.info(f"Contextualized query: '{query}' -> '{rewritten}'")
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning(f"Query contextualization failed: {e}. Using original query.")
            return query

    def generate_streaming_response(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Retrieves context and yields streaming tokens along with metadata.
        Yields dict: {"type": "token"|"sources"|"error", "content": ...}
        """
        # 1. Contextualize query if multi-turn dialogue
        search_query = self.contextualize_query(query, chat_history)

        # 2. Retrieve relevant chunks from persistent ChromaDB
        retrieved_chunks = self.vector_store.search(
            query=search_query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # Send retrieved source information first
        yield {"type": "sources", "content": retrieved_chunks}

        if not retrieved_chunks:
            yield {
                "type": "token",
                "content": "업로드된 문서에서 질문과 관련된 내용을 찾을 수 없습니다. 문서를 먼저 업로드하거나 검색 조건을 완화해보세요."
            }
            return

        # 3. Format context string
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page_count", "")
            score = chunk.get("similarity", 0.0)
            context_blocks.append(
                f"[문서 {i+1} | 출처: {source} (유사도: {score:.2f})]\n{chunk['text']}"
            )
        formatted_context = "\n\n--------------------\n\n".join(context_blocks)

        # 4. Construct conversation prompt
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"[검색된 문서 컨텍스트]\n{formatted_context}\n\n"
            f"[사용자 질문]\n{query}\n\n"
            f"[답변]"
        )

        # 5. Stream LLM Generation
        if self._is_mock or not self._client:
            # Demonstration streaming generator when no API key configured
            mock_reply = (
                f"💡 [데모 모드 응답]\n\n"
                f"'{query}'에 대한 답변입니다.\n\n"
                f"**검색된 주요 내용:**\n"
                f"{retrieved_chunks[0]['text'][:200]}...\n\n"
                f"> **출처:** {retrieved_chunks[0]['metadata'].get('source')}\n\n"
                f"*(안내: 실제 Gemini LLM 답변 생성을 원하시면 `.env` 파일에 유효한 `GEMINI_API_KEY`를 설정해주세요.)*"
            )
            import time
            words = mock_reply.split(" ")
            for w in words:
                time.sleep(0.04)
                yield {"type": "token", "content": w + " "}
            return

        try:
            stream = self._client.models.generate_content_stream(
                model=self.llm_model,
                contents=prompt
            )
            for chunk in stream:
                if chunk.text:
                    yield {"type": "token", "content": chunk.text}
        except Exception as e:
            logger.error(f"Error during Gemini streaming generation: {e}")
            yield {"type": "error", "content": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}
