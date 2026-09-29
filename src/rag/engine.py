"""
Conversational RAG Engine powered by Local Ollama Models.
Utilizes Korean SOTA models (e.g. EXAONE 3.5, Qwen 2.5) completely locally.
Handles conversational memory, follow-up query contextualization,
context retrieval, and token-by-token streaming.
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
    """End-to-end conversational RAG pipeline using local Ollama LLM and persistent ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[PersistentVectorStore] = None,
        llm_model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.vector_store = vector_store or PersistentVectorStore()
        self.llm_model = llm_model or config.ollama_llm_model
        self.base_url = base_url or config.ollama_base_url
        self.client = OllamaClient(base_url=self.base_url)

    def check_status(self) -> Dict[str, Any]:
        """Checks connection to Ollama and installed models."""
        return self.client.check_connection()

    def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates follow-up question into a standalone question if dialogue history exists.
        """
        if not history or len(history) == 0:
            return query

        status = self.check_status()
        if not status["connected"]:
            return query

        # Take last 2 turns
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
            rewritten = self.client.generate(model=self.llm_model, prompt=prompt)
            rewritten = rewritten.strip().replace('"', '').replace("'", "")
            logger.info(f"Contextualized query: '{query}' -> '{rewritten}'")
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning(f"Query contextualization error: {e}. Using original query.")
            return query

    def generate_streaming_response(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Retrieves context and yields streaming tokens along with source metadata.
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
            score = chunk.get("similarity", 0.0)
            context_blocks.append(
                f"[문서 {i+1} | 출처: {source} (유사도: {score:.2f})]\n{chunk['text']}"
            )
        formatted_context = "\n\n--------------------\n\n".join(context_blocks)

        # 4. Construct messages for Ollama chat
        user_prompt_with_context = (
            f"[검색된 문서 컨텍스트]\n{formatted_context}\n\n"
            f"[사용자 질문]\n{query}"
        )

        # Check Ollama connection
        conn_status = self.check_status()
        if not conn_status["connected"]:
            mock_reply = (
                f"⚠️ **[Ollama 로컬 서버 미연결 안내]**\n\n"
                f"현재 로컬 Ollama 서비스(`{self.base_url}`)에 연결할 수 없습니다.\n\n"
                f"**해결 방법:**\n"
                f"1. 터미널에서 `ollama serve` 명령어로 Ollama를 실행해주세요.\n"
                f"2. 추천 한국어 모델 다운로드: `ollama pull {self.llm_model}` 및 `ollama pull {config.ollama_embed_model}`\n\n"
                f"---\n"
                f"**검색된 문서 정보 (ChromaDB 정상 검색됨):**\n"
                f"- 가장 관련성 높은 발췌문 ({retrieved_chunks[0]['metadata'].get('source')}):\n"
                f"> {retrieved_chunks[0]['text'][:200]}..."
            )
            import time
            for word in mock_reply.split(" "):
                time.sleep(0.03)
                yield {"type": "token", "content": word + " "}
            return

        # Prepare chat messages
        messages = []
        # Add recent conversation turns
        for m in chat_history[-4:]:
            messages.append({"role": m["role"], "content": m["content"]})
        # Add current user prompt with context
        messages.append({"role": "user", "content": user_prompt_with_context})

        try:
            stream = self.client.chat_stream(
                model=self.llm_model,
                messages=messages,
                system=SYSTEM_PROMPT
            )
            for token in stream:
                yield {"type": "token", "content": token}
        except Exception as e:
            logger.error(f"Error during Ollama streaming: {e}")
            yield {
                "type": "error",
                "content": f"Ollama 로컬 모델('{self.llm_model}') 응답 생성 중 오류: {str(e)}\n\n"
                           f"모델이 설치되어 있는지 확인해주세요: `ollama pull {self.llm_model}`"
            }
