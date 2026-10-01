"""
==============================================================================
User Conversational RAG Application (user.py)
------------------------------------------------------------------------------
사용자 전용 지식 기반 대화형 질의응답 시스템.
독립적으로 실행 가능: streamlit run user.py

기능:
1. 사용자 자연어 질문 입력 -> 영구 벡터 DB(ChromaDB)에서 관련 컨텍스트 검색.
2. 로컬 Ollama LLM(EXAONE 3.5 / Qwen 2.5)을 통한 고성능 한국어 지식 기반 답변 생성.
3. 실시간 토큰 스트리밍(Streaming) 출력으로 즉각적인 피드백 제공.
4. 멀티턴 대화형 RAG: 이전 대화 기억 및 문맥 기반 후속 질의 재구성.
5. 참고 출처(Citations) 및 발췌 문맥 투명한 제공.
==============================================================================
"""
from typing import List, Dict, Any
import streamlit as st

from src.config import config
from src.rag.vector_store import PersistentVectorStore
from src.rag.engine import ConversationalRAGEngine
from src.rag.ollama_client import OllamaClient
from src.security.logger import get_secure_logger

logger = get_secure_logger("user_app")

# Streamlit Page Settings (Safe for both standalone and imported execution)
try:
    st.set_page_config(
        page_title="RAG Vibe - 사용자 지식 대화 시스템 (User Chat)",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

# Custom Styling
st.markdown(
    """
    <style>
    .user-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .user-desc {
        color: #4B5563;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="로컬 AI 엔진 및 벡터 데이터베이스 초기화 중...")
def initialize_user_rag():
    """Initializes and caches vector store and conversational engine."""
    vector_store = PersistentVectorStore()
    rag_engine = ConversationalRAGEngine(vector_store=vector_store)
    return vector_store, rag_engine


def main():
    st.markdown('<div class="user-header">💬 RAG Vibe - 지식 기반 대화형 AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="user-desc">업로드된 사내/개인 문서를 바탕으로 AI가 정확한 답변을 실시간 스트리밍으로 제공합니다. (Google Gemini 및 로컬 Ollama 지원)</div>',
        unsafe_allow_html=True
    )

    vector_store, rag_engine = initialize_user_rag()
    ollama_client = OllamaClient(base_url=config.ollama_base_url)

    # 1. Sidebar Configuration & Status
    with st.sidebar:
        st.subheader("🤖 AI 모델 및 엔진 설정")
        
        provider_option = st.radio(
            "사용할 LLM 제공자 선택:",
            options=["✨ Google Gemini (추천: 설치 불필요/초고속)", "🦙 로컬 Ollama (오프라인 로컬 실행)"],
            index=0 if config.llm_provider == "gemini" else 1,
            help="Google Gemini를 선택하면 별도의 로컬 Ollama 설치 없이 다수의 사용자가 즉시 쾌적하게 이용할 수 있습니다."
        )

        use_gemini = "Gemini" in provider_option
        rag_engine.llm_provider = "gemini" if use_gemini else "ollama"

        if use_gemini:
            st.caption("🚀 **클라우드 API 모드**: 별도의 로컬 서버 설치 없이 안정적인 스트리밍을 제공합니다.")
            
            # API Key Configuration
            current_key = st.session_state.get("user_gemini_api_key", config.gemini_api_key)
            api_key_input = st.text_input(
                "🔑 Gemini API 키:",
                value=current_key if current_key else "",
                type="password",
                help=".env에 GEMINI_API_KEY를 설정하거나 여기에 직접 입력하세요. (aistudio.google.com 무료 발급)"
            )
            if api_key_input:
                st.session_state.user_gemini_api_key = api_key_input
                rag_engine.gemini_api_key = api_key_input
                st.success("API 키가 적용되었습니다.", icon="✅")
            else:
                rag_engine.gemini_api_key = ""
                st.warning("Gemini API 키가 필요합니다. [무료 발급받기](https://aistudio.google.com/app/apikey)")

            gemini_model_options = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash"]
            default_idx = gemini_model_options.index(config.gemini_model) if config.gemini_model in gemini_model_options else 0
            selected_gemini_model = st.selectbox(
                "Gemini 모델 선택:",
                options=gemini_model_options,
                index=default_idx
            )
            rag_engine.gemini_model = selected_gemini_model

        else:
            st.caption("💻 **로컬 Ollama 모드**: 로컬 하드웨어에서 100% 오프라인으로 실행됩니다.")
            ollama_status = ollama_client.check_connection()
            if ollama_status["connected"]:
                st.success(f"Ollama 연결됨 (`{config.ollama_base_url}`)")
                installed_models = ollama_status["models"]

                if installed_models:
                    selected_model = st.selectbox(
                        "사용할 Ollama 모델 선택:",
                        options=installed_models,
                        index=installed_models.index(config.ollama_llm_model) if config.ollama_llm_model in installed_models else 0
                    )
                    rag_engine.ollama_model = selected_model
                else:
                    st.info(f"설치된 모델 없음. 기본 모델: `{config.ollama_llm_model}`")
            else:
                st.error(f"Ollama 미연결 (`{config.ollama_base_url}`)")
                st.caption("터미널에서 `ollama serve`를 실행하거나, 상단에서 **Google Gemini** 모드를 선택해주세요.")

        st.divider()
        st.subheader("⚙️ 검색 파라미터")
        top_k = st.slider("참고할 청크 수 (Top-K)", min_value=1, max_value=10, value=4, step=1)
        score_threshold = st.slider(
            "최소 유사도 기준 (Threshold)", min_value=0.0, max_value=0.9, value=0.1, step=0.05,
            help="유사도가 이 값 이상인 문서 청크만 답변 컨텍스트로 활용합니다."
        )

        st.divider()
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            st.session_state.user_messages = []
            st.rerun()

        st.divider()
        st.subheader("📚 현재 등록된 문서 목록")
        stats = vector_store.get_stats()
        sources = stats.get("document_sources", [])
        if sources:
            for s in sources:
                st.markdown(f"- 📄 `{s}`")
        else:
            st.info("등록된 문서가 없습니다. `admin.py`에서 먼저 문서를 업로드해주세요.")

    # 2. Initialize chat history
    if "user_messages" not in st.session_state:
        st.session_state.user_messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! 등록된 문서를 기반으로 질문에 명확하고 상세하게 답변해 드립니다. 무엇이든 질문해 보세요.",
                "sources": []
            }
        ]

    # 3. Render conversation history
    for msg in st.session_state.user_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 참고 출처 및 컨텍스트 (Sources & Citations)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"**[{idx+1}] {src['metadata'].get('source', '문서')}** "
                            f"(유사도: `{src.get('similarity', 0.0):.2f}`)"
                        )
                        st.text(src["text"])

    # 4. Handle user input
    if prompt := st.chat_input("문서에 대해 질문하세요... (예: 생물작용제 및 독소 신고 제도의 제출처를 알려줘)"):
        # Append user message
        st.session_state.user_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response with visual indicators
        with st.chat_message("assistant"):
            # Step-by-step progress visual indicator
            status_box = st.status("💡 답변 생성 프로세스를 시작합니다...", expanded=True)
            response_placeholder = st.empty()
            sources_container = st.container()

            # Pass history (excluding current user query) for contextualization
            history_for_engine = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.user_messages[:-1]
            ]

            full_response = ""
            retrieved_sources: List[Dict[str, Any]] = []

            # Stream response from RAG engine
            stream_gen = rag_engine.generate_streaming_response(
                query=prompt,
                chat_history=history_for_engine,
                top_k=top_k,
                score_threshold=score_threshold
            )

            for event in stream_gen:
                event_type = event.get("type")
                if event_type == "status":
                    stage = event.get("stage")
                    msg = event.get("message")
                    if stage == "searching":
                        status_box.update(label=f"🔍 {msg}", state="running")
                    elif stage == "generating":
                        status_box.update(label=f"🧠 {msg}", state="running")
                    elif stage == "streaming":
                        status_box.update(label="✍️ 실시간 답변 스트리밍 중...", state="running")
                    elif stage == "done":
                        status_box.update(label="✅ 답변 생성 완료", state="complete", expanded=False)
                elif event_type == "sources":
                    retrieved_sources = event.get("content", [])
                    if retrieved_sources:
                        status_box.write(f"📖 **{len(retrieved_sources)}개**의 관련 문서 청크를 찾았습니다.")
                    else:
                        status_box.write("⚠️ 관련 문서를 찾지 못했습니다.")
                elif event_type == "token":
                    full_response += event.get("content", "")
                    response_placeholder.markdown(full_response + " ▌")
                elif event_type == "error":
                    full_response += f"\n\n⚠️ {event.get('content')}"
                    response_placeholder.markdown(full_response)
                    status_box.update(label="⚠️ 오류가 발생했습니다.", state="error", expanded=True)

            # Final clean render without cursor
            response_placeholder.markdown(full_response)

            # Display citations expander if sources were retrieved
            if retrieved_sources:
                with sources_container.expander("🔍 참고 출처 및 컨텍스트 (Sources & Citations)", expanded=False):
                    for idx, src in enumerate(retrieved_sources):
                        st.markdown(
                            f"**[{idx+1}] {src['metadata'].get('source', '문서')}** "
                            f"(유사도: `{src.get('similarity', 0.0):.2f}`)"
                        )
                        st.text(src["text"])

            # Store to history
            st.session_state.user_messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_sources
            })


if __name__ == "__main__":
    main()
