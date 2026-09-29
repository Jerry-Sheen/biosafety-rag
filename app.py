"""
Main Streamlit Application Entrypoint.
Orchestrates User Chat View and Admin Management View.
"""
import streamlit as st
from src.config import config
from src.rag.vector_store import PersistentVectorStore
from src.rag.engine import ConversationalRAGEngine
from src.ui.admin_view import render_admin_view
from src.ui.user_view import render_user_view

# Streamlit Page Configuration
st.set_page_config(
    page_title="RAG Vibe - Multi-document Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-dev {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-prod {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="벡터 데이터베이스 및 AI 엔진 초기화 중...")
def initialize_rag_system():
    """Initializes shared persistent vector store and conversational engine."""
    vector_store = PersistentVectorStore()
    rag_engine = ConversationalRAGEngine(vector_store=vector_store)
    return vector_store, rag_engine


def main():
    vector_store, rag_engine = initialize_rag_system()

    # Header section
    col_h1, col_h2 = st.columns([8, 2])
    with col_h1:
        st.markdown('<div class="main-header">🧠 RAG Vibe - 지능형 지식 검색 시스템</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">다양한 문서(PDF, Docx, Excel, CSV, HWP)를 분석하고 실시간 스트리밍 대화를 제공합니다.</div>',
            unsafe_allow_html=True
        )
    with col_h2:
        env_badge_class = "badge-prod" if config.app_env.lower() == "production" else "badge-dev"
        st.markdown(
            f'<div style="text-align: right; margin-top: 15px;">'
            f'<span class="{env_badge_class}">ENV: {config.app_env.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Admin / User Mode Tabs
    mode_tab = st.radio(
        "시스템 모드 선택",
        ["💬 사용자 질의응답 (User Chat)", "🛠️ 관리자 문서 관리 (Admin Management)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    if "사용자 질의응답" in mode_tab:
        render_user_view(rag_engine=rag_engine, vector_store=vector_store)
    else:
        render_admin_view(vector_store=vector_store)


if __name__ == "__main__":
    main()
