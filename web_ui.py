"""RAG 检索增强生成系统 — Streamlit Web UI"""
import os
import sys

import streamlit as st

# 将项目根目录加入 path，确保能 import rag_chroma_QA_fusion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_chroma_QA_fusion import (  # noqa: E402
    llm,
    embeddings,
    chunk_text,
    format_docs,
    build_answer_chain,
    rag_query,
    Chroma,
)

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="wide",
)

# ==================== 常量 ====================

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
COLLECTION_NAME = "knowledge_base"


# ==================== 缓存资源 ====================

@st.cache_resource
def get_vectorstore():
    """缓存 ChromaDB 连接，Streamlit rerun 时复用"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


@st.cache_resource
def get_answer_chain():
    """缓存 LCEL Answer Chain"""
    return build_answer_chain()


# ==================== Session State 初始化 ====================

def init_session_state():
    defaults = {
        "messages": [],
        "use_fusion": True,
        "kb_stats": None,
        "stats_dirty": True,
        "processed_files": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== 工具函数 ====================

def decode_uploaded_file(uploaded_file) -> str:
    """解码上传的文本文件，尝试 UTF-8 和 GBK 编码"""
    raw = uploaded_file.read()
    for enc in ["utf-8", "gbk"]:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError("无法解码文件，请确认编码为 UTF-8 或 GBK。")


def add_documents_to_kb(text: str, source_name: str, vectorstore) -> int:
    """将文本分块并存入向量数据库，返回添加的 chunk 数"""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    existing_ids = vectorstore.get()["ids"]
    start_idx = len(existing_ids)
    ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_name,
            "chunk_index": start_idx + i,
            "char_count": len(t),
            "id": f"chunk_{start_idx + i}",
        }
        for i, t in enumerate(chunks)
    ]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def get_kb_stats(vectorstore) -> dict:
    """获取知识库统计信息"""
    data = vectorstore.get()
    ids = data["ids"]
    metadatas = data["metadatas"]
    sources = sorted(set(m.get("source", "?") for m in metadatas)) if metadatas else []
    return {
        "total_chunks": len(ids),
        "sources": sources,
    }


# ==================== Sidebar ====================

def render_sidebar():
    vectorstore = get_vectorstore()

    with st.sidebar:
        st.header("📚 知识库管理")

        # 统计信息
        if st.session_state.stats_dirty or st.session_state.kb_stats is None:
            st.session_state.kb_stats = get_kb_stats(vectorstore)
            st.session_state.stats_dirty = False

        stats = st.session_state.kb_stats
        st.metric("文本块总数", stats["total_chunks"])

        if stats["sources"]:
            st.caption("已加载文档：")
            for src in stats["sources"]:
                st.caption(f"  • {src}")
        else:
            st.caption("暂无文档，请上传知识库文件。")

        st.divider()

        # 文件上传
        st.subheader("📤 上传文档")
        uploaded = st.file_uploader(
            "选择文本文件 (.txt)",
            type=["txt"],
            help="支持 UTF-8 和 GBK 编码",
            label_visibility="collapsed",
        )
        if uploaded is not None and uploaded.name not in st.session_state.processed_files:
            with st.spinner(f"正在处理 {uploaded.name} ..."):
                try:
                    text = decode_uploaded_file(uploaded)
                except ValueError as e:
                    st.error(str(e))
                    return
                if not text.strip():
                    st.warning("文件内容为空。")
                    return

                added = add_documents_to_kb(text, uploaded.name, vectorstore)
                st.session_state.processed_files.add(uploaded.name)
                st.session_state.stats_dirty = True
                st.success(f"已添加 {added} 个文本块")
                st.rerun()

        st.divider()

        # 检索设置
        st.subheader("⚙️ 检索设置")
        st.session_state.use_fusion = st.toggle(
            "RAG Fusion（多查询融合）",
            value=st.session_state.use_fusion,
            help="开启后 LLM 会生成多个查询变体，通过 RRF 融合提高召回率",
        )

        st.divider()

        # 管理操作
        st.subheader("🔧 操作")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空聊天", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 重置知识库", use_container_width=True, type="secondary"):
                vectorstore.delete_collection()
                get_vectorstore.clear()
                get_answer_chain.clear()
                st.session_state.messages = []
                st.session_state.kb_stats = None
                st.session_state.stats_dirty = True
                st.session_state.processed_files = set()
                st.rerun()


# ==================== 主区域 — 聊天界面 ====================

def render_chat():
    st.title("📚 RAG 知识库问答")
    st.caption("基于 ChromaDB + DeepSeek + DashScope 的检索增强生成系统")

    # 渲染聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("docs"):
                with st.expander("📎 查看检索到的参考文本块"):
                    if msg.get("fusion_queries"):
                        st.caption("Fusion 查询变体：")
                        for q in msg["fusion_queries"]:
                            st.caption(f"  • {q}")
                        st.divider()
                    for i, doc in enumerate(msg["docs"]):
                        src = doc.metadata.get("source", "?")
                        score = ""
                        if msg.get("scores") and i < len(msg["scores"]):
                            score = f"  (RRF: {msg['scores'][i]:.4f})"
                        st.caption(f"参考 {i + 1} — 来源: {src}{score}")
                        st.text(doc.page_content[:600])
                        st.divider()

    # 聊天输入
    if question := st.chat_input("输入你的问题..."):
        vectorstore = get_vectorstore()
        answer_chain = get_answer_chain()

        stats = get_kb_stats(vectorstore)
        if stats["total_chunks"] == 0:
            st.warning("知识库为空，请先在侧边栏上传文档。")
            st.stop()

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答..."):
                try:
                    result = rag_query(
                        question,
                        vectorstore,
                        answer_chain,
                        use_fusion=st.session_state.use_fusion,
                        verbose=False,
                    )
                except Exception as e:
                    st.error(f"查询失败: {e}")
                    return

            if not result["found"]:
                st.warning("未检索到相关文本块，请尝试换个问法或上传相关文档。")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ 未检索到相关文本块。",
                    "docs": [],
                    "scores": [],
                    "fusion_queries": None,
                })
                return

            st.markdown(result["answer"])

            if result["docs"]:
                with st.expander("📎 查看检索到的参考文本块"):
                    if result["fusion_queries"]:
                        st.caption("Fusion 查询变体：")
                        for q in result["fusion_queries"]:
                            st.caption(f"  • {q}")
                        st.divider()
                    for i, doc in enumerate(result["docs"]):
                        src = doc.metadata.get("source", "?")
                        score = ""
                        if result["scores"]:
                            score = f"  (RRF: {result['scores'][i]:.4f})"
                        st.caption(f"参考 {i + 1} — 来源: {src}{score}")
                        st.text(doc.page_content[:600])
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "docs": result["docs"],
                "scores": result["scores"],
                "fusion_queries": result["fusion_queries"],
            })


# ==================== 入口 ====================

def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
