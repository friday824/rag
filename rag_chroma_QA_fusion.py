"""RAG 检索增强生成系统 — 基于 LangChain QA Chain + Chroma 向量数据库"""
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================== 1. LLM & Embeddings 配置 ====================

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY")

if not DEEPSEEK_KEY or not DASHSCOPE_KEY:
    raise RuntimeError(
        "缺少 API 密钥，请在项目根目录的 .env 文件中配置 "
        "DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY（参考 .env.example）"
    )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com/v1",
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    dimensions=1024,
    api_key=DASHSCOPE_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
)


# ==================== 2. 文档切分 ====================

def chunk_text(text, chunk_size=200, chunk_overlap=40):
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "？", "！", "；", "，", " ", ""],
    )
    return splitter.split_text(text)


# ==================== 3. RAG Fusion ====================

QUERY_GEN_PROMPT = """你是一个搜索查询优化专家。根据用户的原始问题，生成 {n} 个不同的搜索查询变体。
每个查询从不同角度重新表述问题，或将其拆解为子问题，以最大化检索召回率。
每行输出一个查询，以 "- " 开头。只输出查询，不要有任何其他内容。

原始问题：{question}"""


def generate_query_variations(question, n=4):
    prompt = QUERY_GEN_PROMPT.format(n=n, question=question)
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    queries = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            queries.append(line[2:])
        elif line.startswith("-"):
            queries.append(line[1:])
    if not queries:
        queries = [question]
    return queries[:n]


def reciprocal_rank_fusion(doc_lists, k=60):
    """RRF: score(d) = Σ 1/(k + rank(d, q))"""
    scores = {}
    doc_map = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in sorted_items]


def fusion_retrieve(question, retriever, n_queries=4, top_k=5, verbose=True):
    queries = generate_query_variations(question, n=n_queries)
    all_queries = [question] + queries

    if verbose:
        print(f"\n查询变体 ({len(queries)} 个):")
        for i, q in enumerate(queries):
            print(f"  {i + 1}. {q}")

    doc_lists = []
    for q in all_queries:
        docs = retriever.invoke(q)
        doc_lists.append(docs)

    fused = reciprocal_rank_fusion(doc_lists, k=60)
    return fused[:top_k], queries


# ==================== 4. QA Chain (LCEL) ====================

QA_SYSTEM_PROMPT = """你是一个知识助手。请根据以下参考资料回答用户的问题。如果参考资料中包含相关信息，请准确回答。如果参考资料中没有相关信息，请如实告知用户，不要编造内容。

参考资料：
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("human", "{input}"),
])


def format_docs(docs):
    return "\n\n".join(f"[参考{i + 1}] {d.page_content}" for i, d in enumerate(docs))


def build_answer_chain():
    """纯 QA 链: context + input → answer（retriever 由外部负责）"""
    return qa_prompt | llm | StrOutputParser()


def rag_query(question, vectorstore, answer_chain, use_fusion=True, verbose=True):
    if verbose:
        print(f"\n问题: {question}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    result = {
        "answer": "",
        "docs": [],
        "scores": [],
        "fusion_queries": None,
        "found": False,
    }

    if use_fusion:
        top_docs, queries = fusion_retrieve(question, retriever, verbose=verbose)

        if not top_docs:
            if verbose:
                print("未检索到相关文本块。")
            return result

        result["fusion_queries"] = queries
        if verbose:
            print(f"\n--- RRF 融合后 Top {len(top_docs)} 条文本块 ---")
        fused_docs = []
        scores = []
        for i, (doc, score) in enumerate(top_docs):
            if verbose:
                preview = doc.page_content.replace("\n", " ")[:100]
                src = doc.metadata.get("source", "?")
                print(f"[参考{i + 1}] 来源: {src}  RRF得分: {score:.4f}")
                print(f"  内容: {preview}...\n")
            fused_docs.append(doc)
            scores.append(score)

        context = format_docs(fused_docs)
        result["docs"] = fused_docs
        result["scores"] = scores
    else:
        if verbose:
            print("正在检索...\n")
        docs = retriever.invoke(question)
        if not docs:
            if verbose:
                print("未检索到相关文本块。")
            return result
        if verbose:
            print(f"--- 检索到的 {len(docs)} 条相关文本块 ---")
            for i, doc in enumerate(docs):
                preview = doc.page_content.replace("\n", " ")[:100]
                src = doc.metadata.get("source", "?")
                print(f"[参考{i + 1}] 来源: {src}")
                print(f"  内容: {preview}...\n")
        context = format_docs(docs)
        result["docs"] = docs

    if verbose:
        print("正在生成回答...\n")
    answer = answer_chain.invoke({"context": context, "input": question})
    result["answer"] = answer
    result["found"] = True

    if verbose:
        print("--- AI 回答 ---")
        print(answer)
        print("-" * 60)

    return result


# ==================== 5. 交互式入口 ====================

def load_knowledge_base():
    print("=" * 60)
    print("RAG 检索增强生成系统 (LangChain QA Chain + Chroma)")
    print("=" * 60)
    print("\n请选择知识库来源：")
    print("  1. 输入文本文件路径")
    print("  2. 直接在控制台输入文本")
    print()

    choice = input("请输入选项 (1 或 2): ").strip()

    if choice == "1":
        path = input("请输入文本文件路径: ").strip()
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            sys.exit(1)
        for enc in ["utf-8", "gbk"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            print("无法读取文件，请确认文件编码为 UTF-8 或 GBK。")
            sys.exit(1)
        source = os.path.basename(path)
        print(f"已读取文件 [{source}]，共 {len(text)} 个字符。")
        return text, source

    elif choice == "2":
        print("请输入知识库文本（输入完成后按回车，然后输入 'END' 结束）：")
        lines = []
        while True:
            line = sys.stdin.readline()
            if line is None or line.strip() == "END":
                break
            lines.append(line)
        text = "".join(lines)
        if not text.strip():
            print("输入文本为空，请重新运行程序。")
            sys.exit(1)
        print(f"已读取，共 {len(text)} 个字符。")
        return text, "console_input"

    else:
        print("无效选项。")
        sys.exit(1)


def main():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")

    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    existing = len(vectorstore.get()["ids"])

    if existing > 0:
        print(f"发现已有向量数据库: {existing} 条文本块")
        reuse = input("是否复用已有知识库？(y/n, 默认 y): ").strip().lower()
        if reuse in ("", "y", "yes"):
            print(f"复用已有知识库，共 {existing} 条文本块。\n")
        else:
            vectorstore.delete_collection()
            vectorstore = Chroma(
                collection_name="knowledge_base",
                embedding_function=embeddings,
                persist_directory=db_path,
            )
            existing = 0

    if existing == 0:
        text, source = load_knowledge_base()

        print("\n正在切分文本...")
        chunks = chunk_text(text)
        if not chunks:
            print("文本切分后为空，请检查输入内容。")
            sys.exit(1)
        print(f"共切分为 {len(chunks)} 个文本块。")

        print(f"正在向量化 {len(chunks)} 条文本块...")
        metadatas = [{"source": source, "chunk_index": i, "char_count": len(t), "id": f"chunk_{i}"} for i, t in enumerate(chunks)]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        print("向量存储完成。")

    answer_chain = build_answer_chain()

    print("\n" + "=" * 60)
    print("知识库已就绪，开始问答交互 (Fusion 模式: 开启)。")
    print("输入 'exit' 或 'quit' 退出程序。")
    print("输入 'fusion off' 关闭多查询融合。")
    print("输入 'fusion on' 开启多查询融合。")
    print("=" * 60)

    use_fusion = True
    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        if question.lower() == "fusion off":
            use_fusion = False
            print("RAG Fusion 已关闭。")
            continue

        if question.lower() == "fusion on":
            use_fusion = True
            print("RAG Fusion 已开启。")
            continue

        if not question:
            continue

        rag_query(question, vectorstore, answer_chain, use_fusion=use_fusion)


if __name__ == "__main__":
    main()
