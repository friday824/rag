"""RAG 检索增强生成系统"""
import os
import sys
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==================== 1. API Client 配置 ====================

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY")

if not DEEPSEEK_KEY or not DASHSCOPE_KEY:
    raise RuntimeError(
        "缺少 API 密钥，请在项目根目录的 .env 文件中配置 "
        "DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY（参考 .env.example）"
    )

deepseek_client = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com/v1",
)

dashscope_client = OpenAI(
    api_key=DASHSCOPE_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ==================== 2. 文档切分 ====================

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


def chunk_text_langchain(text, chunk_size=200, chunk_overlap=40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "？", "！", "；", "，", " ", ""],
    )
    return splitter.split_text(text)


def chunk_text_fallback(text, chunk_size=200, chunk_overlap=40):
    raw = re.split(r"(。|？|！|\…\…)", text)
    sentences = []
    for i in range(0, len(raw), 2):
        sent = raw[i]
        if i + 1 < len(raw):
            sent += raw[i + 1]
        sentences.append(sent)

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current:
                chunks.append(current)
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:] + sent
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text, chunk_size=200, chunk_overlap=40):
    if not text or not text.strip():
        return []
    if HAS_LANGCHAIN:
        return chunk_text_langchain(text, chunk_size, chunk_overlap)
    else:
        return chunk_text_fallback(text, chunk_size, chunk_overlap)


# ==================== 3. 向量化工具 ====================

def get_embedding(text):
    text = text.replace("\n", " ")
    resp = dashscope_client.embeddings.create(
        model="text-embedding-v3",
        input=[text],
        dimensions=1024,
        encoding_format="float",
    )
    return resp.data[0].embedding


def get_embeddings_batch(texts):
    batch_size = 10
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = dashscope_client.embeddings.create(
            model="text-embedding-v3",
            input=batch,
            dimensions=1024,
            encoding_format="float",
        )
        all_embeddings.extend([d.embedding for d in resp.data])
        print(f"  嵌入批次 {i // batch_size + 1}: {len(batch)} 条文本 -> {len(resp.data)} 个向量")
    return np.array(all_embeddings, dtype=np.float64)


def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


# ==================== 4. VectorStore 类 ====================

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self._norms = None

    def add_texts(self, texts):
        if not texts:
            return
        new_emb = get_embeddings_batch(texts)
        self.chunks.extend(texts)
        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])
        self._norms = np.linalg.norm(self.embeddings, axis=1)

    def search(self, query, top_k=3):
        if self.embeddings is None:
            return []
        q = np.array(get_embedding(query), dtype=np.float64)
        q_norm = q / np.linalg.norm(q)
        similarities = np.dot(self.embeddings, q_norm) / self._norms
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            {"chunk": self.chunks[i], "index": int(i), "similarity": float(similarities[i])}
            for i in top_indices if similarities[i] > 0
        ]

    def __len__(self):
        return len(self.chunks)


# ==================== 5. 生成模块 ====================

def generate_answer(question, retrieved_chunks, model="deepseek-chat"):
    context_parts = []
    for i, c in enumerate(retrieved_chunks):
        context_parts.append(f"[参考{i + 1}] {c['chunk']}")
    context = "\n\n".join(context_parts)

    system_prompt = "你是一个知识助手。请根据以下参考资料回答用户的问题。如果参考资料中包含相关信息，请准确回答。如果参考资料中没有相关信息，请如实告知用户，不要编造内容。"

    try:
        resp = deepseek_client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"参考资料：\n{context}\n\n用户问题：{question}\n\n请基于参考资料回答："},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"生成回答时出错: {e}"


def rag_query(question, vector_store, top_k=3):
    print(f"\n问题: {question}")
    print(f"正在检索最相关的 {top_k} 条文本块...")

    results = vector_store.search(question, top_k=top_k)

    if not results:
        print("未检索到相关文本块。")
        return

    print(f"\n--- 检索到的 {len(results)} 条相关文本块 ---")
    for r in results:
        chunk_preview = r['chunk'].replace("\n", " ")[:100]
        print(f"[块 {r['index']}] 相似度: {r['similarity']:.4f}")
        print(f"  内容: {chunk_preview}...")
        print()

    print("正在生成回答...\n")
    answer = generate_answer(question, results)
    print("--- AI 回答 ---")
    print(answer)
    print("-" * 60)


# ==================== 6. 交互式入口 ====================

def load_knowledge_base():
    print("=" * 60)
    print("RAG 检索增强生成系统")
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
        for encoding in ["utf-8", "gbk"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            print("无法读取文件，请确认文件编码为 UTF-8 或 GBK。")
            sys.exit(1)
        print(f"已读取文件，共 {len(text)} 个字符。")
        return text

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
        return text

    else:
        print("无效选项。")
        sys.exit(1)


def main():
    text = load_knowledge_base()

    print("\n正在切分文本...")
    chunks = chunk_text(text)
    if not chunks:
        print("文本切分后为空，请检查输入内容。")
        sys.exit(1)
    print(f"共切分为 {len(chunks)} 个文本块。")

    print(f"\n正在向量化 {len(chunks)} 条文本块...")
    store = VectorStore()
    store.add_texts(chunks)
    print(f"向量存储完成。共 {len(store)} 条文本块, 向量维度: {store.embeddings.shape[1]}")

    print("\n" + "=" * 60)
    print("知识库已就绪，开始问答交互。")
    print("输入 'exit' 或 'quit' 退出程序。")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        if not question:
            continue

        rag_query(question, store)


if __name__ == "__main__":
    main()
