"""启动器：启动 Streamlit 服务并自动打开浏览器"""
import subprocess
import sys
import threading
import webbrowser
import time
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def open_browser():
    time.sleep(4)
    url = "http://localhost:8501"
    try:
        os.startfile(url)
    except Exception:
        webbrowser.open(url)


def main():
    print("正在启动 RAG 知识库问答服务...")
    print("浏览器即将打开: http://localhost:8501")
    print("如果没有自动弹出，请手动访问上述地址")
    print("按 Ctrl+C 停止服务")
    print("-" * 50)

    threading.Thread(target=open_browser, daemon=True).start()

    streamlit = os.path.join(PROJECT_DIR, ".venv", "Scripts", "streamlit.exe")
    subprocess.run(
        [streamlit, "run", "web_ui.py", "--server.headless", "false"],
        cwd=PROJECT_DIR,
        stdin=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
