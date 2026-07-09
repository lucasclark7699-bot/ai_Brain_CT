"""
AI 监控仪 · 守护进程启动器
============================
用途：用项目隔离 venv 里的 streamlit 启动应用；若进程意外退出，自动重启。
这样即使偶发崩溃，应用也会立刻恢复，不会出现“跑着跑着中断”的情况。

用法（在 ai_Brain_CT 目录下）：
    python run.py
然后浏览器打开 http://localhost:8501
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# 项目隔离 venv（不污染系统 Python）。如路径变动请按需修改。
VENV_DIR = r"C:\Users\5420\.workbuddy\binaries\python\envs\ai_brain_ct"
STREAMLIT = os.path.join(VENV_DIR, "Scripts", "streamlit.exe")
PORT = "8501"


def main():
    if not os.path.exists(STREAMLIT):
        print(f"[错误] 未找到 venv 中的 streamlit：{STREAMLIT}\n请先安装依赖（pip install -r requirements.txt）。", file=sys.stderr)
        sys.exit(1)

    print(f"AI 监控仪守护启动器已就绪，端口 {PORT}。Ctrl+C 退出。")
    while True:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 启动 Streamlit ...")
        try:
            subprocess.run(
                [STREAMLIT, "run", "app.py",
                 "--server.port", PORT,
                 "--server.headless", "true"],
                cwd=HERE,
                check=False,
            )
        except Exception as e:  # 启动本身抛错（极少见）
            print(f"[{time.strftime('%H:%M:%S')}] 启动异常：{e}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 进程已退出，2 秒后自动重启 ...")
        time.sleep(2)


if __name__ == "__main__":
    main()
