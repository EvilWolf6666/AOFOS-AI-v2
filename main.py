# main.py
"""啟動入口。中央控制/流程控制已移交 core/agent/，這裡只建立 Agent 並驅動主迴圈。"""

from __future__ import annotations

from dotenv import load_dotenv

# ⚠️ 必須在 import Agent 之前執行：RAGManager 載入 sentence-transformers 時
# 就會讀取 HF_TOKEN 等環境變數，晚一步就來不及。
load_dotenv()

from core.agent.agent import Agent  # noqa: E402


def main() -> None:
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()