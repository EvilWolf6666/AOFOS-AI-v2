# main.py
"""啟動入口。中央控制/流程控制已移交 core/agent/，這裡只建立 Agent 並驅動主迴圈。"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
# 需在 Agent() 初始化（尤其 RAGManager 載入 sentence-transformers）前執行，
# 確保 HF_TOKEN 等環境變數在相關模組運作前已寫入 os.environ

from core.agent.agent import Agent


def main() -> None:
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()