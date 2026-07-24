# main.py
"""
啟動入口。所有中央控制/流程控制職責已移交 core/agent/。
main.py 本身不再持有任何模組或邏輯，只負責建立 Agent 並驅動主迴圈。
"""

from __future__ import annotations

from dotenv import load_dotenv
# 必須在 Agent() 實際初始化（尤其 RAGManager 載入 sentence-transformers 模型）之前執行，
# 讓 HF_TOKEN 等環境變數在任何會用到它們的模組運作前就已經寫進 os.environ。
load_dotenv()

from core.agent.agent import Agent


def main() -> None:
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()