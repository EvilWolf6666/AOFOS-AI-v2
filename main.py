# main.py
"""
啟動入口。所有中央控制/流程控制職責已移交 core/agent/。
main.py 本身不再持有任何模組或邏輯，只負責建立 Agent 並驅動主迴圈。
"""

from __future__ import annotations
from core.agent.agent import Agent


def main() -> None:
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()
