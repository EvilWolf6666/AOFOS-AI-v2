# main.py
"""啟動入口。中央控制/流程控制已移交 core/agent/，這裡只建立 Agent 並驅動主迴圈。"""

from __future__ import annotations

# ⚠️ 必須排在所有其他 import 之前，且不可因為「看起來沒用到」而被移除。
# config 頂部會 load_dotenv() 並設定 HF_HUB_OFFLINE，而 huggingface_hub 是在
# import 當下把這些環境變數讀成模組常數——只要任何一個會連帶拉進 huggingface_hub
# 的模組（sentence-transformers、faster-whisper）先被 import，設定就來不及了，
# 而且不會報錯，只會靜默地變回連網。
#
# 這一行取代了原本寫在此處的 load_dotenv()：環境設定現已集中於 config.py，
# 因為 tools/ 底下的腳本直接 import config、完全不經過 main.py（《歸檔》三十一之4）。
import config  # noqa: F401

from core.agent.agent import Agent  # noqa: E402


def main() -> None:
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()