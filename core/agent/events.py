# core/agent/events.py
"""
事件型別定義。

所有進入主迴圈的東西——使用者打字、語音輸入、螢幕/遊戲狀態變化、聊天室訊息
——一律包成 Event 推進同一個佇列，由 Agent 在唯一的取用點取出後分流。

型別放在獨立檔案而非 state.py 或 agent.py：Event 由 Agent 產生、由 Pipeline
消費，放在任一邊都會使另一邊反向 import。
"""

from __future__ import annotations

from dataclasses import dataclass


# kind 決定注入方式，兩者不可互換：
#   "utterance"   有人在說話。payload 進 user 欄位，走 handle_turn()
#   "observation" 環境狀態變化，沒有人在說話。payload 是情境描述，怎麼送看
#                 config.EVENT_CONFIG["observation_delivery"]；走 handle_observation()
#
# 把遊戲事件當 utterance 送出，模型會把「玩家血量剩 10%」誤認為邪狼說的話。
UTTERANCE = "utterance"
OBSERVATION = "observation"


@dataclass(frozen=True)
class Event:
    """
    source: 產生者標籤，供 log 與分流輔助判斷
            （"user_text" / "user_voice" / "window" / 未來的 "screen" / "chat"）
    kind:   UTTERANCE 或 OBSERVATION
    payload: utterance 為輸入原文；observation 為情境描述文字
    """

    source: str
    kind: str
    payload: str
