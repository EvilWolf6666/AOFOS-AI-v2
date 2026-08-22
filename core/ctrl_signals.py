# core/ctrl_signals.py
"""
控制流訊號（Control Stream Signal）。

跨模組傳遞「流程該轉向」的例外——不是錯誤，是控制流。使用者要求離開語音模式、
未來的聊天室斷線、TTS 被中斷，都屬這一類：發生時沒有東西壞掉，只是主迴圈
應該改走另一條路。

⚠️ **本檔不得 import 任何重量套件，也不得 import 專案內任何會這麼做的模組。**
   它存在的唯一理由，是讓上層能在不付出載入成本的前提下 catch 這些訊號。

放在 core/ 而非 core/agent/：voice_input.py 位於 core/，是 agent 的下層，
下層 import 上層會造成反向依賴。同 core/agent/events.py。
"""

from __future__ import annotations


class VoiceModeExit(Exception):
    """使用者按下退出鍵（預設 ESC）時拋出，呼叫端需 catch 並切回文字模式或結束程式。"""
    pass