# core/ctrl_stream_signal.py
"""
控制流訊號（Control Stream Signal）。

跨模組傳遞「流程該轉向」的例外——不是錯誤，是控制流。使用者要求離開語音模式、
未來的聊天室斷線、TTS 被中斷，都屬這一類：發生時沒有東西壞掉，只是主迴圈
應該改走另一條路。

⚠️ **本檔不得 import 任何重量套件，也不得 import 專案內任何會這麼做的模組。**
   它存在的唯一理由就是讓上層能在「不付出載入成本」的前提下 catch 這些訊號。
   一旦這裡多了一行 `import torch` 之類的東西，整個檔案就失去意義。

放在 core/ 而非 core/agent/：voice_input.py 位於 core/，是 agent 的下層，
下層 import 上層會造成反向依賴。理由同 core/agent/events.py 檔頭所述。

── 為什麼會有這個檔（2026-08-06，見《決策脈絡歸檔》三十二章）──
原本 VoiceModeExit 定義在 core/voice_input.py，而 agent.py 為了 catch 它必須
`from core.voice_input import VoiceModeExit`——這一行在模組層級把整個語音棧
（faster_whisper ~6s、torch ~4s）拉了進來。實測 `import core.voice_input`
需 5.7 秒，而文字模式從頭到尾用不到它。Agent._voice_input_factory() 的
「惰性建立」docstring 因此是假的：函式內的 import 只是查 sys.modules。
"""

from __future__ import annotations


class VoiceModeExit(Exception):
    """使用者按下退出鍵（預設 ESC）時拋出，呼叫端需 catch 並切回文字模式或結束程式。"""
    pass