# core/agent/selftest.py
"""
主迴圈周邊的自測：`python -m core.agent.selftest`

涵蓋：事件佇列（逾時、同來源覆蓋）、使用者輸入讀取器（一次只讀一次、空的不算、讀取壞掉
仍推得出事件、ESC＝/mode t）、語音模型載不起來時留在文字模式、對話視窗只看到一行結果
（原因在 log）、她沒回應時有一行提示、收尾只做一次且不怕某步壞掉、收完會通知、
log 終端輸出（之後才建立的 logger 與第三方套件也走登記的出口、等級照登記的、檔案 log 不受影響）、
惰性載入（import 主迴圈不會拉進語音棧）。

不連推理後端、不開麥克風、不碰 data/。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import types

from core.agent.event_queue import EventQueue
from core.agent.events import Event, OBSERVATION, UTTERANCE
from core.producers.user_input import OnDemandReader, voice_reader

_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _fail
    print(("  ✅ " if ok else "  ❌ ") + name + (f"　— {detail}" if detail and not ok else ""))
    if not ok:
        _fail += 1


def wait_until(pred, timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_queue() -> None:
    print("\n[事件佇列]")
    q = EventQueue()
    t0 = time.monotonic()
    check("等不到回 None（不會無限等）", q.get(timeout=0.05) is None and time.monotonic() - t0 < 1.0)
    check("timeout=0 不等", q.get(timeout=0) is None)
    q.put(Event("w", OBSERVATION, "舊"))
    q.put(Event("u", UTTERANCE, "說話"))
    q.put(Event("w", OBSERVATION, "新"))
    got = [q.get(timeout=0), q.get(timeout=0), q.get(timeout=0)]
    check("同來源 observation 覆蓋且保留排隊位置",
          [e and e.payload for e in got] == ["新", "說話", None], str([e and e.payload for e in got]))
    later: list = []
    th = threading.Thread(target=lambda: later.append(q.get(timeout=2.0)))
    th.start()
    time.sleep(0.05)
    q.put(Event("u", UTTERANCE, "晚到"))
    th.join()
    check("等待中有事件進來立刻醒", later and later[0].payload == "晚到")


class _ScriptedRead:
    """依序吐出預先排好的結果；每次讀取都記下來，並可卡住直到放行。"""

    def __init__(self, results, block: bool = False) -> None:
        self.results = list(results)
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.gate = threading.Event()
        if not block:
            self.gate.set()
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.gate.wait()
        with self._lock:
            self.active -= 1
        r = self.results.pop(0) if self.results else None
        if isinstance(r, Exception):
            raise r
        return r


def test_reader() -> None:
    print("\n[輸入讀取器]")
    q = EventQueue()
    read = _ScriptedRead([None, Event("user_text", UTTERANCE, "第一句")], block=True)
    r = OnDemandReader("t", read, q, on_error=Event("t", UTTERANCE, "quit"))
    time.sleep(0.05)
    check("沒要求就不讀（她輸出時不會先跳出提示）", read.calls == 0)
    for _ in range(5):
        r.request()
    read.gate.set()
    ev = q.get(timeout=2.0)
    check("空的結果直接再讀、不回主迴圈；拿到第一句", ev is not None and ev.payload == "第一句", repr(ev))
    check("重複要求不會疊成多次讀取（同時最多一次）", read.max_active == 1 and read.calls == 2,
          f"max_active={read.max_active} calls={read.calls}")
    time.sleep(0.05)
    check("推完事件就停，等下一次要求", read.calls == 2)

    # 拿到事件後立刻再要：讀取器必須已經回到閒置（先閒置再推事件），否則這次要求會被丟掉
    read.results = [Event("user_text", UTTERANCE, "第二句")]
    r.request()
    ev = q.get(timeout=2.0)
    check("拿到事件後馬上再要，要得到", ev is not None and ev.payload == "第二句", repr(ev))

    read.results = [RuntimeError("麥克風拔掉了")]
    r.request()
    ev = q.get(timeout=2.0)
    check("讀取壞掉仍推出事件（主迴圈不會永遠等）", ev is not None and ev.payload == "quit", repr(ev))

    r.close()
    read.results = [Event("user_text", UTTERANCE, "不該出現")]
    r.request()
    check("close() 之後不再讀", q.get(timeout=0.2) is None)

    print("\n[語音讀取器]")
    q = EventQueue()
    outs = iter(["", "  ", "你好", None])
    fake = types.SimpleNamespace(listen=lambda: next(outs))
    v = voice_reader(q, fake)
    v.request()
    ev = q.get(timeout=2.0)
    check("沒聽到東西就繼續聽；辨識結果包成 user_voice", ev and (ev.source, ev.payload) == ("user_voice", "你好"),
          repr(ev))
    v.request()
    ev = q.get(timeout=2.0)
    check("ESC（listen 回 None）＝ /mode t", ev and ev.payload == "/mode t", repr(ev))
    v.close()


class _dialog:
    """擷取對話視窗（console.emit）的輸出；log 照 dialog_level=None 接上，看會不會混進來。"""

    def __enter__(self) -> list:
        from core import console
        from core import logger as L
        self._console, self._L, self._saved = console, L, console.emit
        shown: list = []
        console.emit = lambda text="", end="\n": shown.append(text)
        L.set_console_writer(console._log_line, None)
        return shown

    def __exit__(self, *exc) -> None:
        self._console.emit = self._saved
        self._L.set_console_writer(None)


def test_agent_parts() -> None:
    """Agent 的模式切換與收尾。不建真的 Agent（要推理後端），只裝上這兩件事用到的欄位。"""
    print("\n[語音模型載不起來]")
    from core.agent.agent import Agent
    from core.agent.state import AgentState

    a = Agent.__new__(Agent)
    a.state = AgentState(mode="t")
    a._text = OnDemandReader("t", lambda: None, a.state.event_queue, on_error=Event("t", UTTERANCE, "quit"))
    a._voice = None
    a._voice_input = None

    bad = types.ModuleType("core.voice_input")

    class _Broken:
        def __init__(self, *a, **k):
            raise OSError("找不到麥克風")

    bad.VoiceInput = _Broken
    saved = sys.modules.get("core.voice_input")
    sys.modules["core.voice_input"] = bad
    with _dialog() as shown:
        try:
            a.state.switch_to_voice()
            got = a._reader_for_mode()
        finally:
            if saved is None:
                sys.modules.pop("core.voice_input", None)
            else:
                sys.modules["core.voice_input"] = saved
    check("退回文字讀取器、模式改回文字", got is a._text and a.state.mode == "t")
    check("對話視窗只有一行結果，原因（例外內容）不在對話視窗",
          len(shown) == 1 and "語音模式啟動失敗" in shown[0] and "找不到麥克風" not in shown[0], str(shown))

    print("\n[她沒回應]")
    from core.agent.pipeline import TurnResult
    done: list[str] = []
    a.pipeline = types.SimpleNamespace(is_command=lambda t: False, handle_turn=lambda e: TurnResult(reply=""),
                                       finalize_turn=lambda: done.append("收尾"))
    with _dialog() as shown:
        a._handle_utterance(Event("user_text", UTTERANCE, "哈囉"))
    check("對話視窗有一行提示（不然只看到提示字元又出現）、收尾照做",
          len(shown) == 1 and "沒有回應" in shown[0] and done == ["收尾"], f"{shown} {done}")

    print("\n[收尾]")
    calls: list[str] = []

    def step(name, boom=False):
        def f():
            calls.append(name)
            if boom:
                raise RuntimeError(name)
        return f

    a.memory = types.SimpleNamespace(close=step("記憶", boom=True))
    a.window_watcher = types.SimpleNamespace(stop=step("視窗"))
    a._text = types.SimpleNamespace(close=step("文字"))
    a._voice = types.SimpleNamespace(close=step("語音"))
    a._voice_input = types.SimpleNamespace(close=step("麥克風"))
    a.vtube = types.SimpleNamespace(disconnect=step("VTube"))
    a._closed = False
    a._shutdown_done = threading.Event()
    a._shutdown()
    a._shutdown()
    check("記憶排第一、某步壞掉其餘照做、只做一次",
          calls == ["記憶", "視窗", "文字", "語音", "麥克風", "VTube"], str(calls))
    check("收完會通知（關視窗時系統只給幾秒，要等的就是這個）", a._shutdown_done.is_set())


def test_log_sink() -> None:
    print("\n[log 終端輸出]")
    import logging

    from core import logger as L

    early = L.setup_logger("SelftestEarly")
    got: list[str] = []
    L.set_console_writer(got.append, "WARNING")
    try:
        late = L.setup_logger("SelftestLate")   # 登記之後才建立（同 TIMING）
        early.warning("早")
        late.warning("晚")
        late.debug("細")
    finally:
        L.set_console_writer(None)
    check("登記前後建立的 logger 都走登記的出口", any("早" in x for x in got) and any("晚" in x for x in got),
          str(got))
    check("低於登記等級的不印", not any("細" in x for x in got), str(got))
    none: list[str] = []
    L.set_console_writer(none.append, None)
    try:
        early.error("不該印")
    finally:
        L.set_console_writer(None)
    check("等級 None＝一筆都不印（對話視窗的預設）", none == [], str(none))
    check("還原後全部印（tools/ 腳本的終端）", L._CONSOLE_SINK.level == logging.DEBUG)

    import warnings
    third: list[str] = []
    L.set_console_writer(third.append, "WARNING")
    try:
        logging.getLogger("some.thirdparty").warning("第三方警告")
        logging.getLogger("some.thirdparty").info("第三方資訊")
        warnings.warn("套件警告", UserWarning, stacklevel=1)
    finally:
        L.set_console_writer(None)
    check("第三方套件的 log 與 warnings 走同一個出口（只收警告以上）",
          any("第三方警告" in x for x in third) and any("套件警告" in x for x in third)
          and not any("第三方資訊" in x for x in third), str(third))
    files = [h for h in early.handlers if isinstance(h, logging.FileHandler)]
    check("檔案 log 不受影響（仍寫檔、不寫進終端出口）",
          len(files) == 1 and files[0].stream is not L._CONSOLE_SINK and len(got) == 2, str(got))


def test_lazy_import() -> None:
    print("\n[惰性載入]")
    code = ("import sys, core.agent.agent; "
            "heavy = [m for m in ('core.voice_input', 'faster_whisper', 'sounddevice', 'torch') if m in sys.modules]; "
            "print(','.join(heavy))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       encoding="utf-8", errors="replace", check=False)
    if p.returncode != 0:
        check("import core.agent.agent 成功", False, p.stderr.strip().splitlines()[-1] if p.stderr else "")
        return
    heavy = p.stdout.strip()
    check("import 主迴圈不會拉進語音棧（文字模式不付語音模型的載入成本）", heavy == "", heavy)


def main() -> int:
    test_queue()
    test_reader()
    test_agent_parts()
    test_log_sink()
    test_lazy_import()
    print("\n全部通過" if _fail == 0 else f"\n❌ {_fail} 項失敗")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
