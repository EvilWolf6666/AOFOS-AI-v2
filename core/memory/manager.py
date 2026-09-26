# core/memory/manager.py
"""
準 v3 最小記憶切片（《現況》〈🔒 隔離區〉記憶模塊〈準 v3（基準 1）〉〈十二〉）。

做的事只有三件：
  ① 每輪把雙方原句寫進原文存放區（沿用準 v2 的 pre_write → transcript）
  ② 每輪依本輪輸入撈回原句（bigram＋FTS5，top_k＋前後鄰域）
  ③ 撈到的原句附時間差，**追加在 context 尾端**
Working Memory 是滑動視窗（MAX_WORKING_MEMORY 輪），超出的直接滑出——
⭐ 原句已經落地，滑出不會丟任何東西，**所以不需要壓縮**。

🔴 **2026-09-26 本檔整份取代 v1 的 manager.py**（v1 原檔由邪狼移出停用）：
   v1 的兩個注入——`[過去對話摘要]`（Compressed Memory）與 ChromaDB `[相關記憶]`——
   **在這裡不存在**。本檔**不 import rag.py／llm_client.py**。

⚠️ **對外介面與 v1 相同**（agent.py／pipeline.py 一行不改）。
   只為相容而保留、在本檔沒有意義的成員，一律標 [相容殼]：
   `compressed_memory`（恆為空）、`rag`（恆為停用）、`needs_compress`／
   `maybe_compress`（恆為 False）、`compress_on_exit`（只補寫、不壓縮）。

🔴 **執行緒**：DB 連線在**第一次使用時**於當下執行緒開啟，之後只能在同一執行緒用
   （sqlite3 預設 check_same_thread，跨執行緒會直接拋 ProgrammingError——大聲，是對的）。
   目前 agent 的事件迴圈是單一執行緒，符合。

📐 兩軸標註：寫進持久化資料的照目標形（transcript 欄位、clean 欄），
   程式行為照最小可行（每輪撈、N、top_k、只存雙方）——見〈十二〉。
"""

from __future__ import annotations

import os
from datetime import datetime

from config import MAX_WORKING_MEMORY, MEMORY_CONFIG, TEST_MODE
from core.logger import setup_logger, timed
from core.memory.db import connect
from core.memory.pre_write import PreWrite
from core.memory.recall import RecallIndex, check_fts5, format_blocks
from core.memory.transcript import Transcript

logger = setup_logger("MemoryManager")


class _RagDisabled:
    """[相容殼] pipeline 的 /status 會讀 `memory.rag.enabled`。本檔沒有向量 RAG。"""
    enabled = False

    def set_enabled(self, _on: bool) -> None:
        pass


class _Store:
    """一個 DB 檔：連線 ＋ 三個存取物件。連線延後到第一次使用才開（見檔頭「執行緒」）。"""

    def __init__(self, path: str, *, writable: bool) -> None:
        self.path = path
        self.writable = writable
        self._cx = None
        self.pre_write: PreWrite | None = None
        self.transcript: Transcript | None = None
        self.index: RecallIndex | None = None

    def open(self) -> "_Store":
        if self._cx is None:
            self._cx = connect(self.path)
            self.pre_write = PreWrite(self._cx)
            self.transcript = Transcript(self._cx)
            self.index = RecallIndex(self._cx, writable=self.writable)
            if self.writable:
                n = self.index.catch_up()
                if n:
                    logger.info(f"檢索索引補上 {n} 筆：{self.path}")
        return self

    def close(self) -> None:
        if self._cx is not None:
            self._cx.close()
            self._cx = None


def _discard_db(path: str) -> None:
    """整包丟棄一個 DB（含 WAL 的兩個附屬檔）。**呼叫前必須已關閉連線。**"""
    for p in (path, path + "-wal", path + "-shm"):
        if os.path.exists(p):
            os.remove(p)


def _now_iso() -> str:
    # ⚠️ 帶時區：時間差是跨場次算的，裸時間在換時區／夏令時間時會算錯且不報錯。
    return datetime.now().astimezone().isoformat(timespec="seconds")


class MemoryManager:
    def __init__(self, llm=None, *, db_path: str | None = None,
                 test_db_path: str | None = None):
        """`llm` 只為介面相容而收（本檔不呼叫 LLM）。兩個路徑參數只給自測用。"""
        # 🔴 依賴檢查放在建構時、大聲失敗：沒有 FTS5 就不要啟動，
        #    否則會是「對話照常、記憶照寫、永遠撈不回來」——最難發現的那種壞。
        check_fts5()

        self.llm = llm
        self.working_memory: list[dict] = []
        self.compressed_memory: str = ""       # [相容殼] 沒有摘要
        self.rag = _RagDisabled()               # [相容殼]
        self._verbatim_repeat_count = 0

        cfg = MEMORY_CONFIG
        self._daily_path = db_path or cfg["db_path"]
        self._test_path = test_db_path or cfg["test_db_path"]
        self._top_k = cfg["top_k"]
        self._n = cfg["neighbor_n"]
        self._max_terms = cfg["max_query_terms"]
        self._labels = cfg["speaker_labels"]

        self._daily = _Store(self._daily_path, writable=True)
        self._test: _Store | None = None
        self.test_mode = False
        if TEST_MODE:
            self._enter_test()
        logger.info(f"記憶（準 v3 最小切片）：原句寫入 {self.memory_path}，摘要與向量 RAG 注入已停用")

    # ── 測試模式：讀取共用、寫入分流（第八章 C8）───────────────

    @property
    def memory_path(self) -> str:
        """[相容] pipeline 的 /status 會印它。現在指向「寫入」的那個 DB。"""
        return self._test_path if self.test_mode else self._daily_path

    def _enter_test(self) -> None:
        # 進場先丟一次：上一輪若沒正常離開，測試區會殘留，第 N 題就被前面污染。
        _discard_db(self._test_path)
        self._test = _Store(self._test_path, writable=True)
        # 日常區在施測中只讀：不 catch-up、不記命中 ⇒ 整輪施測日常區是靜止的。
        self._daily.close()
        self._daily = _Store(self._daily_path, writable=False)
        self.test_mode = True
        logger.warning(f"TEST_MODE：寫入只進 {self._test_path}，日常區唯讀")

    def _leave_test(self) -> None:
        if self._test is not None:
            self._test.close()
            _discard_db(self._test_path)
            self._test = None
        self._daily.close()
        self._daily = _Store(self._daily_path, writable=True)
        self.test_mode = False

    def set_test_mode(self, on: bool) -> None:
        """切換測試模式。⚠️ Working Memory 會清空（對呼叫端可見的副作用，切換的目的就是隔離）。

        先把還沒寫進去的補寫，**寫進切換前的那一區**，再切。
        """
        if on == self.test_mode:
            return
        self._flush()
        self._enter_test() if on else self._leave_test()
        self.working_memory.clear()
        logger.warning(
            f"測試模式{'開啟' if on else '關閉'}：寫入改為 {self.memory_path}，"
            f"Working Memory 已清空"
            + ("" if on else "，測試區已整包丟棄"))

    def _write_store(self) -> _Store:
        return (self._test if self.test_mode else self._daily).open()

    def _read_stores(self) -> list[_Store]:
        stores = [self._write_store()]
        if self.test_mode and os.path.exists(self._daily_path):
            # 唯讀那一區不存在就不開——開了會在日常區憑空建一個空檔
            stores.append(self._daily.open())
        return stores

    # ── 寫入 ──────────────────────────────────────────────────

    def add_turn(self, user_input: str, assistant_reply: str, *,
                 source: str = "", kind: str = "", situation: str = ""):
        """追加進 Working Memory 並寫進原文存放區。

        source／kind／situation 由 Pipeline 從 Event 拆出來傳入。
        🔴 **本模組刻意不 import core.agent.events**：core/memory/ 是 core/agent/ 的
        下層，下層 import 上層是反向依賴 ⇒ 這裡只收值、只記錄，不認識 Event 型別。

        ⚠️ **三個欄位在本層一律只記錄，不做任何分支判斷。**
        · `kind`（utterance / observation）是封閉值域，日後要分支是合法的，
          但要在有需求時才做。
        · `source`（user_text / user_voice / window，未來還會長）是**開放值域**，
          🔴 **永遠不得拿來做分支**——新增一個 producer 時它會靜默走進 else。
        ⚠️ `situation` 只掛在 **user 那一則**：observation 輪次的 user content 是空
        placeholder，情境正是那個位置該有的東西。
        🔴 **這些欄位不得進入送給模型的 messages**，剝除點在 get_context()。
        """
        self._log_verbatim_repeat(assistant_reply)

        ts = _now_iso()
        meta = {"ts": ts}
        if source:
            meta["source"] = source
        if kind:
            meta["kind"] = kind
        user_turn = {"role": "user", "content": user_input, **meta}
        if situation:
            user_turn["situation"] = situation
        self.working_memory.append(user_turn)
        self.working_memory.append({"role": "assistant", "content": assistant_reply, **meta})

        self._flush()
        self._slide()

    def _flush(self) -> None:
        """把 Working Memory 裡還沒有 serial 的紀錄寫進去。

        流程沿用準 v2：預寫區 append（取號＋落盤）→ 讀出 → 原文存放區 land
        （推進 N）→ 清預寫區 → 補索引。
        ⭐ **失敗不中斷對話，而且會自己重試**：沒拿到 serial 的紀錄留在 WM，
           下一輪再寫；已進預寫區但沒 land 的，下一輪 read_pending 會一起帶出來。
        🔴 但失敗要大聲——連續失敗代表原句根本沒在存，關係連續性是假的。
        """
        pending = [t for t in self.working_memory if "serial" not in t]
        if not pending:
            return
        try:
            with timed("原句寫入"):
                st = self._write_store()
                serials = st.pre_write.append(pending)
                for t, s in zip(pending, serials):
                    t["serial"] = s
                rows, halt = st.pre_write.read_pending(st.transcript.landed_upto)
                if rows:
                    st.transcript.land(rows)
                    st.pre_write.purge_upto(st.transcript.landed_upto)
                if halt:
                    logger.error(f"預寫區中止：{halt}（已落地的部分照常可撈）")
                st.index.catch_up()
        except Exception as e:
            logger.error(
                f"原句寫入失敗，{len(pending)} 則暫留 Working Memory、下一輪重試："
                f"{type(e).__name__}: {e}")

    def _slide(self) -> None:
        """滑動視窗：超過 MAX_WORKING_MEMORY 輪時，丟掉最舊的**已落地**紀錄。

        🔴 **沒有 serial 的一律不丟**（寫入失敗的那些）——丟了就真的沒了。
        """
        limit = MAX_WORKING_MEMORY * 2
        while len(self.working_memory) > limit and "serial" in self.working_memory[0]:
            self.working_memory.pop(0)
        if len(self.working_memory) > limit:
            logger.error(
                f"Working Memory {len(self.working_memory)} 則超過上限 {limit}："
                "最舊的紀錄一直沒寫進去，無法滑出。")

    def _log_verbatim_repeat(self, reply: str) -> None:
        """本輪回應若與 Working Memory 內既有 assistant 內容逐字相同，記一次。

        **純診斷，刻意不改變任何行為**：在日常運行中累積「逐字複製」樣本
        （《現況》12之0）。⚠️ 必須在 append 之前比對，之後比對必定自我命中。
        ⚠️ 只比對完全相同、不做相似度——部分重複另有成因，混進同一個計數會使
        這個數字失去意義。記 test_mode 是因為 `/test on` 下也觸發才是乾淨條件的證據。
        """
        if not reply:
            return
        hit_idx = next(
            (i for i in range(len(self.working_memory) - 1, -1, -1)
             if self.working_memory[i].get("role") == "assistant"
             and self.working_memory[i].get("content") == reply),
            None,
        )
        if hit_idx is None:
            return
        turns_ago = (len(self.working_memory) - hit_idx + 1) // 2
        self._verbatim_repeat_count += 1
        logger.warning(
            f"逐字複製：本輪回應與 {turns_ago} 輪前的回應完全相同"
            f"（本次執行第 {self._verbatim_repeat_count} 次，"
            f"WM {len(self.working_memory) // 2} 輪，test_mode={self.test_mode}）"
            f"：{reply[:40]}")

    # ── 讀取 ──────────────────────────────────────────────────

    @staticmethod
    def _to_message(turn: dict) -> dict:
        """只投影 role／content（白名單）。

        🔴 **這不是防禦性程式碼，是「掛欄位不影響模型」成立的前提**：本地欄位
        （ts／serial／source…）若進了 request body，後端拒收 ⇒ 400；後端忽略 ⇒
        「模型看到的完全不變」變成一句沒驗證過的話。⚠️ 用 turn["role"] 直接取：
        缺 key 代表有人繞過 add_turn 寫進來，要大聲失敗。白名單 ⇒ 新增本地欄位不用改這裡。
        """
        return {"role": turn["role"], "content": turn["content"]}

    def get_context(self, user_input: str = "") -> list[dict]:
        """Working Memory → 撈回的原句（追加尾端）。

        ⚠️ 順序依「變動頻率由低到高」：撈回內容每輪不同，放尾端才不會讓
           WM 落在 prompt cache 斷點之後（prompt cache 比對的是最長共同前綴）。
        ⚠️ user_input 為空（observation 輪次的 placeholder）時不撈——沿用舊行為，
           屬 ① 類未確認項，不在本切片改動。
        """
        with timed("記憶組裝"):
            context = [self._to_message(m) for m in self.working_memory]
            if user_input:
                text = self._recall(user_input)
                if text:
                    context.append({"role": "system", "content": text})
        return context

    def _recall(self, query: str) -> str:
        """撈回失敗不中斷對話，但要大聲。"""
        try:
            stores = self._read_stores()
            wm = {t["serial"] for t in self.working_memory if "serial" in t}
            write_st = stores[0]

            # 各區各撈 top_k，再依 bm25 合併取 top_k。
            # 🟩 兩個 FTS 表的 bm25 分數嚴格說不可直接比（語料統計不同），
            #    施測時才有兩區，日常只有一區——先這樣，量到偏差再改。
            cands = []
            for st in stores:
                excl = wm if st is write_st else frozenset()
                cands += [(score, s, st) for s, score in
                          st.index.search(query, top_k=self._top_k,
                                          max_terms=self._max_terms, exclude=excl)]
            cands.sort(key=lambda c: c[0])
            cands = cands[: self._top_k]
            if not cands:
                return ""

            blocks = []
            for st in stores:
                hits = [s for _, s, owner in cands if owner is st]
                if not hits:
                    continue
                excl = wm if st is write_st else frozenset()
                blocks += st.index.neighborhood(hits, self._n, exclude=excl)
                st.index.mark_recalled(hits)
            blocks.sort(key=lambda b: b[0]["ts"] or "")
            logger.debug(f"撈回 {len(cands)} 則命中、{len(blocks)} 段")
            return format_blocks(blocks, self._labels)
        except Exception as e:
            logger.error(f"取回失敗，本輪不注入記憶：{type(e).__name__}: {e}")
            return ""

    # ── [相容殼] 舊版的壓縮介面 ────────────────────────────────

    def needs_compress(self) -> bool:
        return False

    def maybe_compress(self) -> bool:
        return False

    def compress_on_exit(self) -> None:
        """不壓縮；只把還沒寫進去的補寫一次，然後關連線。"""
        self._flush()
        left = [t for t in self.working_memory if "serial" not in t]
        if left:
            logger.error(f"離場時仍有 {len(left)} 則沒寫進原文存放區，程式結束後消失。")
        if self.test_mode:
            self._leave_test()   # 施測結束整包丟棄
        self._daily.close()

    def save_memory(self) -> None:
        """[相容殼] 每輪已即時落盤，沒有需要另外存的東西。"""
