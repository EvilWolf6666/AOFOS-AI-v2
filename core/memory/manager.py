# core/memory/manager.py
"""
準 v3 記憶（《現況》〈🔒 隔離區〉記憶模塊〈準 v3（基準 1）〉〈十二〉）。

做的事：
  ① 每輪把雙方原句寫進原文存放區（沿用準 v2 的 pre_write → transcript）
  ② 每輪依本輪輸入撈回原句——兩路：bigram＋FTS5（字面）與向量（語意，背景載入），
     以 RRF 合併取 top_k，每則帶前後鄰域（組合 6）
  ③ 本場開頭幾輪另帶「狀態」與「上一場」（組合 7＋組合 9 甲案事件層）
  ④ 全部包在一個 `⟦記憶⟧…⟦/記憶⟧` system 區塊裡，**追加在本輪輸入之前**；
     區塊第一行自帶說明（prompt 版，不訓練、不改人格 prompt）
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

🆕 2026-09-30 **準 v4 疊加在本檔**（《現況》〈🔒 隔離區〉記憶模塊〈準 v4〉）：
   · 寫入：add_turn 多寫 speaker_id；落地後通知背景判斷（core/memory/judge.py）建主題。
   · 讀取：主題向量比對（core/memory/topic_index.py）→ 注入最新有效版本的主題句。
   · 準 v3 的撈原句／開場以 config 開關可逆關閉（v3_recall／v3_opening），程式碼保留。
   🔺 上面「本檔不 import llm_client.py」只對準 v3 成立：準 v4 的背景判斷要打 LLM，
     這裡建 BackgroundLLM（可取消的背景專用 client），對話用的 client 仍不經本檔。

📐 兩軸標註：寫進持久化資料的照目標形（transcript 欄位、clean 欄），
   程式行為照最小可行（每輪撈、N、top_k、只存雙方）——見〈十二〉。
"""

from __future__ import annotations

import os
from datetime import datetime

from config import MAX_WORKING_MEMORY, MEMORY_CONFIG, TEST_MODE
from core.llm_client import BackgroundLLM
from core.logger import setup_logger, timed
from core.memory.db import connect
from core.memory.embed import Embedder, VectorSearch, VectorWorker, ensure_schema
from core.memory.episode import EpisodeIndex, time_bucket
from core.memory.judge import BackgroundJudge
from core.memory.pre_write import PreWrite
from core.memory.recall import RecallIndex, check_fts5, format_blocks
from core.memory.topic_index import TopicSearch
from core.memory.transcript import Transcript

logger = setup_logger("MemoryManager")

_DEFAULT = object()   # add_turn 的 user_speaker 預設值（None 有意義：observation 輪次沒人說話）


class _RagDisabled:
    """[相容殼] pipeline 的 /status 會讀 `memory.rag.enabled`。本檔沒有向量 RAG。"""
    enabled = False

    def set_enabled(self, _on: bool) -> None:
        pass


class _Store:
    """一個 DB 檔：連線 ＋ 三個存取物件。連線延後到第一次使用才開（見檔頭「執行緒」）。"""

    def __init__(self, path: str, *, writable: bool, emb: Embedder | None = None,
                 ep_cfg: dict | None = None, v3_vec: bool = True, v4: bool = False) -> None:
        self.path = path
        self._v3_vec = v3_vec
        self._v4 = v4
        self.tsearch: TopicSearch | None = None
        self.writable = writable
        self._emb = emb
        self._ep_cfg = ep_cfg or {}
        self._cx = None
        self.pre_write: PreWrite | None = None
        self.transcript: Transcript | None = None
        self.index: RecallIndex | None = None
        self.episodes: EpisodeIndex | None = None
        self.vsearch: VectorSearch | None = None
        self.session_start = 1   # 本場第一句的 serial；之前的都是「以前的場次」

    def open(self) -> "_Store":
        if self._cx is None:
            self._cx = connect(self.path)
            self.pre_write = PreWrite(self._cx)
            self.transcript = Transcript(self._cx)
            self.index = RecallIndex(self._cx, writable=self.writable)
            self.episodes = EpisodeIndex(self._cx, writable=self.writable)
            self.session_start = self.transcript.landed_upto + 1
            if self.writable:
                n = self.index.catch_up()
                if n:
                    logger.info(f"檢索索引補上 {n} 筆：{self.path}")
                self.rebuild_episodes()
                if self._emb is not None and self._v3_vec:
                    ensure_schema(self._cx)
            if self._emb is not None and self._v3_vec:
                self.vsearch = VectorSearch(self._cx, self._emb)
            if self._emb is not None and self._v4:
                self.tsearch = TopicSearch(self._cx, self._emb)
        return self

    def rebuild_episodes(self) -> None:
        """事件層整張重建（確定性、不經 LLM）。失敗只記錄，不擋任何事。"""
        if not (self.writable and self._ep_cfg):
            return
        try:
            self.episodes.build(gap_minutes=self._ep_cfg["gap"],
                                n_keywords=self._ep_cfg["kw"], n_reps=self._ep_cfg["reps"])
        except Exception as e:  # noqa: BLE001
            logger.error(f"事件層重建失敗（不影響原句與檢索）：{type(e).__name__}: {e}")

    def close(self) -> None:
        if self._cx is not None:
            self._cx.close()
            self._cx = None


def _discard_db(path: str) -> None:
    """整包丟棄一個 DB（含 WAL 的兩個附屬檔）。**呼叫前必須已關閉所有連線。**

    ⚠️ 刪不掉（Windows 上有別的連線開著）時大聲、但不拋：測試模式切換不該因此崩潰；
       殘留的測試區在下次 `/test on` 進場時會再丟一次。
    """
    for p in (path, path + "-wal", path + "-shm"):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError as e:
                logger.error(f"測試區檔案刪不掉（仍有連線開著？），殘留到下次進場再丟：{p}：{e}")


def _now_iso() -> str:
    # ⚠️ 帶時區：時間差是跨場次算的，裸時間在換時區／夏令時間時會算錯且不報錯。
    return datetime.now().astimezone().isoformat(timespec="seconds")


class MemoryManager:
    def __init__(self, llm=None, *, db_path: str | None = None,
                 test_db_path: str | None = None, embed_factory=None,
                 embed_enabled: bool | None = None, bg_llm=None, cfg_override: dict | None = None):
        """`llm`（對話用 client）只為介面相容而收，本檔不用它。
        其餘參數只給自測用（bg_llm＝指向假伺服器的 BackgroundLLM、cfg_override 蓋 MEMORY_CONFIG）。"""
        # 🔴 依賴檢查放在建構時、大聲失敗：沒有 FTS5 就不要啟動，
        #    否則會是「對話照常、記憶照寫、永遠撈不回來」——最難發現的那種壞。
        check_fts5()

        self.llm = llm
        self.working_memory: list[dict] = []
        self.compressed_memory: str = ""       # [相容殼] 沒有摘要
        self.rag = _RagDisabled()               # [相容殼]
        self._verbatim_repeat_count = 0

        cfg = {**MEMORY_CONFIG, **(cfg_override or {})}
        self._cfg = cfg
        self._v3_recall, self._v3_opening = cfg["v3_recall"], cfg["v3_opening"]
        self._v4 = cfg["v4_enabled"]
        self._spk = cfg["speaker_ids"]
        self._daily_path = db_path or cfg["db_path"]
        self._test_path = test_db_path or cfg["test_db_path"]
        self._top_k = cfg["top_k"]
        self._n = cfg["neighbor_n"]
        self._max_terms = cfg["max_query_terms"]
        self._labels = cfg["speaker_labels"]
        self._open, self._close, self._note = cfg["block_open"], cfg["block_close"], cfg["block_note"]
        self._carry = cfg["carry_rounds"]
        self._opening_turns = cfg["opening_turns"]
        self._ep_cfg = {"gap": cfg["episode_gap_minutes"], "kw": cfg["episode_keywords"],
                        "reps": cfg["episode_reps"]}
        self._vec_cfg = (cfg["embed_candidates"], cfg["embed_min_sim"], cfg["rrf_k"])
        self._session_turns = 0
        self._opening_cache: tuple[str, set[int]] | None = None

        # 組合 6：向量那一路在背景載入，載好之前只用字面檢索（不擋啟動、不擋對話）
        self._emb: Embedder | None = None
        self._worker: VectorWorker | None = None
        if cfg["embed_enabled"] if embed_enabled is None else embed_enabled:
            self._emb = Embedder(cfg["embed_model"], factory=embed_factory)
            self._emb.start()
            if self._v3_recall:   # 原句向量只給準 v3 撈原句用；關了就不補算
                self._worker = VectorWorker(self._emb)
        elif self._v4:
            logger.warning("準 v4 開著但向量關閉：主題照建，但讀取端撈不到任何主題")

        # 準 v4 背景判斷（乙：同一顆對話模型，閒置時跑、一開口就取消）
        self._judge: BackgroundJudge | None = None
        if self._v4:
            self._judge = BackgroundJudge(
                bg_llm or BackgroundLLM(params=cfg["judge_params"]), self._emb, cfg)

        self._daily = self._new_store(self._daily_path, writable=True)
        self._test: _Store | None = None
        self.test_mode = False
        if TEST_MODE:
            self._enter_test()
        if self._worker is not None:
            self._worker.notify(self.memory_path)
        self._sync_judge()
        paths = [n for n, on in (("準v3撈原句", self._v3_recall), ("準v3開場", self._v3_opening),
                                 ("準v4主題", self._v4)) if on]
        logger.info(f"記憶：原句寫入 {self.memory_path}；讀取路徑＝{'、'.join(paths) or '（全關）'}"
                    f"{'' if self._emb else '（向量已關閉）'}")

    def _sync_judge(self) -> None:
        """背景判斷的對象跟著寫入區走；施測中預設停用（日常區必須靜止、不搶推論槽）。"""
        if self._judge is None:
            return
        if self.test_mode and not self._cfg["judge_in_test_mode"]:
            self._judge.set_path(None)
        else:
            self._judge.set_path(self.memory_path)

    # ── 背景工作的讓路（Pipeline 呼叫）──────────────────────────

    def bg_pause(self) -> None:
        """輪次開始：背景判斷立刻讓出推論槽（取消進行中的背景請求）。"""
        if self._judge is not None:
            self._judge.pause()

    def bg_resume(self) -> None:
        """輪次收尾（輸出之後）：背景判斷重新開始算閒置時間。"""
        if self._judge is not None:
            self._judge.resume()

    def _new_store(self, path: str, *, writable: bool) -> _Store:
        return _Store(path, writable=writable, emb=self._emb, ep_cfg=self._ep_cfg,
                      v3_vec=self._v3_recall, v4=self._v4)

    # ── 測試模式：讀取共用、寫入分流（第八章 舊C8（已結案））───────────────

    @property
    def memory_path(self) -> str:
        """[相容] pipeline 的 /status 會印它。現在指向「寫入」的那個 DB。"""
        return self._test_path if self.test_mode else self._daily_path

    def _enter_test(self) -> None:
        # 進場先丟一次：上一輪若沒正常離開，測試區會殘留，第 N 題就被前面污染。
        if self._worker is not None:
            self._worker.release(self._test_path)
        _discard_db(self._test_path)
        self._test = self._new_store(self._test_path, writable=True)
        # 日常區在施測中只讀：不 catch-up、不記命中、不補向量 ⇒ 整輪施測日常區是靜止的。
        self._daily.close()
        self._daily = self._new_store(self._daily_path, writable=False)
        self.test_mode = True
        logger.warning(f"TEST_MODE：寫入只進 {self._test_path}，日常區唯讀")

    def _leave_test(self) -> None:
        if self._test is not None:
            self._test.close()
            if self._worker is not None:
                self._worker.release(self._test_path)
            _discard_db(self._test_path)
            self._test = None
        self._daily.close()
        self._daily = self._new_store(self._daily_path, writable=True)
        self.test_mode = False

    def set_test_mode(self, on: bool) -> None:
        """切換測試模式。⚠️ Working Memory 會清空（對呼叫端可見的副作用，切換的目的就是隔離）。

        先把還沒寫進去的補寫，**寫進切換前的那一區**，再切。
        """
        if on == self.test_mode:
            return
        self._flush()
        if self._judge is not None:
            self._judge.detach()   # 先放開 DB 再切：進場時日常區要靜止、離場時測試區要能刪
        self._enter_test() if on else self._leave_test()
        self._sync_judge()
        self.working_memory.clear()
        self._session_turns = 0
        self._opening_cache = None
        logger.warning(
            f"測試模式{'開啟' if on else '關閉'}：寫入改為 {self.memory_path}，"
            f"Working Memory 已清空"
            + ("" if on else "，測試區已整包丟棄"))

    def has_history(self) -> bool:
        """原句存放區（讀取範圍內）有沒有任何一句。給啟動訊息用。

        ⚠️ 讀失敗時回 False 並記 ERROR——啟動訊息印錯只是顯示問題，不該擋啟動。
        """
        try:
            return any(st.transcript.count() > 0 for st in self._read_stores())
        except Exception as e:
            logger.error(f"讀不到原句存放區，啟動訊息當作無歷史：{type(e).__name__}: {e}")
            return False

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
                 source: str = "", kind: str = "", situation: str = "",
                 user_speaker=_DEFAULT):
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

        `user_speaker`（準 v4）：user 那一則的 speaker_id。預設＝config 的使用者 ID；
        observation 輪次由 Pipeline 明傳 None（沒有人說話）——由呼叫端決定，本層不看 kind 分支。
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
        uspk = self._spk["user"] if user_speaker is _DEFAULT else user_speaker
        if uspk:
            user_turn["speaker_id"] = uspk
        self.working_memory.append(user_turn)
        self.working_memory.append({"role": "assistant", "content": assistant_reply,
                                    "speaker_id": self._spk["assistant"], **meta})
        self._session_turns += 1

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
            if self._worker is not None:
                self._worker.notify(st.path)
            if self._judge is not None:
                self._judge.notify()
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
        """Working Memory → 一個 `⟦記憶⟧` 區塊（追加在尾端、本輪輸入之前）。

        ⚠️ 順序依「變動頻率由低到高」：記憶區塊每輪不同，放尾端才不會讓
           WM 落在 prompt cache 斷點之後（prompt cache 比對的是最長共同前綴）。
        ⚠️ user_input 為空（observation 輪次的 placeholder）時不做檢索——沿用舊行為；
           但本場開頭的「狀態／上一場」照帶（它不依賴輸入）。
        """
        with timed("記憶組裝"):
            context = [self._to_message(m) for m in self.working_memory]
            sections, shown = [], set()
            if self._v3_opening and self._session_turns < self._opening_turns:
                opening, shown = self._opening()
                if opening:
                    sections.append(opening)
            if user_input and self._v3_recall:
                rel = self._recall(user_input, extra_exclude=shown)
                if rel:
                    sections.append(rel)
            if user_input and self._v4:
                top = self._topics(user_input)
                if top:
                    sections.append(top)
            if sections:
                context.append({"role": "system", "content": "\n".join(
                    [self._open, self._note, *sections, self._close])})
        return context

    def _opening(self) -> tuple[str, set[int]]:
        """組合 7（狀態）＋組合 9 甲案（上一場的關鍵詞、代表句、最後幾輪）。

        ⭐ 本場只算一次、之後原樣重用 ⇒ 開頭幾輪這段字不變（對 prompt cache 友善），
           且不依輸入——問「上次聊甚麼」這種沒有內容詞的話也有東西可用。
        ⭐ 時間只給口語的時間感（`time_bucket`），不給日期與天數，避免她照唸。
        回傳（文字, 已經帶出的 serial）——後者讓同一輪的檢索不重複帶同一句。
        """
        if self._opening_cache is not None:
            return self._opening_cache
        text, shown = "", set()
        try:
            if os.path.exists(self._daily_path) or not self.test_mode:
                st = self._daily.open()
                ep = st.episodes.last_before(st.session_start)
                if ep is not None:
                    bucket = time_bucket(ep["end_ts"])
                    today = datetime.now().astimezone().date().isoformat()
                    n_today = st.episodes.count_on(today)
                    state = f"上次聊天是{bucket}" if bucket else "有上次聊天的紀錄"
                    if n_today and bucket in ("剛剛", "今天稍早"):
                        state += f"，今天已經聊過 {n_today} 次"
                    state += f"；那次聊了大約 {max(1, ep['n_rows'] // 2)} 輪。"
                    lines = [f"[狀態] {state}"]
                    if ep["keywords"]:
                        lines.append("[上一場] 聊到：" + "、".join(ep["keywords"]))
                    reps = st.index.fetch(ep["reps"])
                    lo = max(ep["start_serial"], st.session_start - self._carry * 2)
                    tail = st.index.fetch(list(range(lo, st.session_start)))
                    reps = [r for r in reps if r["serial"] < lo]   # 已在最後幾輪裡的不重複
                    if reps:
                        lines.append(format_blocks([reps], self._labels, title="  那次你特別說過：",
                                                   when=lambda _ts: ""))
                    if tail:
                        lines.append(format_blocks([tail], self._labels, title="  那次最後幾句：",
                                                   when=lambda _ts: ""))
                    shown = {r["serial"] for r in reps + tail}
                    text = "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.error(f"開場記憶組裝失敗，本場不帶上一場：{type(e).__name__}: {e}")
            text, shown = "", set()
        self._opening_cache = (text, shown)
        return self._opening_cache

    def _recall(self, query: str, *, extra_exclude: set[int] = frozenset()) -> str:
        """字面＋向量兩路，RRF 合併取 top_k；撈回失敗不中斷對話，但要大聲。"""
        try:
            stores = self._read_stores()
            wm = {t["serial"] for t in self.working_memory if "serial" in t}
            write_st = stores[0]
            n_cand, min_sim, k = self._vec_cfg

            qvec = None
            if self._emb is not None and self._emb.ready.is_set():
                qvec = self._emb.encode([query])[0]

            # RRF：每路依名次給 1/(k+名次)，兩路加總。⭐ 不比較兩路的原始分數
            #    （bm25 與餘弦不同尺度、兩個 DB 的 bm25 也不同尺度），只比名次。
            score: dict[tuple[int, int], float] = {}
            n_fts = n_vec = 0
            for si, st in enumerate(stores):
                # 開場已帶出的句子只在日常區（stores 的最後一個）排除
                excl = ((wm if st is write_st else set())
                        | (set(extra_exclude) if si == len(stores) - 1 else set()))
                fts = st.index.search(query, top_k=self._top_k * 3,
                                      max_terms=self._max_terms, exclude=excl)
                n_fts += len(fts)
                for rank, (s, _) in enumerate(fts):
                    score[(si, s)] = score.get((si, s), 0.0) + 1.0 / (k + rank + 1)
                if qvec is not None and st.vsearch is not None:
                    vec = st.vsearch.search(qvec, candidates=n_cand, min_sim=min_sim, exclude=excl)
                    n_vec += len(vec)
                    for rank, (s, _) in enumerate(vec):
                        score[(si, s)] = score.get((si, s), 0.0) + 1.0 / (k + rank + 1)
            top = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[: self._top_k]
            if not top:
                return ""

            blocks = []
            for si, st in enumerate(stores):
                hits = [s for (i, s), _ in top if i == si]
                if not hits:
                    continue
                # 鄰域也排除開場已帶出的句子（同一句不在區塊裡出現兩次，省 token）
                excl = ((wm if st is write_st else set())
                        | (set(extra_exclude) if si == len(stores) - 1 else set()))
                blocks += st.index.neighborhood(hits, self._n, exclude=excl)
                st.index.mark_recalled(hits)
            blocks.sort(key=lambda b: b[0]["ts"] or "")
            logger.debug(f"撈回 {len(top)} 則命中、{len(blocks)} 段"
                         f"（字面候選 {n_fts}、向量候選 {n_vec if qvec is not None else '未就緒'}）")
            return format_blocks(blocks, self._labels, when=time_bucket,
                                 title="[相關記憶] 過去對話的原句，依時間排列：")
        except Exception as e:  # noqa: BLE001
            logger.error(f"取回失敗，本輪不注入記憶：{type(e).__name__}: {e}")
            return ""

    def _topics(self, query: str) -> str:
        """準 v4 讀取：主題向量比對 → 最新有效版本的主題句，依事件時間排列。

        ⚠️ 第 4 步「依當下狀況重寫」尚未實作：現在注入的是建立時的主題句本身。
        ⭐ 時間只給口語時間感（time_bucket），庫裡存絕對時間（準 v4 決策）。
        """
        try:
            if self._emb is None or not self._emb.ready.is_set():
                return ""
            qvec = self._emb.encode([query])[0]
            hits = []
            for st in self._read_stores():
                if st.tsearch is not None:
                    hits += st.tsearch.search(qvec, top_k=self._cfg["topic_top_k"],
                                              min_sim=self._cfg["topic_min_sim"])
            hits = sorted(hits, key=lambda h: -h[1])[: self._cfg["topic_top_k"]]
            if not hits:
                return ""
            hits.sort(key=lambda h: h[0]["event_ts"] or "")
            lines = ["[相關記憶] 你記得的事（括號是事情發生的時候）："]
            for t, _ in hits:
                b = time_bucket(t["event_ts"] or "")
                lines.append(f"· {'（' + b + '）' if b else ''}{t['text']}")
            logger.debug(f"主題命中 {len(hits)} 件：{[(t['id'], round(s, 3)) for t, s in hits]}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.error(f"主題取回失敗，本輪不注入：{type(e).__name__}: {e}")
            return ""

    def status_text(self) -> str:
        """給 /status：檢索各路狀態與向量積壓（背景執行緒默默死掉時，這裡看得到積壓一直長）。"""
        v3 = self._status_v3() if (self._v3_recall or not self._v4) else "準v3撈原句關閉"
        if self._judge is None:
            return v3
        try:
            tp = sum(st.tsearch.pending() for st in self._read_stores() if st.tsearch)
            tv = f"｜主題未補向量 {tp}"
        except Exception as e:  # noqa: BLE001
            tv = f"｜主題積壓查詢失敗：{type(e).__name__}"
        return f"{v3}｜{self._judge.status()}{tv}"

    def _status_v3(self) -> str:
        if self._emb is None:
            return "字面（向量已關閉）"
        if self._emb.failed:
            return f"字面（向量停用：{self._emb.failed[:60]}）"
        if not self._emb.ready.is_set():
            return "字面（向量背景載入中）"
        try:
            pend = sum(st.vsearch.pending() for st in self._read_stores() if st.vsearch)
        except Exception as e:  # noqa: BLE001
            return f"字面＋向量（積壓查詢失敗：{type(e).__name__}）"
        return f"字面＋向量（未補向量 {pend} 句）"

    # ── [相容殼] 舊版的壓縮介面 ────────────────────────────────

    def needs_compress(self) -> bool:
        return False

    def maybe_compress(self) -> bool:
        return False

    def compress_on_exit(self) -> None:
        """不壓縮；只把還沒寫進去的補寫一次，然後關連線。"""
        if self._judge is not None:
            self._judge.close()
        self._flush()
        left = [t for t in self.working_memory if "serial" not in t]
        if left:
            logger.error(f"離場時仍有 {len(left)} 則沒寫進原文存放區，程式結束後消失。")
        if self.test_mode:
            self._leave_test()   # 施測結束整包丟棄
        else:
            try:
                self._daily.open().rebuild_episodes()   # 把本場切進事件層（確定性、毫秒級）
            except Exception as e:  # noqa: BLE001
                logger.error(f"離場時事件層重建失敗：{type(e).__name__}: {e}")
        self._daily.close()
        if self._worker is not None:
            # 背景補向量的執行緒自己開著日常區的連線；不放掉的話 Windows 上檔案刪不掉／搬不動
            # （2026-09-30 Windows 跑 selftest 抓到，暫存資料夾清不掉）。
            self._worker.release(self._daily_path)

    def save_memory(self) -> None:
        """[相容殼] 每輪已即時落盤，沒有需要另外存的東西。"""
