# core/memory/manager.py
"""
記憶入口（準 v4；《現況》〈🔒 隔離區〉記憶模塊〈準 v4〉）。

做的事：
  ① 每輪把雙方原句寫進原文存放區（pre_write → transcript），一句不刪
  ② 落地後通知背景判斷（core/memory/judge.py）：閒置時判「這段值不值得記」、建主題（原句的索引）
  ③ 讀取：本輪輸入 → 主題兩路檢索（向量＋關鍵字，各取配額、聯集、不重排）→ 注入主題句
  ④ 本場開頭幾輪另帶開場：「上次聊天是多久以前／今天聊過幾次／那次聊了幾輪」＋上一場的關鍵詞
     （事件層，程式算、不經 LLM）
  ⑤ 全部包在一個 `⟦記憶⟧…⟦/記憶⟧` system 區塊裡，**追加在本輪輸入之前**；區塊第一行自帶說明
     區塊內兩段標題：④ 開場＝「妳記得的事」、③ 主題＝「妳想起的事」（config 的 label_remember／label_recall；
     🔴 必須與人格 prompt〈記憶誠實原則〉的用詞逐字相同，人格規定只有這兩段裡的才算知道）
Working Memory 是滑動視窗（MAX_WORKING_MEMORY 輪、一次丟 WM_DROP_ROUNDS 輪），超出的直接滑出——
⭐ 原句已經落地，滑出不會丟任何東西，**所以不需要壓縮**。

🔺 2026-10-09 準 v3 讀取路徑淘汰（邪狼裁「淘汰沒用的，有用的併進準 v4」）：
   刪：撈原句（原句的關鍵字索引與每輪補索引、原句向量、RRF）、開場的代表句與最後幾句原句、
       `v3_recall`／`v3_opening` 開關。
   留：原句存放、事件層開場的「狀態」與「上一場關鍵詞」（她才答得出「上次聊天是多久以前」）。
   舊的設計與理由見《歸檔》（準 v3 主題 D、E）。

⚠️ 🔺 2026-10-09 v1 的相容殼（`compressed_memory`、`rag`、`needs_compress`、`maybe_compress`、
   `save_memory`）已移除，pipeline 同步拿掉呼叫。`compress_on_exit()` 名稱沿用（呼叫點多），
   實際做的是**離場收尾**：關背景判斷、補寫原句、重建事件層或丟棄測試區、關連線。

🔴 **執行緒**：DB 連線在**第一次使用時**於當下執行緒開啟，之後只能在同一執行緒用
   （sqlite3 預設 check_same_thread，跨執行緒會直接拋 ProgrammingError——大聲，是對的）。
   目前 agent 的事件迴圈是單一執行緒，符合。背景判斷自己開自己的連線。
   本檔只建背景專用的 BackgroundLLM（可取消），對話用的 client 不經本檔。
"""

from __future__ import annotations

import os
from datetime import datetime

from config import MAX_WORKING_MEMORY, MEMORY_CONFIG, TEST_MODE, WM_DROP_ROUNDS
from core.llm_client import BackgroundLLM
from core.logger import setup_logger, timed
from core.memory.db import connect
from core.memory.bigram import check_fts5
from core.memory.embed import Embedder
from core.memory.episode import EpisodeIndex, time_bucket
from core.memory.judge import BackgroundJudge
from core.memory.pre_write import PreWrite
from core.memory.topic_index import TopicSearch
from core.memory.topics import TopicStore
from core.memory.transcript import Transcript

logger = setup_logger("MemoryManager")

_DEFAULT = object()   # add_turn 的 user_speaker 預設值（None 有意義：observation 輪次沒人說話）


class _Store:
    """一個 DB 檔：連線 ＋ 存取物件。連線延後到第一次使用才開（見檔頭「執行緒」）。"""

    def __init__(self, path: str, *, writable: bool, emb: Embedder | None = None,
                 ep_cfg: dict | None = None, v4: bool = True) -> None:
        self.path = path
        self._v4 = v4
        self.writable = writable
        self._emb = emb
        self._ep_cfg = ep_cfg or {}
        self._cx = None
        self.pre_write: PreWrite | None = None
        self.transcript: Transcript | None = None
        self.episodes: EpisodeIndex | None = None
        self.tsearch: TopicSearch | None = None
        self.session_start = 1   # 本場第一句的 serial；之前的都是「以前的場次」

    def open(self) -> "_Store":
        if self._cx is None:
            self._cx = connect(self.path)
            self.pre_write = PreWrite(self._cx)
            self.transcript = Transcript(self._cx)
            self.episodes = EpisodeIndex(self._cx, writable=self.writable)
            self.session_start = self.transcript.landed_upto + 1
            if self.writable:
                self.rebuild_episodes()
            if self._v4:   # 向量關閉時仍建：關鍵字那一路不需要向量
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
            logger.error(f"事件層重建失敗（不影響原句與主題）：{type(e).__name__}: {e}")

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
        #    否則主題的關鍵字那一路會是「照建、永遠撈不回來」——最難發現的那種壞。
        check_fts5()

        self.llm = llm
        self.working_memory: list[dict] = []
        self._verbatim_repeat_count = 0

        cfg = {**MEMORY_CONFIG, **(cfg_override or {})}
        self._cfg = cfg
        self._v4 = cfg["v4_enabled"]   # 關掉＝只存原句、不判不撈（config 註解：P1 第 ② 點）
        self._spk = cfg["speaker_ids"]
        # 滑動視窗兩個參數的正本在 config 頂層；cfg_override 只給自測固定舊行為用
        self._wm_max = cfg.get("wm_max_rounds", MAX_WORKING_MEMORY)
        self._wm_drop = max(1, min(cfg.get("wm_drop_rounds", WM_DROP_ROUNDS), self._wm_max))
        self._daily_path = db_path or cfg["db_path"]
        self._test_path = test_db_path or cfg["test_db_path"]
        self._max_terms = cfg["max_query_terms"]
        self._open, self._close, self._note = cfg["block_open"], cfg["block_close"], cfg["block_note"]
        self._lab_rem, self._lab_rec = cfg["label_remember"], cfg["label_recall"]
        # 說明放在哪（config 的 note_position）："fixed" 時說明是開頭一則不變的 system 訊息
        if cfg["note_position"] not in ("fixed", "inline"):
            raise ValueError(f"MEMORY_CONFIG['note_position'] 只能是 fixed／inline：{cfg['note_position']!r}")
        self._fixed_note = (cfg["fixed_note"].format(open=self._open, close=self._close,
                                                     remember=self._lab_rem, recall=self._lab_rec)
                            if cfg["note_position"] == "fixed" else None)
        self._opening_turns = cfg["opening_turns"]
        self._ep_cfg = {"gap": cfg["episode_gap_minutes"], "kw": cfg["episode_keywords"],
                        "reps": cfg["episode_reps"]}
        self._session_turns = 0
        self._opening_cache: str | None = None

        # 主題向量的模型在背景載入，載好之前主題只走關鍵字那一路（不擋啟動、不擋對話）
        self._emb: Embedder | None = None
        # 向量只有主題用得到 ⇒ v4 關閉時不載（省約 800 MB 常駐與開機載入）
        if self._v4 and (cfg["embed_enabled"] if embed_enabled is None else embed_enabled):
            self._emb = Embedder(cfg["embed_model"], factory=embed_factory)
            self._emb.start()
        elif self._v4:
            logger.warning("準 v4 開著但向量關閉：主題照建，讀取端只剩關鍵字那一路")

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
        self._sync_judge()
        logger.info(f"記憶：原句寫入 {self.memory_path}；讀取＝開場狀態"
                    f"{'＋準v4主題' if self._v4 else '（主題關閉）'}{'' if self._emb else '（向量已關閉）'}")

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
        return _Store(path, writable=writable, emb=self._emb, ep_cfg=self._ep_cfg, v4=self._v4)

    # ── 測試模式：讀取共用、寫入分流（第八章 舊C8（已結案））───────────────

    @property
    def memory_path(self) -> str:
        """[相容] pipeline 的 /status 會印它。現在指向「寫入」的那個 DB。"""
        return self._test_path if self.test_mode else self._daily_path

    def _enter_test(self) -> None:
        # 進場先丟一次：上一輪若沒正常離開，測試區會殘留，第 N 題就被前面污染。
        _discard_db(self._test_path)
        self._test = self._new_store(self._test_path, writable=True)
        # 日常區在施測中只讀：不重建事件層、不判斷、不補主題向量 ⇒ 整輪施測日常區是靜止的。
        self._daily.close()
        self._daily = self._new_store(self._daily_path, writable=False)
        self.test_mode = True
        logger.warning(f"TEST_MODE：寫入只進 {self._test_path}，日常區唯讀")

    def _leave_test(self) -> None:
        if self._test is not None:
            self._test.close()
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
                    logger.error(f"預寫區中止：{halt}（已落地的部分照常可判可撈）")
            if self._judge is not None:
                self._judge.notify()
        except Exception as e:
            logger.error(
                f"原句寫入失敗，{len(pending)} 則暫留 Working Memory、下一輪重試："
                f"{type(e).__name__}: {e}")

    def _slide(self) -> None:
        """滑動視窗：超過上限輪數時，**一次**丟掉最舊的 WM_DROP_ROUNDS 輪（只丟已落地的）。

        ⭐ 一次丟一批而不是每輪丟一輪：送給模型的開頭只在丟的那一輪改變 ⇒ 其餘輪次吃得到
           後端的提示詞快取（每輪丟一輪時，滿了之後每輪 TTFT 多 2.5～3 秒，2026-10-04 LM Studio 時期實測）。
        🔴 **沒有 serial 的一律不丟**（寫入失敗的那些）——丟了就真的沒了。
        """
        limit = self._wm_max * 2
        if len(self.working_memory) <= limit:
            return
        # 超出上限的那一刻共有 max+1 輪 ⇒ 丟 K 輪後剩 max+1−K（K=1 即舊行為：維持在 max）
        target = (self._wm_max + 1 - self._wm_drop) * 2
        dropped = 0
        while len(self.working_memory) > target and "serial" in self.working_memory[0]:
            self.working_memory.pop(0)
            dropped += 1
        logger.info(f"Working Memory 滑出最舊 {dropped // 2} 輪（剩 {len(self.working_memory) // 2} 輪；"
                    f"下一輪會整段重讀一次）")
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
        """〔固定說明〕＋ Working Memory ＋ 一個 `⟦記憶⟧` 區塊（追加在尾端、本輪輸入之前）。

        ⚠️ 順序依「變動頻率由低到高」：記憶區塊每輪不同，放尾端才不會讓
           WM 落在 prompt cache 斷點之後（prompt cache 比對的是最長共同前綴）。
        ⚠️ 區塊不留在 WM ⇒ 下一輪快取斷在「上一輪區塊原本的位置」，區塊裡的字每輪都重算。
           note_position="fixed" 時，不變的說明改成開頭一則 system 訊息、**每輪都送**
           （有沒有區塊都送：時有時無會讓開頭變動，快取反而斷得更早）。
        ⚠️ user_input 為空（observation 輪次的 placeholder）時不做檢索；
           但本場開頭的開場（狀態＋上一場關鍵詞）照帶（它不依賴輸入）。
        """
        with timed("記憶組裝"):
            context = [self._to_message(m) for m in self.working_memory]
            if self._fixed_note:
                context.insert(0, {"role": "system", "content": self._fixed_note})
            sections = []
            if self._session_turns < self._opening_turns:
                opening = self._opening()
                if opening:
                    sections.append(opening)
            if user_input and self._v4:
                top = self._topics(user_input)
                if top:
                    sections.append(top)
            if sections:
                head = [self._open] if self._fixed_note else [self._open, self._note]
                context.append({"role": "system", "content": "\n".join(
                    [*head, *sections, self._close])})
        return context

    def _opening(self) -> str:
        """開場：上次聊天是多久以前／今天聊過幾次／那次聊了幾輪＋上一場的關鍵詞（事件層）。

        🔺 2026-10-09 準 v3 淘汰時留下（邪狼裁「開場只留關鍵詞」）：代表句與最後幾句原句不再帶——
           原句直接注入是逐字照抄的來源（《現況》12之0），上次聊了什麼的細節改由主題依輸入撈。
        ⭐ 本場只算一次、之後原樣重用 ⇒ 開頭幾輪這段字不變（對 prompt cache 友善），
           且不依輸入——問「上次聊甚麼」「上次聊天是多久以前」這種沒有內容詞的話也有東西可用。
        ⭐ 時間只給口語的時間感（`time_bucket`），不給日期與天數，避免她照唸。
        """
        if self._opening_cache is not None:
            return self._opening_cache
        text = ""
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
                    lines = [f"[{self._lab_rem}]", f"[狀態] {state}"]
                    if ep["keywords"]:
                        lines.append("[上一場] 聊到：" + "、".join(ep["keywords"]))
                    text = "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.error(f"開場記憶組裝失敗，本場不帶上一場：{type(e).__name__}: {e}")
            text = ""
        self._opening_cache = text
        return text

    def _topics(self, query: str) -> str:
        """準 v4 讀取：向量＋關鍵字兩路，**各取自己的配額、聯集去重、不重新排名**，依事件時間排列。

        🔺 2026-10-09 邪狼裁（《現況》〈🔒 隔離區〉準 v4 檢索列）：兩路一律並行；不用 RRF——
           RRF 讓兩路搶同一組名額，只有一路命中的（例：關鍵字逐字命中的專有名詞）可能被擠掉。
           去重空出的名額不遞補（遞補就得決定「下一名給誰」，又回到排名次）。
        ⭐ 兩路各自的命中名單寫 debug log ⇒ 日後可離線重算 RRF 比對，不必重跑對話。
        ⚠️ 第 4 步「依當下狀況重寫」尚未實作：現在注入的是建立時的主題句本身。
        ⭐ 時間只給口語時間感（time_bucket），庫裡存絕對時間（準 v4 決策）。
        """
        try:
            cfg = self._cfg
            stores = [st for st in self._read_stores() if st.tsearch is not None]
            qvec = None
            if self._emb is not None and self._emb.ready.is_set():
                qvec = self._emb.encode([query])[0]
            # (store 序號, topic id) 當鍵：測試模式讀兩個 DB，id 會撞
            vec, kw = [], []
            for si, st in enumerate(stores):
                if qvec is not None:
                    vec += [(si, t, s) for t, s in st.tsearch.search(
                        qvec, top_k=cfg["topic_vec_k"], min_sim=cfg["topic_min_sim"])]
                kw += [(si, t, s) for t, s in st.tsearch.search_bm25(
                    query, top_k=cfg["topic_bm25_k"], max_terms=self._max_terms)]
            vec = sorted(vec, key=lambda h: -h[2])[: cfg["topic_vec_k"]]
            kw = kw[: cfg["topic_bm25_k"]]   # 各 DB 的 bm25 尺度不同，不跨庫比分數：寫入區優先
            picked: dict[tuple[int, int], dict] = {}
            for si, t, _ in vec + kw:
                picked.setdefault((si, t["id"]), t)
            logger.debug(f"主題命中 向量 {[(si, t['id'], round(s, 3)) for si, t, s in vec]}"
                         f"｜關鍵字 {[(si, t['id'], round(s, 3)) for si, t, s in kw]}"
                         f"｜聯集 {len(picked)} 件{'' if qvec is not None else '（向量未就緒）'}")
            if not picked:
                return ""
            hits = sorted(picked.values(), key=lambda t: t["event_ts"] or "")
            # fixed：「括號是什麼」已在固定說明裡，區塊只留標籤（少重算的字）
            lines = [f"[{self._lab_rec}]" if self._fixed_note
                     else f"[{self._lab_rec}]（括號是事情發生的時候）："]
            for t in hits:
                b = time_bucket(t["event_ts"] or "")
                lines.append(f"· {'（' + b + '）' if b else ''}{t['text']}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.error(f"主題取回失敗，本輪不注入：{type(e).__name__}: {e}")
            return ""

    def status_text(self) -> str:
        """給 /status：向量狀態、背景判斷與積壓（背景執行緒默默死掉時，這裡看得到積壓一直長）。"""
        if self._emb is None:
            vec = "主題：關鍵字（向量已關閉）"
        elif self._emb.failed:
            vec = f"主題：關鍵字（向量停用：{self._emb.failed[:60]}）"
        elif not self._emb.ready.is_set():
            vec = "主題：關鍵字（向量背景載入中）"
        else:
            vec = "主題：向量＋關鍵字"
        if self._judge is None:
            return vec
        try:
            tp = sum(st.tsearch.pending() for st in self._read_stores() if st.tsearch)
            tv = f"｜主題未補向量 {tp}"
        except Exception as e:  # noqa: BLE001
            tv = f"｜主題積壓查詢失敗：{type(e).__name__}"
        try:
            pend = TopicStore(self._write_store()._cx, writable=False).pending_count()
        except Exception as e:  # noqa: BLE001
            logger.error(f"待判句數查詢失敗：{type(e).__name__}: {e}")
            pend = None
        return f"{vec}｜{self._judge.status(pend)}{tv}"

    # ── 離場 ──────────────────────────────────────────────────

    def compress_on_exit(self) -> None:
        """離場收尾（名稱沿用 v1，不壓縮）：關背景判斷、補寫原句、事件層重建或丟棄測試區、關連線。"""
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

