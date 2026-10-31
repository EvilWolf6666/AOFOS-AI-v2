# core/llm_client.py
# 透過 OpenAI 相容 API 連接本機推理後端

import time
from typing import Callable, Optional

from openai import OpenAI

from config import LLAMA_SERVER, LLM_CONFIG, TEST_CONFIG, TEST_MODE
from core.backend_check import verify_llamaserver
from core.logger import setup_logger, timed
from core.text_utils import fix_traditional

logger = setup_logger("LLMClient")

# 非 OpenAI 標準的取樣參數（後端原生支援），需透過 extra_body 送出；
# 直接當具名參數傳會被 SDK 丟掉或報錯。
_EXTRA_BODY_KEYS = ("top_k", "repeat_penalty", "min_p")

# context 用量（prompt + max_tokens）達此比例即警告。
# 不放進 config：這個值不會為了改變行為而被調整，只在換模型／context 變動時
# 重新檢視一次（判準見《歸檔》I0010）。
_CONTEXT_WARN_RATIO = 0.85


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
        )
        self.model = LLM_CONFIG["model"]

        # 所有影響輸出的參數都在此固定，不留給後端預設值或 GGUF 內嵌取樣值決定。
        # 少帶一個 = 少鎖一個變數，切換模型時會靜默改變行為。
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.top_p = LLM_CONFIG["top_p"]
        self.frequency_penalty = LLM_CONFIG["frequency_penalty"]
        self.presence_penalty = LLM_CONFIG["presence_penalty"]
        self.seed = LLM_CONFIG.get("seed")
        self.max_tokens_by_purpose = LLM_CONFIG.get("max_tokens_by_purpose", {})

        # 輸入側預算的分母。與後端啟動參數、verify_llamaserver 取自同一處，故不會有兩份真相。
        # 為 None 時輸入側檢查自動停用，不報錯。
        self.context_length = LLAMA_SERVER.get("ctx")

        # 值為 None 的欄位不送出
        self.extra_body = {
            k: LLM_CONFIG[k]
            for k in _EXTRA_BODY_KEYS
            if LLM_CONFIG.get(k) is not None
        }

        self._logged_actual_model = False
        self.test_mode = False

        # 串流輸出開關。實際是否串流還要看呼叫端有沒有傳 on_delta，
        # 以及是否處於評測模式——三個條件的合成見 _should_stream()。
        self.stream = LLM_CONFIG.get("stream", False)
        self._warned_no_usage = False

        logger.info("初始化完成")
        logger.info(
            f"參數：model={self.model} temperature={self.temperature} "
            f"top_p={self.top_p} max_tokens={self.max_tokens} seed={self.seed} "
            f"extra_body={self.extra_body}"
        )
        if self.seed is not None:
            # 固定 seed＝每次請求同一串亂數，日常對話會一再抽出同一句
            logger.warning(f"seed 固定為 {self.seed}：日常對話容易反覆出現同一句；施測請用 /test on（貪婪解碼）")

        # 載入層參數記進 log，讓每份 log 都能還原當時環境（server 由 tools/start_llama_server.py 依同一份設定啟動）
        logger.info(f"llama-server 載入參數（config.LLAMA_SERVER）：{LLAMA_SERVER}")

        # 後端專屬的環境查驗，換推理後端時只需替換 backend_check 內的函式
        verify_llamaserver(
            self.client,
            LLM_CONFIG["base_url"],
            self.model,
            expected_ctx=LLAMA_SERVER.get("ctx"),
            expected_slots=LLAMA_SERVER.get("parallel"),
            expected_model_path=LLAMA_SERVER.get("model"),
        )
        self._warned_no_timings = False

        # config.TEST_MODE 開啟時，啟動即套用評測推理參數（與 /test on 走同一條路徑；
        #    少了這步，記憶層是測試模式而推理參數仍是日常值，且不報錯）。
        if TEST_MODE:
            self.set_test_mode(True)

    # ── 評測模式 ──────────────────────────────────────────────

    def set_test_mode(self, on: bool) -> None:
        """
        切換為評測用推理參數（貪婪解碼），由 Pipeline 的 /test 指令呼叫。
        參數內容與選用理由見 config.TEST_CONFIG["llm_overrides"]。

        不寫入 config，重開程式即回到日常參數。
        """
        if on == self.test_mode:
            return
        self.test_mode = on
        overrides = TEST_CONFIG.get("llm_overrides", {})

        if on:
            self.temperature = overrides.get("temperature", self.temperature)
            self.top_p = overrides.get("top_p", self.top_p)
            self.frequency_penalty = overrides.get(
                "frequency_penalty", self.frequency_penalty
            )
            self.presence_penalty = overrides.get(
                "presence_penalty", self.presence_penalty
            )
            for k in ("top_k", "min_p", "repeat_penalty"):
                if k in overrides:
                    self.extra_body[k] = overrides[k]
        else:
            self.temperature = LLM_CONFIG["temperature"]
            self.top_p = LLM_CONFIG["top_p"]
            self.frequency_penalty = LLM_CONFIG["frequency_penalty"]
            self.presence_penalty = LLM_CONFIG["presence_penalty"]
            self.extra_body = {
                k: LLM_CONFIG[k]
                for k in _EXTRA_BODY_KEYS
                if LLM_CONFIG.get(k) is not None
            }

        logger.warning(
            f"推理參數已切換為{'評測（貪婪解碼）' if on else '日常'}："
            f"temperature={self.temperature} top_p={self.top_p} "
            f"frequency_penalty={self.frequency_penalty} "
            f"presence_penalty={self.presence_penalty} "
            f"extra_body={self.extra_body}"
        )

    # ── 對話 ──────────────────────────────────────────────────

    # ── 輸出檢查（兩條路徑共用，不得各寫一份）───────────────────

    def _check_model(self, actual) -> None:
        """回應帶回的 model id 是唯一可信的「到底跑了哪顆」證據。"""
        if not self._logged_actual_model:
            logger.info(f"實際回應模型：{actual}")
            self._logged_actual_model = True

        # 不一致代表後端跑的不是 config 指定那顆，整批實驗數據失效，故每次都警告
        if actual and actual != self.model:
            logger.error(
                f"⚠ 實際回應模型（{actual}）與 config 指定（{self.model}）不一致，"
                f"本次結果不可用於跨模型比較"
            )

    def _check_usage(self, usage, purpose: str, max_tokens: int) -> None:
        """
        截斷與 context 預算檢查。

        ⚠️ 抽成獨立方法是必要的，不是整理癖：串流路徑也需要這三項檢查，
        內嵌在 chat() 裡會立刻變成兩份真相，而它們防的全是「不會報錯」的失敗。
        """
        if usage is None:
            return

        # completion_tokens 貼齊上限時，輸出極可能是被截斷而非自然結束
        if usage.completion_tokens >= max_tokens:
            logger.warning(
                f"輸出可能被截斷（purpose={purpose}，"
                f"completion_tokens={usage.completion_tokens} 已達上限 {max_tokens}）"
            )

        # ── 輸入側預算 ──
        # prompt 與本次允許的輸出必須「一起」裝進 context，故分子是兩者之和，
        # 不是 prompt 單獨對 context 的佔比——max_tokens 調大的用途，prompt 還沒到
        # 固定百分比就已無餘裕，用固定百分比會漏掉這種情形。
        #
        # ⚠️ 溢出時後端是報錯還是靜默截掉中間段，現行後端未實測：截斷會讓人格 prompt 之後的
        #    早期對話消失且無徵兆；報錯則 chat() 回空字串並記 ERROR。
        #
        # ⚠️ usage.prompt_tokens 是否為「截斷前」的原始長度未經證實。若後端回報
        #    的是截斷後的值，下方 needed > context 那條永遠不會觸發，實際發揮
        #    作用的是 _CONTEXT_WARN_RATIO 那條。故該比例是主要機制而非備援，
        #    調低它比等溢出更可靠。
        if self.context_length:
            needed = usage.prompt_tokens + max_tokens
            if needed > self.context_length:
                logger.error(
                    f"⚠ context 溢出（purpose={purpose}）："
                    f"prompt={usage.prompt_tokens} + max_tokens={max_tokens} "
                    f"= {needed} > context_length={self.context_length}。"
                    f"早期對話可能已被靜默截掉，本次輸出不可信"
                )
            elif needed > self.context_length * _CONTEXT_WARN_RATIO:
                logger.warning(
                    f"context 餘裕不足（purpose={purpose}）："
                    f"{needed}/{self.context_length} "
                    f"（{needed / self.context_length:.0%}），"
                    f"剩餘 {self.context_length - needed} tokens"
                )

        logger.debug(
            f"token[{purpose}]：prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens} "
            f"context={self.context_length}"
        )

    def _log_timings(self, obj, purpose: str) -> None:
        """記下 llama-server 回應自帶的 timings（server 端自己量的 prefill／生成時間）。

        存在理由：本程式量的 TTFT 混了 HTTP 與 SDK 開銷；server 端數字能把兩者分開
        （《參考庫》A.2「原生 timing 可分離 wrapper 開銷」）。cache_n＝本次重用快取的
        prompt token 數，可直接驗證 Working Memory 一次丟一批的快取效果（config.WM_DROP_ROUNDS）。
        timings 不是 OpenAI 標準欄位，SDK 放在 model_extra。拿不到只警告一次。
        """
        timings = (getattr(obj, "model_extra", None) or {}).get("timings") if obj is not None else None
        if not timings:
            if not self._warned_no_timings:
                self._warned_no_timings = True
                logger.warning("回應沒有 timings 欄位：本次起只有本程式量的 TTFT，分不出 server 與傳輸開銷。")
            return
        logger.info(
            f"server timings[{purpose}]：prompt {timings.get('prompt_n')} tok／{timings.get('prompt_ms', 0):.0f} ms"
            f"（快取重用 {timings.get('cache_n')} tok）｜生成 {timings.get('predicted_n')} tok／"
            f"{timings.get('predicted_ms', 0):.0f} ms（{timings.get('predicted_per_second', 0):.1f} tok/s）"
        )

    def _seed_kw(self) -> dict:
        """seed 為 None 時整個欄位不送（不送 null）：交給後端每次隨機。"""
        return {} if self.seed is None else {"seed": self.seed}

    def _should_stream(self, on_delta) -> bool:
        """
        三個條件全部成立才串流：呼叫端要看、設定開啟、不在評測模式。

        ⚠️ 評測模式強制走非串流。理由不是串流比較慢，是「串流與非串流在貪婪解碼下
        逐字相同」目前只是**假設**（見 config.TEST_CONFIG["disable_stream"]）。
        用未驗證的假設去換施測數據，代價是整批分數不可信，而那不會有任何徵兆。
        """
        if on_delta is None or not self.stream:
            return False
        if self.test_mode and TEST_CONFIG.get("disable_stream", True):
            return False
        return True

    # ── 對話 ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list,
        purpose: str = "dialogue",
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        purpose 只影響計時標籤與輸出長度上限，不影響請求的其他內容。

        存在理由：同一個方法若有多種用途，計時紀錄同名就分不出「這次呼叫是回應使用者，
        還是別的處理」——而那正是判斷某個呼叫是否落在延遲路徑上所需的資訊。
        新增呼叫點時記得傳入對應 purpose。目前只有對話在用；背景判斷走下方 BackgroundLLM，
        計時標籤是「LLM背景[…]」。

        on_delta：逐段輸出的回呼，收到的是**已完成繁簡處理**的片段。傳 None
        （預設）即走原本的一次性路徑，沒有觀眾的呼叫點不需要改。

        ⚠️ 本模組刻意不 import core.console——輸出到哪裡是終端層的事。回呼由
        呼叫端提供，日後要同時餵 TTS 與畫面時也只需換那個回呼。

        回傳值在兩條路徑下一致：完整、已處理、已 strip 的回應字串。
        """
        max_tokens = self.max_tokens_by_purpose.get(purpose, self.max_tokens)

        try:
            if self._should_stream(on_delta):
                return self._chat_streaming(messages, purpose, max_tokens, on_delta)
            return self._chat_blocking(messages, purpose, max_tokens)

        except Exception as e:
            # 回傳 "" 但必須夠醒目：施測時一個失敗的呼叫看起來會跟
            # 「模型答得很差」一模一樣
            logger.error(f"LLM 呼叫失敗（purpose={purpose}，本輪回傳空字串，勿當作模型表現）：{e}")
            return ""

    # ── 非串流路徑 ────────────────────────────────────────────

    def _chat_blocking(self, messages: list, purpose: str, max_tokens: int) -> str:
        with timed(f"LLM呼叫[{purpose}]", model=self.model, purpose=purpose):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **self._seed_kw(),
                extra_body=self.extra_body,
            )

        self._check_model(getattr(response, "model", None))
        self._check_usage(getattr(response, "usage", None), purpose, max_tokens)
        self._log_timings(response, purpose)

        reply = (response.choices[0].message.content or "").strip()
        reply = fix_traditional(reply)

        if not reply:
            logger.warning(f"模型回傳空內容（非例外，purpose={purpose}）")

        logger.debug(f"回應：{reply[:50]}...")
        return reply

    # ── 串流路徑 ──────────────────────────────────────────────

    def _chat_streaming(
        self,
        messages: list,
        purpose: str,
        max_tokens: int,
        on_delta: Callable[[str], None],
    ) -> str:
        """
        逐 chunk 取回並即時交給 on_delta，同時保留與非串流路徑相同的三項檢查。

        ⚠️ **usage 在串流下不保證拿得到**，而 usage 承載的是截斷偵測與 context
        溢出偵測——本專案唯一能發現「記憶被靜默截掉」的機制。故：
          · 明確要求 stream_options={"include_usage": True}
          · 真的沒拿到時**警告一次**，明說本輪這三項檢查沒有執行
        不做靜默降級：檢查沒跑跟檢查通過在 log 上長得一模一樣，那正是最該避免的。

        ⚠️ 繁簡轉換的兩種算法在此並存，是刻意的：
          · **畫面**用逐塊轉換（不能等整段結束才顯示，否則串流沒有意義）
          · **回傳值**用整段轉換（與非串流路徑逐字相同，記憶裡存的東西不因顯示方式而變）
        兩者理論上等價（s2tw 是字形轉換、不跨字），但這正是「未驗證的判準」，
        所以下面直接比對，不一致就記 log ——把「要驗一次」交給程式，不靠人記得去驗。
        """
        t_start = time.monotonic()
        ttft: float | None = None

        shown_parts: list[str] = []   # 已送給 on_delta 的（逐塊轉換後）
        raw_parts: list[str] = []     # 模型原始輸出（未轉換）
        usage = None
        actual_model = None
        last_with_timings = None      # llama-server 把 timings 附在最後幾個 chunk

        with timed(f"LLM呼叫[{purpose}]", model=self.model, purpose=purpose):
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **self._seed_kw(),
                extra_body=self.extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if actual_model is None:
                    actual_model = getattr(chunk, "model", None)
                if (getattr(chunk, "model_extra", None) or {}).get("timings"):
                    last_with_timings = chunk

                # 帶 usage 的收尾 chunk 其 choices 為空陣列，不可直接取 [0]
                if not getattr(chunk, "choices", None):
                    continue

                piece = getattr(chunk.choices[0].delta, "content", None)
                if not piece:
                    continue

                if ttft is None:
                    ttft = time.monotonic() - t_start

                raw_parts.append(piece)
                shown = fix_traditional(piece)
                shown_parts.append(shown)
                on_delta(shown)

        self._check_model(actual_model)
        self._log_timings(last_with_timings, purpose)

        if usage is not None:
            self._check_usage(usage, purpose, max_tokens)
        elif not self._warned_no_usage:
            # 只警告一次：這是後端能力問題，每輪重複只會讓人學會忽略它
            self._warned_no_usage = True
            logger.warning(
                "串流回應未帶 usage，本次起串流路徑的『輸出截斷』與『context 溢出』"
                "兩項檢查皆未執行。context 若溢出，早期對話會被靜默截掉而無徵兆。"
                "若需要這兩項保障，將 LLM_CONFIG['stream'] 設為 False。"
            )

        raw = "".join(raw_parts)
        reply = fix_traditional(raw).strip()

        # 逐塊轉換 vs 整段轉換的一致性——不一致代表 opencc 在切點上有跨字行為，
        # 屆時畫面顯示與記憶內容會分岔，必須知道
        shown_joined = "".join(shown_parts).strip()
        if shown_joined != reply:
            logger.warning(
                "opencc 逐塊轉換與整段轉換結果不一致（畫面顯示與寫入記憶的內容有差異）。"
                f"逐塊：{shown_joined[:80]!r} / 整段：{reply[:80]!r}"
            )

        if ttft is not None:
            logger.info(
                f"TTFT[{purpose}]：{ttft * 1000:.0f} ms"
                f"（總計 {(time.monotonic() - t_start) * 1000:.0f} ms）"
            )

        if not reply:
            logger.warning(f"模型回傳空內容（非例外，purpose={purpose}）")

        logger.debug(f"回應：{reply[:50]}...")
        return reply


# ══════════════════════════════════════════════════════════════
# 準 v4：背景 LLM 呼叫（可跨執行緒取消）
# ══════════════════════════════════════════════════════════════
#
# 為什麼另開一個 client、不共用上面的 LLMClient：
#   · 背景呼叫要能被**另一條執行緒**中途取消（使用者開口時），取消的手段是關掉
#     這次請求的 HTTP 連線；共用 client 會連帶影響對話那一條。
#   · 參數不同：判斷與標主題一律貪婪解碼、輸出上限很低，不跟著日常取樣參數走。
#
# 🔴 **為什麼取消這麼重要**：推理後端只有一個推理槽（config.LLAMA_SERVER["parallel"]＝1）。
#    背景呼叫還在跑時使用者開口，對話那一次會排在它後面，**多等一次 5～15 秒**。
#    ⇒ 取消若做不出來，準 v4 裁決是直接改上小模型（甲），見《現況》〈🔒 隔離區〉〈準 v4〉。
#    ⚠️ 本類別能保證的是「我們這端立刻放手、連線關掉」；後端何時真的停是它的行為：
#       現行後端正在做的那一批 prefill 不會被打斷，做完那批才停（批次大小見 config.LLAMA_SERVER["batch"]）。
#       量法：tools/chat_probe 的「排隊」欄（TTFT 扣掉後端自算的 prompt 時間）。

class BackgroundCancelled(Exception):
    """背景呼叫被取消（使用者開口）。不是錯誤，呼叫端照常跳過本次工作即可。"""


class BackgroundLLM:
    """背景專用、可取消的 LLM 呼叫。**只給記憶的背景工作用，不接對話。**

    🔴 **刻意不用 OpenAI SDK，改用標準庫 http.client 自己送**：取消要在兩個階段都立刻生效——
      ① 請求已送出、後端還在 prefill（連回應標頭都還沒回）
      ② 已在串流
    SDK（httpx）在 ① 階段沒有從別的執行緒中斷的手段（關 client 不會打斷正在等的讀取，
    假伺服器實測：取消後仍卡到標頭回來才放手）。
    自己持有 socket ⇒ `shutdown()` 會讓另一條執行緒卡住的 recv 立刻返回 ⇒ 兩個階段都是毫秒級。
    ⚠️ 代價：SSE 解析自己做（只需 `data:` 行與 `[DONE]`，格式固定）。
    """

    def __init__(self, *, base_url: str | None = None, model: str | None = None,
                 params: dict | None = None, timeout_s: float = 120.0) -> None:
        import threading
        from urllib.parse import urlparse
        u = urlparse(base_url or LLM_CONFIG["base_url"])
        self._host, self._port = u.hostname, u.port or 80
        self._path = u.path.rstrip("/") + "/chat/completions"
        self.model = model or LLM_CONFIG["model"]
        self.params = dict(params or {})
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        # ⚠️ 存 socket 本身、不存 conn：伺服器回 Connection: close 時，http.client 在
        #    getresponse() 之後就把 conn.sock 設成 None（socket 轉交給 response），
        #    只存 conn 的話串流階段 cancel() 會找不到 socket。
        self._sock = None

    # ── 取消 ──────────────────────────────────────────────────

    def cancel(self) -> bool:
        """要求停止。回傳 True＝當下有請求在跑、已切斷連線。可從任何執行緒呼叫。"""
        import socket
        self._cancel.set()
        with self._lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            sock.shutdown(socket.SHUT_RDWR)   # 讓卡在 recv 的那條執行緒立刻返回
        except OSError:
            pass
        return True

    def resume(self) -> None:
        """允許下一次背景呼叫。"""
        self._cancel.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ── 呼叫 ──────────────────────────────────────────────────

    def call(self, messages: list, *, purpose: str, max_tokens: int,
             schema: dict | None = None) -> str:
        """送出一次背景呼叫，回傳完整文字（schema 給了就是 JSON 字串）。

        被取消 ⇒ 拋 BackgroundCancelled；其他失敗照常拋（呼叫端記錄、下次再做）。
        schema：OpenAI 相容的 `response_format: json_schema`（GGUF 走 llama.cpp grammar 取樣，
        輸出被硬性限制在 schema 內）。
        """
        return self.call_lp(messages, purpose=purpose, max_tokens=max_tokens, schema=schema)[0]

    def call_lp(self, messages: list, *, purpose: str, max_tokens: int,
                schema: dict | None = None, top_logprobs: int = 0) -> tuple[str, list[dict]]:
        """同 call()，另回傳每個輸出 token 的 logprobs（OpenAI 格式的 `logprobs.content` 逐項）。

        top_logprobs＝0 時不要求 logprobs，回傳空清單。
        現行後端在串流下逐塊附上（tools/judge_examples 讀得到 P(建)）。
        ⚠️ 拿不到時仍回空清單，由呼叫端大聲處理（judge.py 會記 WARNING 並退回看文字）。
        ⚠️ 用了 json_schema 時回的是格式限制「之前」的分布，讀不出選項機率，兩者不要合用。
        """
        import http.client
        import json as _json
        if self._cancel.is_set():
            raise BackgroundCancelled()
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=self.params.get("temperature", 0.0),
                    top_p=self.params.get("top_p", 1.0), stream=True)
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        for k in ("top_k", "min_p", "repeat_penalty"):
            if self.params.get(k) is not None:
                body[k] = self.params[k]
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": purpose, "strict": True, "schema": schema}}
        parts: list[str] = []
        lps: list[dict] = []
        resp = None
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout_s)
        try:
            with timed(f"LLM背景[{purpose}]", model=self.model, purpose=purpose):
                conn.connect()
                with self._lock:
                    self._sock = conn.sock
                if self._cancel.is_set():
                    raise BackgroundCancelled()
                conn.request("POST", self._path, body=_json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {LLM_CONFIG['api_key']}"})
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}：{resp.read()[:200]!r}")
                for raw in resp:
                    if self._cancel.is_set():
                        raise BackgroundCancelled()
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ch = (_json.loads(data).get("choices") or [{}])[0]
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)
                    if top_logprobs:
                        lps.extend((ch.get("logprobs") or {}).get("content") or [])
        except BackgroundCancelled:
            raise
        except Exception as e:
            if self._cancel.is_set():
                raise BackgroundCancelled() from e
            raise
        finally:
            with self._lock:
                self._sock = None
            if resp is not None:
                resp.close()   # socket 在 Connection: close 時歸 response 所有，只關 conn 會漏
            conn.close()
        if self._cancel.is_set():
            raise BackgroundCancelled()
        return "".join(parts).strip(), lps

