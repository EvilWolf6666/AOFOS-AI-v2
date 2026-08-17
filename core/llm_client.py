# core/llm_client.py
# 透過 OpenAI 相容 API 連接 LM Studio 本機模型

import time
from typing import Callable, Optional

from openai import OpenAI

from config import LLM_CONFIG, LMSTUDIO_LOAD_PARAMS, TEST_CONFIG
from core.backend_check import verify_lmstudio
from core.logger import setup_logger, timed
from core.text_utils import fix_traditional

logger = setup_logger("LLMClient")

# LM Studio 專屬、非 OpenAI 標準的參數，需透過 extra_body 送出；
# 直接當具名參數傳會被 SDK 丟掉或報錯。
_EXTRA_BODY_KEYS = ("top_k", "repeat_penalty", "min_p")

# context 用量（prompt + max_tokens）達此比例即警告。
# 不放進 config：這個值不會為了改變行為而被調整，只在換模型／context 變動時
# 重新檢視一次（判準見《歸檔》十九之3）。
_CONTEXT_WARN_RATIO = 0.85


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
        )
        self.model = LLM_CONFIG["model"]

        # 所有影響輸出的參數都在此固定，不留給 LM Studio 的模型預設值決定。
        # 少帶一個 = 少鎖一個變數，切換模型時會靜默改變行為。
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.top_p = LLM_CONFIG["top_p"]
        self.frequency_penalty = LLM_CONFIG["frequency_penalty"]
        self.presence_penalty = LLM_CONFIG["presence_penalty"]
        self.seed = LLM_CONFIG.get("seed")
        self.max_tokens_by_purpose = LLM_CONFIG.get("max_tokens_by_purpose", {})

        # 輸入側預算的分母。與 verify_lmstudio 取自同一處，故不會有兩份真相。
        # 為 None 時（未來換後端而該欄位不存在）輸入側檢查自動停用，不報錯。
        self.context_length = LMSTUDIO_LOAD_PARAMS.get("context_length")

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
        if self.seed is None:
            logger.warning("seed 未固定，同一顆模型的多次輸出會有隨機差異，不適合做對照組比較")

        # 載入層參數無法由本程式設定，只能記錄，讓每份 log 都能還原當時環境
        logger.info(f"LM Studio 載入層參數（需人工於 GUI 對齊）：{LMSTUDIO_LOAD_PARAMS}")

        # 後端專屬的環境查驗，換推理後端時只需替換 backend_check 內的函式
        verify_lmstudio(
            self.client,
            LLM_CONFIG["base_url"],
            self.model,
            LMSTUDIO_LOAD_PARAMS.get("context_length"),
        )

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

        # 不一致代表 LM Studio 用了別顆模型，整批實驗數據失效，故每次都警告
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

        # completion_tokens 貼齊上限時，輸出極可能是被截斷而非自然結束；
        # 對 compress 而言這代表「淘汰：」那段可能整段消失
        if usage.completion_tokens >= max_tokens:
            logger.warning(
                f"輸出可能被截斷（purpose={purpose}，"
                f"completion_tokens={usage.completion_tokens} 已達上限 {max_tokens}）"
            )

        # ── 輸入側預算 ──
        # prompt 與本次允許的輸出必須「一起」裝進 context，故分子是兩者之和，
        # 不是 prompt 單獨對 context 的佔比。compress 的 max_tokens 為 800，
        # prompt 到 7400 就已無餘裕——用固定百分比會漏掉這種情形。
        #
        # ⚠️ 溢出的後果不是報錯，是 llama.cpp 靜默截掉 context 的「中間」段，
        #    也就是 Compressed Memory 與早期對話（見 config.LMSTUDIO_LOAD_PARAMS
        #    的 max_concurrent_predictions 註解）。整份記憶消失且無任何徵兆。
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
                    f"摘要與早期對話可能已被靜默截掉，本次輸出不可信"
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

        存在理由：_compress() 內部也呼叫本方法，兩者在計時紀錄中同名就無法分辨
        「這次呼叫是回應使用者，還是在壓縮記憶」——而這正是判斷 _compress 是否
        落在延遲路徑上所需的資訊。新增呼叫點時記得傳入對應 purpose。

        on_delta：逐段輸出的回呼，收到的是**已完成繁簡處理**的片段。傳 None
        （預設）即走原本的一次性路徑，_compress() 等沒有觀眾的呼叫點不需要改。

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
                seed=self.seed,
                extra_body=self.extra_body,
            )

        self._check_model(getattr(response, "model", None))
        self._check_usage(getattr(response, "usage", None), purpose, max_tokens)

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

        with timed(f"LLM呼叫[{purpose}]", model=self.model, purpose=purpose):
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed,
                extra_body=self.extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if actual_model is None:
                    actual_model = getattr(chunk, "model", None)

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

        if usage is not None:
            self._check_usage(usage, purpose, max_tokens)
        elif not self._warned_no_usage:
            # 只警告一次：這是後端能力問題，每輪重複只會讓人學會忽略它
            self._warned_no_usage = True
            logger.warning(
                "串流回應未帶 usage，本次起串流路徑的『輸出截斷』與『context 溢出』"
                "兩項檢查皆未執行。context 若溢出，摘要與早期對話會被靜默截掉而無徵兆。"
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
