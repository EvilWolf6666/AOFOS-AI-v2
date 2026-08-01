# core/llm_client.py
# 透過 OpenAI 相容 API 連接 LM Studio 本機模型

import json
import urllib.request

from openai import OpenAI

from config import LLM_CONFIG, LMSTUDIO_LOAD_PARAMS, TEST_CONFIG
from core.logger import setup_logger, timed
from core.text_utils import fix_traditional

logger = setup_logger("LLMClient")

# LM Studio 專屬、非 OpenAI 標準的參數，需透過 extra_body 送出；
# 直接當具名參數傳會被 SDK 丟掉或報錯。
_EXTRA_BODY_KEYS = ("top_k", "repeat_penalty", "min_p")


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

        # 值為 None 的欄位不送出
        self.extra_body = {
            k: LLM_CONFIG[k]
            for k in _EXTRA_BODY_KEYS
            if LLM_CONFIG.get(k) is not None
        }

        self._logged_actual_model = False
        self.test_mode = False

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

        self._log_available_models()
        self._verify_loaded_model()

    # ── 啟動時的環境查驗 ──────────────────────────────────────

    def _log_available_models(self) -> None:
        """列出 LM Studio 目前提供的 model id（含未載入者）。"""
        try:
            models = [m.id for m in self.client.models.list().data]
            logger.info(f"LM Studio 可用模型：{models}")
            if self.model not in models:
                logger.error(
                    f"⚠ config 指定的 model=\"{self.model}\" 不在可用清單內。"
                    f"請確認名稱拼寫，或該模型是否已加入 LM Studio。"
                )
        except Exception as e:
            logger.warning(f"無法取得模型清單（不影響對話）：{e}")

    def _verify_loaded_model(self) -> None:
        """
        查驗「實際已載入」的模型與其 context 長度，在提問之前就給出結論。

        存在理由是兩個實際踩過的坑（詳見《決策脈絡歸檔》十六章）：
          1. JIT 載入失敗時 LM Studio 不報錯，會沿用當前已載入的模型回應——
             回答看起來完全正常，但那是錯的模型，整批實驗數據靜默作廢。
             （JIT 現已關閉，改為人工 Eject 後載入，此查驗仍作為第二道防線。）
          2. JIT 自動載入時使用模型自己記住的載入設定，而非上次 GUI 手動設的那組，
             context 長度與 GPU 卸載層數可能被靜默改掉。

        任何失敗都只記 warning，不中斷程式（API 路徑隨 LM Studio 版本可能變動）。
        GPU 卸載層數此 API 不提供，仍需人工確認。
        """
        try:
            root = LLM_CONFIG["base_url"].rstrip("/")
            if root.endswith("/v1"):
                root = root[: -len("/v1")]
            with urllib.request.urlopen(f"{root}/api/v0/models", timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(
                f"無法查詢實際載入狀態（不影響對話，但請人工於 GUI 確認模型與載入參數）：{e}"
            )
            return

        loaded = [m for m in data.get("data", []) if m.get("state") == "loaded"]

        if not loaded:
            logger.warning(
                "LM Studio 目前沒有已載入的模型。若 JIT 已關閉，接下來的呼叫會直接失敗；"
                "請先在 GUI 載入 config 指定的模型。"
            )
            return

        ids = [m.get("id") for m in loaded]
        if self.model not in ids:
            logger.error(
                f"⚠ config 指定 model=\"{self.model}\"，但 LM Studio 實際載入的是 {ids}。"
                f"此時取得的回應不屬於你指定的模型，不可用於任何比較。"
                f"請 Eject 後手動載入正確模型再重跑。"
            )
            return

        logger.info(f"已載入模型：{ids}（含 config 指定的 {self.model}）")

        expected = LMSTUDIO_LOAD_PARAMS.get("context_length")
        target = next(m for m in loaded if m.get("id") == self.model)
        actual = target.get("loaded_context_length")

        if expected and actual is not None:
            if actual == expected:
                logger.info(f"實際載入 context 長度：{actual}（與設定一致）")
            else:
                logger.error(
                    f"⚠ 實際載入 context 長度為 {actual}，與設定的 {expected} 不符。"
                    f"GPU 卸載層數很可能也一併被改。請到 GUI 重設載入參數後再施測。"
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

    def chat(self, messages: list, purpose: str = "dialogue") -> str:
        """
        purpose 只影響計時標籤與輸出長度上限，不影響請求的其他內容。

        存在理由：_compress() 內部也呼叫本方法，兩者在計時紀錄中同名就無法分辨
        「這次呼叫是回應使用者，還是在壓縮記憶」——而這正是判斷 _compress 是否
        落在延遲路徑上所需的資訊。新增呼叫點時記得傳入對應 purpose。
        """
        max_tokens = self.max_tokens_by_purpose.get(purpose, self.max_tokens)

        try:
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

            # 回應帶回的 model id 是唯一可信的「到底跑了哪顆」證據
            actual = getattr(response, "model", None)
            if not self._logged_actual_model:
                logger.info(f"實際回應模型：{actual}")
                self._logged_actual_model = True

            # 不一致代表 LM Studio 用了別顆模型，整批實驗數據失效，故每次都警告
            if actual and actual != self.model:
                logger.error(
                    f"⚠ 實際回應模型（{actual}）與 config 指定（{self.model}）不一致，"
                    f"本次結果不可用於跨模型比較"
                )

            usage = getattr(response, "usage", None)
            if usage is not None:
                # completion_tokens 貼齊上限時，輸出極可能是被截斷而非自然結束；
                # 對 compress 而言這代表「淘汰：」那段可能整段消失
                if usage.completion_tokens >= max_tokens:
                    logger.warning(
                        f"輸出可能被截斷（purpose={purpose}，"
                        f"completion_tokens={usage.completion_tokens} 已達上限 {max_tokens}）"
                    )
                logger.debug(
                    f"token[{purpose}]：prompt={usage.prompt_tokens} "
                    f"completion={usage.completion_tokens} total={usage.total_tokens}"
                )

            reply = (response.choices[0].message.content or "").strip()
            reply = fix_traditional(reply)

            if not reply:
                logger.warning(f"模型回傳空內容（非例外，purpose={purpose}）")

            logger.debug(f"回應：{reply[:50]}...")
            return reply

        except Exception as e:
            # 回傳 "" 但必須夠醒目：施測時一個失敗的呼叫看起來會跟
            # 「模型答得很差」一模一樣
            logger.error(f"LLM 呼叫失敗（purpose={purpose}，本輪回傳空字串，勿當作模型表現）：{e}")
            return ""