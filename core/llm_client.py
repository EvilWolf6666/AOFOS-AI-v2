# core/llm_client.py
# 透過 OpenAI 相容 API 連接 LM Studio 本機模型

from openai import OpenAI

from config import LLM_CONFIG, LMSTUDIO_LOAD_PARAMS, TEST_CONFIG
from core.backend_check import verify_lmstudio
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