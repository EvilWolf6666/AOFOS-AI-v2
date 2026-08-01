# core/backend_check.py
"""
推理後端專屬的環境查驗。

存在理由：查驗邏輯與「呼叫 LLM」是不同的關注點，且**綁定特定後端**——
`/api/v0/models` 是 LM Studio 專屬端點，不屬於 OpenAI 相容標準。核心的
`chat()` 換到 llama.cpp / Ollama / vLLM 幾乎不必改，這裡則要整份替換。
集中在本檔後，換後端時只需新增一個 `verify_xxx()` 並改 LLMClient 的一行呼叫。

本檔刻意不 import config：所有需要的值由呼叫端傳入，使它不綁定任何一組設定，
可直接複製給其他後端改寫。
"""

from __future__ import annotations

import json
import urllib.request

from core.logger import setup_logger

logger = setup_logger("BackendProbe")


def verify_lmstudio(client, base_url: str, model: str, expected_context_length: int | None) -> None:
    """
    啟動時查驗 LM Studio 環境，在提問之前就給出結論。

    任何失敗都只記 warning，不中斷程式（API 路徑隨 LM Studio 版本可能變動）。
    GPU 卸載層數此 API 不提供，仍需人工於 GUI 確認。
    """
    _log_available_models(client, model)
    _verify_loaded_model(base_url, model, expected_context_length)


def _log_available_models(client, model: str) -> None:
    """列出 LM Studio 目前提供的 model id（含未載入者）。"""
    try:
        models = [m.id for m in client.models.list().data]
        logger.info(f"LM Studio 可用模型：{models}")
        if model not in models:
            logger.error(
                f"⚠ config 指定的 model=\"{model}\" 不在可用清單內。"
                f"請確認名稱拼寫，或該模型是否已加入 LM Studio。"
            )
    except Exception as e:
        logger.warning(f"無法取得模型清單（不影響對話）：{e}")


def _verify_loaded_model(base_url: str, model: str, expected_context_length: int | None) -> None:
    """
    查驗「實際已載入」的模型與其 context 長度。

    存在理由是兩個實際踩過的坑（詳見《決策脈絡歸檔》十六章）：
      1. JIT 載入失敗時 LM Studio 不報錯，會沿用當前已載入的模型回應——
         回答看起來完全正常，但那是錯的模型，整批實驗數據靜默作廢。
         （JIT 現已關閉，改為人工 Eject 後載入，此查驗仍作為第二道防線。）
      2. JIT 自動載入時使用模型自己記住的載入設定，而非上次 GUI 手動設的那組，
         context 長度與 GPU 卸載層數可能被靜默改掉。
    """
    try:
        root = base_url.rstrip("/")
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
    if model not in ids:
        logger.error(
            f"⚠ config 指定 model=\"{model}\"，但 LM Studio 實際載入的是 {ids}。"
            f"此時取得的回應不屬於你指定的模型，不可用於任何比較。"
            f"請 Eject 後手動載入正確模型再重跑。"
        )
        return

    logger.info(f"已載入模型：{ids}（含 config 指定的 {model}）")

    target = next(m for m in loaded if m.get("id") == model)
    actual = target.get("loaded_context_length")

    if expected_context_length and actual is not None:
        if actual == expected_context_length:
            logger.info(f"實際載入 context 長度：{actual}（與設定一致）")
        else:
            logger.error(
                f"⚠ 實際載入 context 長度為 {actual}，與設定的 {expected_context_length} 不符。"
                f"GPU 卸載層數很可能也一併被改。請到 GUI 重設載入參數後再施測。"
            )