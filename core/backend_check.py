# core/backend_check.py
"""
推理後端專屬的環境查驗。

存在理由：查驗邏輯與「呼叫 LLM」是不同的關注點，且**綁定特定後端**（例：llama-server 的
`/props`、`/lora-adapters` 不屬於 OpenAI 相容標準）。核心的 `chat()` 換後端幾乎不必改，
這裡則要整份替換：換後端時新增一個 `verify_xxx()` 並改 LLMClient 的一行呼叫。

本檔刻意不 import config：所有需要的值由呼叫端傳入，使它不綁定任何一組設定，
可直接複製給其他後端改寫。

現行：verify_llamaserver()。
"""

from __future__ import annotations

import json
import os
import urllib.request

from core.logger import setup_logger

logger = setup_logger("BackendProbe")


def _root_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


# ══════════════════════════════════════════════════════════════
# llama-server（現行）
# ══════════════════════════════════════════════════════════════

def verify_llamaserver(client, base_url: str, model: str, *, expected_ctx: int | None,
                       expected_slots: int | None, expected_model_path: str | None) -> None:
    """
    啟動時查驗 llama-server 實際狀態是否與 config 一致，在提問之前就給出結論。

    防的事（《歸檔》J0004）：回應看起來完全正常，但跑的不是
    你以為的那顆模型／那個 context，整批實驗數據靜默作廢。llama-server 的載入參數雖然
    來自同一份 config，仍可能是「改了 config 沒重開 server」或「手動另開了一個」。

    任何失敗只記 warning／error，不中斷程式（端點欄位可能隨版本變動）。
    """
    _log_available_models(client, model)

    try:
        with urllib.request.urlopen(f"{_root_url(base_url)}/props", timeout=5) as resp:
            props = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"無法查詢 llama-server /props（請人工確認模型與 context）：{e}")
        return

    logger.info(f"llama-server 版本：{props.get('build_info', '（未回報）')}")

    # 模型檔
    actual_path = props.get("model_path")
    if expected_model_path and actual_path:
        if os.path.normcase(os.path.normpath(actual_path)) != os.path.normcase(os.path.normpath(expected_model_path)):
            logger.error(
                f"⚠ llama-server 載入的是 {actual_path}，config 指定的是 {expected_model_path}。"
                f"此時的回應不屬於你指定的模型，不可用於任何比較。請關掉 server 重開。"
            )
        else:
            logger.info(f"載入模型檔與 config 一致：{actual_path}")
    elif expected_model_path:
        logger.warning("/props 未回報 model_path，無法比對模型檔，請人工確認。")

    # context（各版本放的位置不同：頂層或 default_generation_settings 底下）
    dgs = props.get("default_generation_settings") or {}
    actual_ctx = props.get("n_ctx") or dgs.get("n_ctx")
    if expected_ctx and actual_ctx is not None:
        if actual_ctx == expected_ctx:
            logger.info(f"實際 context 長度：{actual_ctx}（與設定一致）")
        else:
            logger.error(
                f"⚠ 實際 context 長度為 {actual_ctx}，與設定的 {expected_ctx} 不符"
                f"（改了 config 沒重開 server？）。施測前先重開。"
            )
    elif expected_ctx:
        logger.warning("/props 未回報 n_ctx，無法比對 context 長度，請人工確認。")

    # 推理槽數：多於 1 會切分 context，且背景判斷的「單槽讓出」前提不成立
    slots = props.get("total_slots")
    if expected_slots and slots is not None and slots != expected_slots:
        logger.error(f"⚠ llama-server 推理槽數為 {slots}，設定為 {expected_slots}。")

    # chat template：只確認有；內容一致性用 /apply-template 人工看（《現況》第五章 #4 雙 BOS）
    if not props.get("chat_template"):
        logger.warning("/props 未回報 chat_template：server 可能沒加 --jinja，或該版欄位改名。")


# ══════════════════════════════════════════════════════════════
# 輔助（verify_llamaserver 用）
# ══════════════════════════════════════════════════════════════

def _log_available_models(client, model: str) -> None:
    """列出後端目前提供的 model id（llama-server 只有一個＝--alias）。"""
    try:
        models = [m.id for m in client.models.list().data]
        logger.info(f"後端可用模型：{models}")
        if model not in models:
            logger.error(
                f"⚠ config 指定的 model=\"{model}\" 不在可用清單內。"
                f"--alias 與 config 不一致（server 沒用新 config 重開？）。"
            )
    except Exception as e:
        logger.warning(f"無法取得模型清單（不影響對話）：{e}")
