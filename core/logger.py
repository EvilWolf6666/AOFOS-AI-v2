# core/logger.py
# 統一 Logging 系統 — 所有模組從這裡取得 logger；並提供延遲量測工具 timed()

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime

import colorlog

from config import PROJECT_ROOT, TIMING_CONFIG

# 日常紀錄（每日 log、timing-*.jsonl）的位置。專案根只取 config.PROJECT_ROOT（config.py 開頭：不得自己推算）。
# 紀錄結構：logs/daily（本檔）、logs/backend（start_llama_server）、logs/runs/<工具>（各工具每次執行）、logs/archive_lmstudio（舊紀錄）。
# AOFOS_LOG_DIR：子行程用的覆寫——tools/eval_runner 每題開的行程設成該題資料夾，施測流量不混進日常 log。
LOG_DIR = os.environ.get("AOFOS_LOG_DIR") or os.path.join(PROJECT_ROOT, "logs", "daily")
LOG_LEVEL = logging.DEBUG

# 所有計時輸出共用同一個 logger，方便單獨調整等級或關閉
_timing_logger = None

# 終端輸出目標。None 表示直接寫 stderr（colorlog 預設）。
#
# ⚠️ 由 core/console.py 在啟動時透過 set_console_stream() 設定。相依方向刻意
# 是「console 告訴 logger」而非「logger 去 import console」——logger 的職責是
# 決定記什麼、什麼格式，輸出到哪裡是終端層的事，日後改成寫檔或送遠端時
# 不必動 logger。
_console_stream = None

# 本模組建立過的 logger 名稱。
#
# 用途：core/console.py 需要把「專案自己的終端輸出」導向終端協調器，但
# logging.root.manager.loggerDict 會一併取到 chromadb、sentence-transformers
# 等所有 import 過的第三方套件。由建立方登記是唯一不會漏也不會多的來源——
# 在 console.py 另外硬編一份清單會變成兩處真相，新增模組時漏改即靜默失效。
KNOWN_LOGGERS: list[str] = []


def set_console_stream(stream) -> None:
    """
    登記終端輸出目標。設定之後才建立的 logger 會直接使用它，不必再改接。

    ⚠️ 存在理由是「惰性建立的 logger 會漏掉」：TIMING logger 在第一次 timed()
    區塊結束時才誕生，而那必定發生在 console.install_logging_bridge() 之後
    ——它是全專案輸出最密集的 logger，且必定發生在使用者等待回應期間
    （此時輸入提示正在等待），漏掉它等於終端協調在最主要的來源上失效。
    只靠「改接既有 logger」的做法無法涵蓋這種情況。
    """
    global _console_stream
    _console_stream = stream


def setup_logger(name: str) -> logging.Logger:
    """建立並回傳指定名稱的 logger，同時輸出到終端機（彩色）與 LOG_DIR（純文字，每日一份）。"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重複加 handler

    if name not in KNOWN_LOGGERS:
        KNOWN_LOGGERS.append(name)

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False  # 避免日後有人加了 root handler 造成重複輸出

    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s]%(reset)s %(name)s — %(message)s",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        }
    ))

    log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, log_filename),
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S"
    ))

    # 在 addHandler 之前套用，涵蓋 set_console_stream() 之後才建立的 logger
    if _console_stream is not None:
        console_handler.stream = _console_stream

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def _get_timing_logger() -> logging.Logger:
    global _timing_logger
    if _timing_logger is None:
        _timing_logger = setup_logger("TIMING")
    return _timing_logger


def _write_timing_jsonl(record: dict) -> None:
    """把單筆計時寫進 LOG_DIR/timing-YYYY-MM-DD.jsonl。失敗不得影響主流程。"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(
            LOG_DIR,
            "timing-" + datetime.now().strftime("%Y-%m-%d") + ".jsonl"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 量測工具不得成為新的故障點


@contextmanager
def timed(stage: str, **extra):
    """
    量測一段區塊的耗時。

        with timed("記憶組裝"):
            context = build(...)

        with timed("LLM呼叫", model=self.model):
            response = client.chat.completions.create(...)

    stage 名稱請保持穩定，否則日後無法跨次比較；extra 會一併寫進 jsonl，
    用來標註模型名稱等脈絡。區塊內拋出例外時同樣會記錄（標記 ok=False），
    避免失敗案例從統計中消失。
    """
    if not TIMING_CONFIG.get("enabled", False):
        yield
        return

    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        status = "" if ok else " (失敗)"
        _get_timing_logger().info(f"[TIMING] {stage}: {elapsed_ms:.1f}ms{status}")

        if TIMING_CONFIG.get("jsonl", False):
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "ms": round(elapsed_ms, 1),
                "ok": ok,
            }
            record.update(extra)
            _write_timing_jsonl(record)