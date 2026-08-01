# core/logger.py
# 統一 Logging 系統 — 所有模組從這裡取得 logger；並提供延遲量測工具 timed()

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime

import colorlog

from config import TIMING_CONFIG

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_LEVEL = logging.DEBUG

# 所有計時輸出共用同一個 logger，方便單獨調整等級或關閉
_timing_logger = None


def setup_logger(name: str) -> logging.Logger:
    """建立並回傳指定名稱的 logger，同時輸出到終端機（彩色）與 logs/（純文字，每日一份）。"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重複加 handler

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

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def _get_timing_logger() -> logging.Logger:
    global _timing_logger
    if _timing_logger is None:
        _timing_logger = setup_logger("TIMING")
    return _timing_logger


def _write_timing_jsonl(record: dict) -> None:
    """把單筆計時寫進 logs/timing-YYYY-MM-DD.jsonl。失敗不得影響主流程。"""
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

        with timed("RAG查詢"):
            results = rag.search(query)

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