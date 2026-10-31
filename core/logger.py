# core/logger.py
# 統一 Logging 系統 — 所有模組從這裡取得 logger；並提供延遲量測工具 timed()

import json
import logging
import os
import sys
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


class _ConsoleSink:
    """
    所有 logger 的終端 handler 共用這一個 stream；寫入當下才決定去哪、印多少：
    有登記 writer（core/console.py 啟動時登記）就交給它，沒有就寫 stderr；
    level 以下的不印（每個終端 handler 的 filter 看這裡）。預設 DEBUG＝全部印，tools/ 腳本的終端用這個；
    AOFOS 對話視窗登記 writer 時一併給自己的等級（config LOG_CONFIG["dialog_level"]）。

    ⭐ 因為每個 handler 從建立起就指向這裡，之後才建立的 logger（例如第一次 timed()
       才誕生的 TIMING）也自動涵蓋，不必在啟動時回頭改接任何 handler。

    ⚠️ 相依方向刻意是「console 告訴 logger」而非「logger 去 import console」——
       logger 只決定記什麼、什麼格式，輸出到哪裡是終端層的事。
    """

    writer = None
    level = logging.DEBUG

    def passes(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.level

    def write(self, text: str) -> int:
        w = self.writer
        if w is None:
            sys.stderr.write(text)   # 當下的 sys.stderr，不是 import 時的那個
        else:
            w(text)
        return len(text)

    def flush(self) -> None:
        if self.writer is None:
            sys.stderr.flush()


_CONSOLE_SINK = _ConsoleSink()


def set_console_writer(writer, level: str | None = "DEBUG") -> None:
    """
    登記終端輸出函式（收到的是一筆已格式化、結尾帶換行的 log）與要印的最低等級（None＝都不印）。
    writer 傳 None 還原成 stderr。
    """
    _CONSOLE_SINK.writer = writer
    _CONSOLE_SINK.level = logging.CRITICAL + 1 if level is None else logging.getLevelName(level)


def _add_handlers(logger: logging.Logger) -> None:
    """終端（彩色，經 _ConsoleSink，印多少由它決定）＋ LOG_DIR 每日一份純文字檔（全部）。"""
    os.makedirs(LOG_DIR, exist_ok=True)

    console_handler = colorlog.StreamHandler(_CONSOLE_SINK)
    console_handler.addFilter(_CONSOLE_SINK.passes)
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


def setup_logger(name: str) -> logging.Logger:
    """建立並回傳指定名稱的 logger，同時輸出到終端機（彩色）與 LOG_DIR（純文字，每日一份）。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重複加 handler

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False  # 不往 root 傳：root 另有自己的 handler（見下方第三方套件），傳上去會印兩次
    _add_handlers(logger)
    return logger


# 第三方套件的 log 與 warnings 也走同一套出口：寫進每日 log 檔，終端印不印同上
# （不再繞過 console 直接寫 stderr、把輸入列推走）。只收 WARNING 以上（root 預設等級）：
# 第三方的 INFO／DEBUG 量大且與本專案無關。⚠️ 進度條（tqdm 等）直接寫 stderr、不經 logging，管不到。
logging.captureWarnings(True)
if not logging.getLogger().handlers:
    _add_handlers(logging.getLogger())


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