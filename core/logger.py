# core/logger.py
# 統一 Logging 系統 — 所有模組從這裡取得 logger

import logging
import os
from datetime import datetime
import colorlog

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_LEVEL = logging.DEBUG


def setup_logger(name: str) -> logging.Logger:
    """
    建立並回傳指定名稱的 logger，同時輸出到終端機（彩色）與 logs/ 資料夾（純文字）。
    各模組呼叫一次、傳入模組名稱即可。
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重複加 handler

    logger.setLevel(LOG_LEVEL)

    # 終端機 handler（彩色）
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

    # 檔案 handler（純文字，每日一份）
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