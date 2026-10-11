"""依 config.LLAMA_SERVER 啟動 llama-server。

設定的唯一來源是 config.py；本檔只負責：啟動前檢查 → 組指令 → 落檔紀錄 → 啟動。
用法（專案根目錄）：python -m tools.start_llama_server
"""
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def fail(msg: str) -> None:
    print(f"\n❌ {msg}\n", file=sys.stderr)
    sys.exit(1)


def check_files(cfg: dict) -> None:
    for key in ("exe", "model"):
        p = Path(cfg[key])
        if not p.is_file():
            fail(f"LLAMA_SERVER['{key}'] 找不到檔案：{p}")


def check_build(cfg: dict) -> None:
    """config 記的版號必須與實際執行檔一致，否則文件記的版本是假的。"""
    out = subprocess.run(
        [cfg["exe"], "--version"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    text = (out.stdout + out.stderr).strip()
    # 現行格式：「version: 0.6.0-dev (build 11541, commit …)」；舊格式：「version: 11541 (…)」
    m = re.search(r"build\s+(\d+)", text) or re.search(r"version:\s*(\d+)\s*\(", text)
    if not m:
        fail(f"讀不到 llama-server 版號，原始輸出：\n{text}")
    expected = cfg["build"].lstrip("b")
    if m.group(1) != expected:
        fail(f"版號不符：config 寫 {cfg['build']}，執行檔是 b{m.group(1)}")


def check_port_free(cfg: dict) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((cfg["host"], cfg["port"])) == 0:
            fail(f"{cfg['host']}:{cfg['port']} 已有程式在聽（上一個 llama-server 沒關？）")


def warn_if_lmstudio_running() -> None:
    """兩個後端同時載模型會搶 VRAM，觸發 Sysmem Fallback 無聲降速。只警告不擋。"""
    out = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if "LM Studio.exe" in out.stdout:
        print("⚠️ LM Studio 正在執行：若它載著模型，請先卸載或關閉，否則會搶 VRAM。\n")


def build_cmd(cfg: dict, log_file: Path) -> list[str]:
    return [
        cfg["exe"],
        "-m", cfg["model"],
        "--alias", cfg["alias"],
        "--host", cfg["host"],
        "--port", str(cfg["port"]),
        "-c", str(cfg["ctx"]),
        "-ngl", str(cfg["ngl"]),
        "-np", str(cfg["parallel"]),
        "-t", str(cfg["threads"]),
        "-b", str(cfg["batch"]),
        "-ub", str(cfg["ubatch"]),
        "-fa", cfg["flash_attn"],
        "--jinja",
        "--metrics",
        "--log-file", str(log_file),
        *cfg["extra_args"],
    ]


def main() -> None:
    cfg = config.LLAMA_SERVER
    check_files(cfg)
    check_build(cfg)
    check_port_free(cfg)
    warn_if_lmstudio_running()

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"llama-server_{stamp}.log"
    cmd = build_cmd(cfg, log_file)

    # 完整指令落檔：日後比對延遲／施測時能查到當時用了哪組參數
    cmd_line = subprocess.list2cmdline(cmd)
    (LOG_DIR / f"llama-server_{stamp}.cmd.txt").write_text(cmd_line + "\n", encoding="utf-8")
    print(f"▶ {cmd_line}\n")

    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        print("\n已停止 llama-server。")


if __name__ == "__main__":
    main()
