# core/log_window.py
# 工具執行時另開一個 log 視窗，即時顯示「該次執行資料夾」底下最新的每日 log
"""
為什麼有這支（2026-11-21 邪狼，待辦 C24）
  AOFOS 本體的 log 視窗是 start/log.bat，盯的是 logs/daily/。工具（對話探測、自動施測）的 log 寫在
  自己的執行資料夾，原本只印在工具自己的視窗、和進度摘要混在一起；自動施測的子行程 log 更是跑完才看得到。
  ⇒ 需要的工具在執行時自己跳一個對應的 log 視窗；不動、不接管已經開著的其他 log 視窗。

哪些工具用（判準：工具視窗本身要看結果、而 log 量大會把結果洗掉，或 log 平常看不到）：
  tools/chat_probe（同一行程跑正式路徑，log 量大）、tools/eval_runner（每題子行程的 log 平常看不到）。
  不用的：judge_worker（它的視窗本來就是 log）、start_llama_server（視窗就是後端輸出）、
  memory_review／memory_peek／judge_examples／backup_memory（短、輸出就是結果）。

視窗行為
  · 每 0.5 秒找資料夾（含子資料夾）裡最新的 YYYY-MM-DD.log，從頭印、之後跟著印新增的整行；
    出現更新的檔（施測換下一題）就換過去，換檔時印一行檔名；換回看過的檔從上次的位置接著印。
    child.log 這類其他檔不看。
  · 開它的那個行程結束後再讀最後一次就停，視窗留著（-NoExit），看完自己關。
  · 只在 Windows 開；開不了（不是 Windows、沒有 powershell、config 關掉）回 False，
    呼叫端就照舊把 log 印在自己的視窗——不會因為少一個視窗而看不到 log。
"""
from __future__ import annotations

import base64
import os
import subprocess

from config import LOG_CONFIG, PROJECT_ROOT

# 腳本用 -EncodedCommand（UTF-16LE base64）送，路徑、標題、主程式 PID 走環境變數：
#    命令列裡不出現引號、換行、中文，不必處理 cmd／powershell 兩層跳脫。
# 只吃到最後一個換行為止：寫到一半的那一行（或被切開的中文字）留到下一圈再印。
_PS = r"""
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = $env:AOFOS_TAIL_TITLE
$d = $env:AOFOS_TAIL_DIR; $owner = [int]$env:AOFOS_TAIL_PID
$cur = $null; $pos = 0; $seen = @{}   # 每個檔讀到哪：換回同一個檔時接著印，不從頭重印
Write-Host ('log 資料夾：' + $d) -ForegroundColor DarkGray
while ($true) {
  $gone = -not (Get-Process -Id $owner -ErrorAction SilentlyContinue)
  $f = Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue |
       Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}\.log$' } |
       Sort-Object LastWriteTime | Select-Object -Last 1
  if ($f -and $f.FullName -ne $cur) {
    if ($cur) { $seen[$cur] = $pos }
    $cur = $f.FullName
    $pos = if ($seen.ContainsKey($cur)) { $seen[$cur] } else { 0 }
    Write-Host ('──── ' + $cur.Substring($d.Length).TrimStart('\') + ' ────') -ForegroundColor DarkGray
  }
  if ($cur) {
    try {
      $fs = [IO.File]::Open($cur, 'Open', 'Read', 'ReadWrite, Delete')
      try {
        if ($fs.Length -gt $pos) {
          [void]$fs.Seek($pos, 'Begin')
          $buf = New-Object byte[] ($fs.Length - $pos)
          $n = $fs.Read($buf, 0, $buf.Length)
          $last = [Array]::LastIndexOf($buf, [byte]10, $n - 1)
          if ($last -ge 0) {
            [Console]::Write([Text.Encoding]::UTF8.GetString($buf, 0, $last + 1))
            $pos += $last + 1
          }
        }
      } finally { $fs.Close() }
    } catch { }
  }
  if ($gone) { Write-Host "`n（主程式已結束，這個視窗可以關掉）" -ForegroundColor DarkGray; break }
  Start-Sleep -Milliseconds 500
}
"""


def open_log_window(title: str, folder: str) -> bool:
    """另開視窗即時顯示 folder 底下最新的每日 log；開成功回 True（呼叫端可讓自己的視窗不再印 log）。"""
    if os.name != "nt" or not LOG_CONFIG.get("tool_log_window", True):
        return False
    env = dict(os.environ, AOFOS_TAIL_DIR=os.path.abspath(folder),
               AOFOS_TAIL_TITLE=title, AOFOS_TAIL_PID=str(os.getpid()))
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NoExit", "-EncodedCommand",
             base64.b64encode(_PS.encode("utf-16-le")).decode("ascii")],
            cwd=PROJECT_ROOT, env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,   # 獨立視窗；不接管、不影響其他開著的 log 視窗
        )
    except OSError as e:
        print(f"⚠️ 開不了 log 視窗（{type(e).__name__}: {e}），log 照舊印在這個視窗")
        return False
    return True
