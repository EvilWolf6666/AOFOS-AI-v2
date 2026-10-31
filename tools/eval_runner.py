"""eval_runner：照《題本》自動施測一組模型，輸出成《施測記錄表》的組別區塊（blind_eval.py 可直接吃）。

為什麼有這支
  擂台一輪要測好幾版 × 20 題、每題重開。手動貼輸入、複製回答，漏貼或貼錯不會有任何警告。

做法（與《題本》〈施測〉手動流程的對應）
  · 「啟動程式」：每一題開一個新的 Python 行程（_one 子命令），Working Memory、測試區都是全新的。
  · 「/test on」：呼叫 Pipeline._set_test_mode(True)——/test on 指令走的就是這一個函式
    （記憶隔離＋貪婪解碼＋關串流）。進場後檢查 llm.test_mode、memory.test_mode 都為真，否則該題作廢。
  · 「依序貼輸入」：每一輪用 Pipeline.handle_turn(Event("user_text", UTTERANCE, 輸入))，
    與 chat_probe.py 同一條正式路徑（LLMClient＋Persona＋MemoryManager＋Pipeline），不另寫對話邏輯。
  · 「確認 log」：開跑前查後端 /props（模型檔）、/v1/models（alias）、/lora-adapters（LoRA 與 scale），
    與 config.LLAMA_SERVER 不符就不跑。

與手動流程刻意不同的地方
  · 不經 Agent：不開視窗監看、不跑主動發話，只有題目本身的輸入。
  · 記憶用每題各自的暫存空庫（logs/runs/eval/<批次>/eval_<alias>_<時間>/<題號>/），🔴 正式記憶檔一個字不動。
  · 每題子行程的 log 寫進該題資料夾（環境變數 AOFOS_LOG_DIR，core/logger.py），不混進日常 log。
  · 一個 ``` 區塊裡有兩行的題目（I1、I2）：視為同一則訊息，兩行直接接起來（換行是文件排版）。

prompt_tokens 比對
  只有「每題第一輪」跨模型應該相同（送出的內容一樣）；第二輪起帶著前一輪各自的回覆，本來就會不同。
  compare 子命令只把第一輪不一致當錯誤，其餘列出供參考。

用法（專案根目錄，推理後端先依 config 開好，AOFOS 本體要關著）：
  python -m tools.eval_runner run                        題本預設 evaluation/題本_v2.md、組別預設＝config 的 alias
  python -m tools.eval_runner run --batch 擂台第二輪       同一批測試放同一個資料夾（預設＝當天日期）
  python -m tools.eval_runner run --group 甲 --only D1,B1  只跑指定題（驗證用）；兩個參數都可另給
  python -m tools.eval_runner run --no-persona            不送人格 system（《現況》第八章 C22 補測；組別與資料夾自動加 _nopersona）
  start\\自動施測_無人格.bat                                同上一行（雙擊用）
  start\\自動施測.bat                                       同第一行（雙擊用）
  python -m tools.eval_runner compare <results.json> <results.json> ...                       比對第一輪 prompt_tokens
  python -m tools.eval_runner combine <輸出記錄表.md> <組別.md> <組別.md> ...                  把各組區塊併成一份記錄表

離開碼：0 正常／1 條件不符或有題目失敗／2 找不到檔案
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime

# 記錄表的題目順序（同《施測記錄表_v2模板》：計分 14 題在前、驗證 6 題在後）
SCORED = ["B1", "B2", "B3", "C1", "C2", "D1", "D2", "D3", "E1", "G1", "G2", "H1", "I1", "I2"]
CHECK = ["F1", "F2", "A1", "A2", "O1", "O2"]
ORDER = SCORED + CHECK
# v2《題本》各題輪數。只拿來核對解析結果（解析錯了要大聲），題目內容一律從題本讀。
EXPECTED_TURNS_V2 = {"F1": 1, "F2": 1, "A1": 1, "A2": 2, "B1": 3, "B2": 4, "B3": 3, "C1": 3, "C2": 3,
                     "D1": 1, "D2": 3, "D3": 2, "E1": 3, "G1": 6, "G2": 5, "H1": 3, "I1": 3, "I2": 3,
                     "O1": 1, "O2": 1}
QID_RE = re.compile(r"^###\s+([A-Z]\d+)(?:\b|｜)")
CHILD_TIMEOUT_S = 1800


def fail(msg: str, code: int = 1) -> None:
    print(f"\n❌ {msg}\n", file=sys.stderr)
    sys.exit(code)


# ── 題本解析 ─────────────────────────────────────────────────────────

def parse_questions(path: str) -> dict[str, list[str]]:
    """只取 <!-- 給評審 --> 與 <!-- /給評審 --> 之間的 ``` 區塊；每個區塊＝一輪輸入。"""
    if not os.path.isfile(path):
        fail(f"找不到題本：{path}", 2)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not any("版本：v2" in ln for ln in lines[:5]):
        fail("題本開頭找不到「版本：v2」。本腳本的輪數核對表只對 v2 成立，換題本版本要先改 EXPECTED_TURNS_V2。")
    qs: dict[str, list[str]] = {}
    cur, in_rev, in_fence, buf = None, False, False, []
    for ln in lines:
        s = ln.strip()
        if not in_fence:
            m = QID_RE.match(ln)
            if m:
                cur = m.group(1)
                continue
        if s == "<!-- 給評審 -->":
            in_rev = True
            continue
        if s == "<!-- /給評審 -->":
            in_rev = False
            continue
        if in_rev and cur and s.startswith("```"):
            if in_fence:
                qs.setdefault(cur, []).append("".join(x.strip() for x in buf))
                buf, in_fence = [], False
            else:
                in_fence = True
            continue
        if in_fence:
            buf.append(ln)
    got = {q: len(t) for q, t in qs.items()}
    if got != EXPECTED_TURNS_V2:
        diff = {q: (got.get(q), EXPECTED_TURNS_V2.get(q)) for q in set(got) | set(EXPECTED_TURNS_V2)
                if got.get(q) != EXPECTED_TURNS_V2.get(q)}
        fail(f"題本解析結果與 v2 輪數不符（題號：(解析到, 應為)）：{dict(sorted(diff.items()))}")
    if any(not t for v in qs.values() for t in v):
        fail("題本有空白的輸入輪。")
    return qs


# ── 後端狀態核對 ─────────────────────────────────────────────────────

def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _same_path(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def check_backend(cfg: dict) -> dict:
    root = f"http://{cfg['host']}:{cfg['port']}"
    bad: list[str] = []
    try:
        props = _get(f"{root}/props")
        models = [m.get("id") for m in _get(f"{root}/v1/models").get("data", [])]
        loras = _get(f"{root}/lora-adapters")
    except Exception as e:  # noqa: BLE001
        fail(f"連不上推理後端 {root}（先用 start\\啟動推理後端.bat 開好）：{type(e).__name__}: {e}")
    if not _same_path(props.get("model_path"), cfg["model"]):
        bad.append(f"後端載入的模型檔 {props.get('model_path')} ≠ config 的 {cfg['model']}")
    if cfg["alias"] not in models:
        bad.append(f"後端的 model id {models} 不含 config 的 alias「{cfg['alias']}」（改了 config 沒重開後端？）")
    want = cfg.get("optional_args", {}).get("--lora")
    active = [x for x in loras if float(x.get("scale", 0)) != 0]
    if want:
        if len(active) != 1 or not _same_path(active[0].get("path"), want) or float(active[0]["scale"]) != 1:
            bad.append(f"LoRA 不符：config 要 {want}（scale 1），後端是 {loras}")
    elif active:
        bad.append(f"config 沒有 --lora，但後端掛著作用中的 LoRA：{active}")
    if bad:
        fail("後端狀態與 config 不符，不跑：\n  " + "\n  ".join(bad))
    return {"build_info": props.get("build_info"), "model_path": props.get("model_path"),
            "models": models, "lora_adapters": loras}


# ── 單題（子行程）────────────────────────────────────────────────────

def run_one(qid: str, turns_file: str, out_file: str) -> int:
    import config  # noqa: F401  必須最先 import（HF_HUB_OFFLINE，見 main.py 開頭）
    from core.agent.events import UTTERANCE, Event
    from core.agent.pipeline import Pipeline
    from core.llm_client import LLMClient
    from core.memory.manager import MemoryManager
    from core.persona import Persona

    with open(turns_file, encoding="utf-8") as f:
        turns = json.load(f)
    work = os.path.dirname(out_file)
    # --no-persona 由父行程以環境變數傳入；結果記下實際狀態，父行程逐題核對。
    no_persona = os.environ.get("AOFOS_EVAL_NO_PERSONA") == "1"
    rec: dict = {"qid": qid, "turns": [], "error": None, "persona": not no_persona}
    memory = None
    try:
        llm = LLMClient()
        persona = Persona()
        memory = MemoryManager(llm, db_path=os.path.join(work, "memory.db"),
                               test_db_path=os.path.join(work, "memory_test.db"))
        pipeline = Pipeline(llm, persona, memory)

        usage_log: list[dict] = []
        orig_usage = llm._check_usage

        def check_usage(usage, purpose, max_tokens):
            if purpose == "dialogue":
                usage_log.append(None if usage is None else
                                 {"prompt_tokens": usage.prompt_tokens,
                                  "completion_tokens": usage.completion_tokens})
            return orig_usage(usage, purpose, max_tokens)

        sent: list = []
        orig_build = persona.build_messages

        def build(history, user_input):
            msgs = orig_build(history, user_input)
            if no_persona:
                # 只拿掉開頭的人格 system；開頭不是它就報錯（Persona.build_messages 形狀變了）。
                if not (msgs and msgs[0].get("role") == "system" and msgs[0].get("content") == persona.system_prompt):
                    raise RuntimeError("--no-persona：送出的第一則不是人格 system，Persona.build_messages 形狀可能改了")
                msgs = msgs[1:]
            sent.append(msgs)
            return msgs

        llm._check_usage, persona.build_messages = check_usage, build

        pipeline._set_test_mode(True)   # ＝ /test on
        if not (llm.test_mode and memory.test_mode):
            raise RuntimeError(f"測試模式沒開成（llm={llm.test_mode}, memory={memory.test_mode}）")

        for t in turns:
            nu, ns = len(usage_log), len(sent)
            res = pipeline.handle_turn(Event("user_text", UTTERANCE, t))
            pipeline.finalize_turn()
            u = usage_log[nu] if len(usage_log) > nu else None
            rec["turns"].append({
                "input": t,
                "reply": res.reply,
                "prompt_tokens": u and u["prompt_tokens"],
                "completion_tokens": u and u["completion_tokens"],
                "messages": sent[ns] if len(sent) > ns else None,
            })
    except Exception as e:  # noqa: BLE001  錯誤寫進結果，由父行程大聲列出
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if memory is not None:
            try:
                memory.compress_on_exit()   # 測試模式下＝丟棄測試區、關連線（暫存資料夾裡的）
            except Exception as e:  # noqa: BLE001
                rec["error"] = (rec["error"] or "") + f"｜收尾失敗：{type(e).__name__}: {e}"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    return 0 if rec["error"] is None else 1


# ── 一組（父行程）────────────────────────────────────────────────────

def _quant(path: str) -> str:
    m = re.search(r"(I?Q\d_[A-Z0-9_]+|F16|BF16|Q8_0)", os.path.basename(path), re.I)
    return m.group(1).upper() if m else "（檔名看不出）"


def render_group(group: str, meta: dict, results: dict[str, dict]) -> str:
    out = [f"## 組別：{group}", "", "```"]
    for k in ("模型", "量化", "LoRA", "推理後端", "GPU 卸載層數", "後端回報 VRAM",
              "測試模式開法（/test on 或 config）", "施測日期"):
        out.append(f"{k}：{meta.get(k, '')}")
    out += ["```", "", "### ── 計分題（14）──", ""]

    def block(q: str, scored: bool) -> None:
        out.append(f"### {q}")
        out.append("<<<")
        rounds = [f"你：{t['input']}\n奧狐：{t['reply']}" for t in results[q]["turns"]]
        out.append("\n\n".join(rounds))
        out.append(">>>")
        if scored:
            out.append("**評分**：_/2")
        out.append("")

    for q in SCORED:
        block(q, True)
    out += ["### ── 驗證題（6，不評分）──", ""]
    for q in CHECK:
        block(q, False)
    return "\n".join(out).rstrip() + "\n"


def cmd_run(args) -> int:
    import config
    from core.memory.run_lock import RunLock, lock_path

    cfg = config.LLAMA_SERVER
    if args.questions is None:
        args.questions = os.path.join(config.PROJECT_ROOT, "evaluation", "題本_v2.md")
    if args.group is None:
        args.group = cfg["alias"] + ("_nopersona" if args.no_persona else "")
    qs = parse_questions(args.questions)
    order = ORDER
    if args.only:
        order = [q.strip().upper() for q in args.only.split(",") if q.strip()]
        unknown = [q for q in order if q not in qs]
        if unknown:
            fail(f"--only 有題本裡沒有的題號：{unknown}")

    backend = check_backend(cfg)
    lock = RunLock(lock_path(config.MEMORY_CONFIG["db_path"]))
    if not lock.acquire():
        fail("AOFOS 正在執行（記憶檔被占用）。先關掉 AOFOS 再跑——兩邊會搶同一個推理槽。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 同一批測試的各組放在同一個資料夾：logs/runs/eval/<批次>/eval_<alias>_<時間>/
    #   批次預設＝當天日期（雙擊 bat 不帶參數時，同一天跑的各組自動歸在一起）。
    batch = args.batch or datetime.now().strftime("%Y-%m-%d")
    if any(c in batch for c in '\\/:*?"<>|'):
        fail(f"--batch 不能含路徑字元：{batch}")
    tag = cfg["alias"] + ("_nopersona" if args.no_persona else "")
    root = os.path.join(config.PROJECT_ROOT, "logs", "runs", "eval", batch, f"eval_{tag}_{stamp}")
    os.makedirs(root, exist_ok=True)
    print(f"模型 {cfg['alias']}｜後端 {backend['build_info']}｜LoRA {backend['lora_adapters']}")
    print("人格 system：" + ("❗ 不送（--no-persona）" if args.no_persona else "照常送出"))
    print(f"結果資料夾：{root}\n")

    results: dict[str, dict] = {}
    failed: list[str] = []
    try:
        for i, q in enumerate(order, 1):
            qdir = os.path.join(root, q)
            os.makedirs(qdir, exist_ok=True)
            tf, of = os.path.join(qdir, "turns.json"), os.path.join(qdir, "result.json")
            with open(tf, "w", encoding="utf-8") as f:
                json.dump(qs[q], f, ensure_ascii=False)
            print(f"[{i}/{len(order)}] {q}（{len(qs[q])} 輪）…", flush=True)
            env = dict(os.environ, PYTHONUTF8="1", AOFOS_LOG_DIR=qdir,
                       AOFOS_EVAL_NO_PERSONA="1" if args.no_persona else "0")
            try:
                p = subprocess.run([sys.executable, "-m", "tools.eval_runner", "_one", q, tf, of],
                                   cwd=config.PROJECT_ROOT, env=env, timeout=CHILD_TIMEOUT_S,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                with open(os.path.join(qdir, "child.log"), "w", encoding="utf-8") as f:
                    f.write(p.stdout + "\n──── stderr ────\n" + p.stderr)
            except subprocess.TimeoutExpired:
                failed.append(f"{q}：逾時（{CHILD_TIMEOUT_S} 秒）")
                break
            if not os.path.isfile(of):
                failed.append(f"{q}：子行程沒寫出結果（看 {qdir}\\child.log）")
                break
            with open(of, encoding="utf-8") as f:
                r = json.load(f)
            if r.get("error"):
                failed.append(f"{q}：{r['error']}（看 {qdir}\\child.log）")
                break
            if r.get("persona") is not (not args.no_persona):
                failed.append(f"{q}：子行程的人格狀態與 --no-persona 不符（{r.get('persona')}）")
                break
            empties = [n for n, t in enumerate(r["turns"], 1) if not t["reply"]]
            if empties:
                print(f"   ⚠️ 第 {empties} 輪空回覆（照原樣記，不補字）")
            for n, t in enumerate(r["turns"], 1):
                print(f"   {n}. prompt_tokens={t['prompt_tokens']}｜{(t['reply'] or '（空）')[:40]}")
            results[q] = r
    finally:
        lock.release()

    lora = cfg.get("optional_args", {}).get("--lora")
    meta = {
        "模型": f"{cfg['alias']}（{os.path.basename(cfg['model'])}）" + ("｜不送人格 system" if args.no_persona else ""),
        "量化": _quant(cfg["model"]),
        "LoRA": os.path.basename(lora) if lora else "無（模型檔本身；合併版或底模）",
        "推理後端": f"llama-server {backend['build_info']}",
        "GPU 卸載層數": cfg.get("ngl", ""),
        "後端回報 VRAM": "",
        "測試模式開法（/test on 或 config）": "/test on（tools/eval_runner.py 呼叫同一函式）",
        "施測日期": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(os.path.join(root, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"group": args.group, "persona": not args.no_persona, "meta": meta, "backend": backend, "order": order,
                   "results": results, "failed": failed}, f, ensure_ascii=False, indent=2)

    if failed:
        print("\n❌ 有題目失敗，本組不產生記錄表區塊：\n  " + "\n  ".join(failed))
        return 1
    if order != ORDER:
        print(f"\n✅ 指定題跑完（{','.join(order)}）。只跑部分題目時不產生記錄表區塊；結果在 {root}\\results.json")
        return 0
    md = os.path.join(root, f"記錄表.{args.group}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(render_group(args.group, meta, results))
    print(f"\n✅ 20 題完成。\n   記錄表區塊：{md}\n   完整紀錄：{os.path.join(root, 'results.json')}")
    print("   「後端回報 VRAM」請照 llama-server 啟動 log 手動補上。")
    return 0


def cmd_compare(args) -> int:
    runs = []
    for p in args.results:
        if not os.path.isfile(p):
            fail(f"找不到 {p}", 2)
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    names = [r["meta"]["模型"].split("（")[0] for r in runs]
    print("第一輪 prompt_tokens（跨模型必須相同；不同＝模板或送出內容沒對齊，該組分數作廢）")
    print(f"{'題號':<6}" + "".join(f"{n:>16}" for n in names))
    bad = []
    for q in ORDER:
        vals = [(r["results"].get(q) or {}).get("turns", [{}])[0].get("prompt_tokens") for r in runs]
        mark = "" if len(set(vals)) == 1 and None not in vals else "  ❌"
        if mark:
            bad.append(q)
        print(f"{q:<6}" + "".join(f"{str(v):>16}" for v in vals) + mark)
    if bad:
        print(f"\n❌ 第一輪不一致：{bad}")
        return 1
    print("\n✅ 20 題第一輪全部一致（第二輪起含各自回覆，本來就不同，不比）")
    return 0


def cmd_combine(args) -> int:
    parts = []
    for p in args.groups:
        if not os.path.isfile(p):
            fail(f"找不到 {p}", 2)
        with open(p, encoding="utf-8") as f:
            parts.append(f.read().strip())
    head = ("# AOFOS 施測記錄表　v2 題本\n\n"
            f"> 題本版本：`v2`　由 tools/eval_runner.py 產生並合併（{datetime.now():%Y-%m-%d}）\n"
            "> 格式同《施測記錄表_v2模板》，由 `blind_eval.py` 解析，不要改結構。\n")
    if os.path.exists(args.out):
        fail(f"{args.out} 已存在，不覆寫。換個檔名或先自己移走。")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(head + "\n---\n\n" + "\n\n---\n\n".join(parts) + "\n")
    print(f"✅ 已合併 {len(parts)} 組：{args.out}")
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "_one":
        if len(sys.argv) != 5:
            fail("_one 需要：題號 turns.json result.json")
        return run_one(sys.argv[2], sys.argv[3], sys.argv[4])
    ap = argparse.ArgumentParser(description="照《題本》自動施測")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--questions", default=None, help="《題本》v2 的路徑（預設 evaluation/題本_v2.md）")
    r.add_argument("--group", default=None, help="記錄表的組別名稱（預設＝config 的 alias）")
    r.add_argument("--batch", default=None, help="批次資料夾名稱（logs/runs/eval/ 底下；預設＝當天日期）")
    r.add_argument("--only", default=None, help="只跑指定題號，逗號分隔（驗證用，不產生記錄表區塊）")
    r.add_argument("--no-persona", action="store_true", help="不送人格 system（C22 補測）；組別與資料夾自動加 _nopersona")
    c = sub.add_parser("compare")
    c.add_argument("results", nargs="+", help="各組的 results.json")
    m = sub.add_parser("combine")
    m.add_argument("out", help="輸出的記錄表路徑（已存在就不覆寫）")
    m.add_argument("groups", nargs="+", help="各組的 記錄表.<組>.md")
    args = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
