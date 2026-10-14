#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blind_eval.py — AOFOS 施測記錄表的盲評工具（make / merge）

═══════════════════════════════════════════════════════════════════
本檔與 AOFOS 程式碼庫**完全不耦合**，這是刻意的：

  · 不 import 專案任何模組，特別是 **不 import config.py**
  · 只用 Python 標準庫，無第三方依賴
  · 獨立指令觸發，不掛進 pipeline._commands，不進 Agent 生命週期

理由同《歸檔》C0035 對 `chat_strata.py` 的判定（⚠️ 該檔已移出程式碼庫，
tools/ 底下現已無此檔，判定本體留在 C0035）：`tools/` 底下的腳本共用 config.py
唯一入口，而本工具不需要任何設定值，硬接會讓一個純文字處理工具綁上
`HF_HUB_OFFLINE` 那條啟動路徑。施測時推理後端可能正載著別的模型、
config.py 可能改到一半，本工具不該因此跑不起來。

⚠️ 日後若有人想「順手」讓它讀 config 取路徑——那就是耦合的入口。
   路徑一律由命令列給。
═══════════════════════════════════════════════════════════════════

為什麼要有這支工具
  去識別化如果靠手動，遲早會漏一次，而漏的那次**不會有任何警告**
  ——輸出看起來完全正常，只是 Claude 那欄的分數帶著期望偏差。
  （形狀同「靠自覺運作的規則會失效」，見 aofos-docs/cases.md。）

─── v2 相對於初版的四項改動（2026-08-26）───────────────────────

1. 🔴 **逐題洗牌，不是整組洗牌。**
   初版把一組十二題綁在同一個代號底下，F 類認出一題 → 該組八個計分題
   全部跟著曝光。改成每題內獨立洗牌後，一題曝光只損失一題。
   代價：對照表從「組 → 代號」變成「題 × 組 → 代號」矩陣，故 key 改用
   JSON，並新增 merge 子命令負責還原（人工還原這種矩陣必定會錯一次）。

2. 🔴 **抽取改白名單。**
   初版逐行黑名單刪除（評分行、prompt_tokens…），漏一個欄位就洩漏且
   無警告。改成只取 `<<<` 與 `>>>` 之間，其餘一律不輸出——
   **失敗方式從「多了不該有的東西」變成「少了東西」，後者看得見。**

3. 🔴 **F 類與 A 類預設不進盲評版。**
   F 類復現訓練資料、A 類語氣特徵，兩者都是身分標記。且 F 類的判讀
   本來就必須知道組別（要拿無 LoRA 組對照），盲評對它沒有意義。
   要放進來用 --include F,A。

4. 🔵 **--seed 省略時自動產生一個並寫進對照表。**
   初版省略時不可重現，事後想重跑同一組順序做不到。成本兩行。

離開碼
  0 = 正常
  1 = 記錄表格式不對 / 有殘留字串需人工確認
  2 = 找不到檔案
"""
import argparse
import json
import os
import random
import re
import sys
from collections import OrderedDict, defaultdict

# Windows 終端預設 cp950，直接 print emoji 會 UnicodeEncodeError。
# 這支工具在 VSCode 內建終端與 cmd 都要能跑，所以先把 stdout 轉成 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

GROUP_RE = re.compile(r'^##\s*組別[：:]\s*(.*)$')
QID_RE = re.compile(r'^###\s*([A-Z]\d+)\s*$')
SCORE_LINE_RE = re.compile(
    r'^\s*([A-Z]\d+-\d+)\s*[：:]\s*(\d+(?:\.\d+)?)\s*$')

# 分數尺度。0～2，允許 0.5 步進（2026-09-04 邪狼指定，由整數改為含小數）。
# 🔴 **刻意只放到 0.5，不放任意小數**：判準文件寫得出界線的只有「整格」與
#    「介於兩格之間」，1.37 這種數字沒有任何判準支撐，評分者只是在猜——
#    而猜出來的精確值，看起來與有依據的一模一樣。
SCORE_MIN, SCORE_MAX, SCORE_STEP = 0.0, 2.0, 0.5


def parse_score_value(raw, where):
    """把分數字串轉 float 並驗證尺度。不合法回 (None, 錯誤訊息)。"""
    try:
        v = float(raw)
    except ValueError:
        return None, f'{where}：「{raw}」不是數字'
    if not (SCORE_MIN <= v <= SCORE_MAX):
        return None, f'{where}：{raw} 超出 {SCORE_MIN:g}～{SCORE_MAX:g}'
    if abs(round(v / SCORE_STEP) * SCORE_STEP - v) > 1e-9:
        return None, f'{where}：{raw} 不是 {SCORE_STEP:g} 的倍數'
    return v, None


def fmt_score(v):
    """整數不帶小數點、非整數留一位——免得整張表變成 2.0／1.0 的雜訊。"""
    return '_' if v is None else f'{v:g}'

RAW_OPEN, RAW_CLOSE = '<<<', '>>>'

# metadata 區塊裡要抽出來做對照表的欄位
META_KEYS = ('模型', '量化', 'LoRA', 'GPU 卸載層數', 'GPU 卸載層數 / 後端回報 VRAM',
             '施測日期')
# 舊欄名 → 現行欄名（2026-10-14 改名：欄名不綁特定後端；舊施測紀錄照樣抽得到，存成現行欄名）
META_ALIASES = {'GPU 卸載層數 / LM Studio 估計 VRAM': 'GPU 卸載層數 / 後端回報 VRAM'}

# 計分類別。F（指紋）與 A（語氣）不計分，預設也不進盲評版。
# 沒有判準檔時的計分類別白名單（v1 題本的形狀）。
# ⚠️ **給了 --rubric 就不看這個**——判準檔列了哪些題，哪些題就計分。
#    理由：v2 新增 G／H／I 三個計分類別時，這個常數沒跟著改，
#    三類就被靜默當成不計分排除掉了（2026-09-05 端到端實跑抓到）。
#    ⇒ **計分範圍的單一來源是判準檔**，寫死字母只是沒有判準檔時的退路。
SCORED_PREFIXES = ('B', 'C', 'D', 'E')

# 題目與判準的區塊標頭。兩份文件的標題層級不同（一份 ###、一份 ##），
# 所以這裡刻意放寬到 ## ～ ####，只認題號本身。
# ⚠️ 放寬的代價是「章節標題剛好長得像題號」會被誤收，但那種標題不存在於這兩份文件。
SECTION_RE = re.compile(r'^#{2,4}\s*([A-Z]\d+)(?:[｜|│]|\s|$)')


# 題本裡「這一段可以給評審模型看」的白名單標記。
# 🔴 **為什麼是白名單而不是「把標準答案剝掉」**：黑名單漏一個就外流，
#    而外流的那份看起來完全正常；白名單漏一個是「少了東西」，評審會問。
#    形狀同本檔改動 2（抽取改白名單）——**讓失敗方式變成看得見的那一種。**
SHARE_OPEN = re.compile(r'^\s*<!--\s*給評審\s*-->\s*$')
SHARE_CLOSE = re.compile(r'^\s*<!--\s*/給評審\s*-->\s*$')


def parse_sections(path, require_marker=False):
    """
    從題本或判準文件切出 {題號: [該題的行]}。

    require_marker=True 時**只取 `<!-- 給評審 -->` 與 `<!-- /給評審 -->` 之間**，
    其餘一律不取——題本每題底下還有標準答案與判讀重點，那些是給邪狼看的，
    不能跟著進評分表：評審模型看到標準答案會往它收斂，
    **答得同樣好但寫法不同的回應會被扣分**，而那個扣分看起來完全合理。

    ⚠️ 這是**單一來源**機制的實作點：題目與判準只存在於各自那一份文件，
    評分表是每次跑出來的生成物，不是第二份抄本（同《現況》20之0
    「能由執行結果算出來的，不在文件裡抄第二份」）。
    ⚠️ 因此**不得**把生成出來的評分表回存成常駐文件——那會把副本數加回來。
    """
    out, cur, sharing = OrderedDict(), None, False
    with open(path, encoding='utf-8') as f:
        for line in f.read().split('\n'):
            m = SECTION_RE.match(line)
            if m:
                cur, sharing = m.group(1), False
                out.setdefault(cur, [])
                continue
            if cur is None:
                continue
            if require_marker:
                if SHARE_OPEN.match(line):
                    sharing = True
                    continue
                if SHARE_CLOSE.match(line):
                    sharing = False
                    continue
                if not sharing:
                    continue
            out[cur].append(line)
    # 去掉每段前後的空行，避免注入後版面鬆散
    for k, v in out.items():
        while v and not v[0].strip():
            v.pop(0)
        while v and not v[-1].strip():
            v.pop()
    return out



# ══════════════════════════ 解析 ══════════════════════════

def parse_record(lines):
    """
    回傳 (groups, samples)
      groups  = [{'title': 原標題, 'meta': {...}}]
      samples = OrderedDict[qid] -> {group_index: [原始輸出行]}

    只認三種東西：`## 組別：`、`### 題號`、`<<< >>>` 之間。
    其餘一律不進產物——白名單，見檔頭改動 2。
    """
    groups, samples = [], OrderedDict()
    gi = -1
    qid = None
    in_meta = in_raw = False
    buf = []

    for line in lines:
        s = line.strip()

        m = GROUP_RE.match(line)
        if m:
            if in_raw:
                raise ValueError(f'組別「{m.group(1).strip()}」之前有沒收尾的 <<< 區塊')
            groups.append({'title': m.group(1).strip(), 'meta': {}})
            gi += 1
            qid, in_meta = None, False
            continue

        if gi < 0:
            continue  # 第一個組別之前的說明文字，整段丟掉

        if in_raw:
            if s == RAW_CLOSE:
                in_raw = False
                samples.setdefault(qid, {})[gi] = buf
                buf = []
            else:
                buf.append(line)
            continue

        m = QID_RE.match(line)
        if m:
            qid = m.group(1)
            samples.setdefault(qid, {})
            continue

        if s == RAW_OPEN:
            if qid is None:
                raise ValueError(f'組別「{groups[gi]["title"]}」有 <<< 但上方沒有 ### 題號')
            in_raw, buf = True, []
            continue

        if s.startswith('```'):
            in_meta = not in_meta
            continue

        if in_meta:
            for k in (*META_KEYS, *META_ALIASES):
                mm = re.match(rf'^\s*{re.escape(k)}\s*[：:]\s*(.*)$', line)
                if mm and mm.group(1).strip():
                    groups[gi]['meta'][META_ALIASES.get(k, k)] = mm.group(1).strip()

    if in_raw:
        raise ValueError('檔尾有沒收尾的 <<< 區塊')
    return groups, samples


def collect_secrets(groups):
    """組別身分字串，用於事後殘留檢查（不用於自動取代）。"""
    secrets = set()
    for g in groups:
        for k in ('模型', 'LoRA'):
            v = g['meta'].get(k, '')
            if v:
                secrets.add(v)
                secrets.update(re.findall(r'[A-Za-z0-9_.\-]{4,}', v))
    return {s for s in secrets if s}


# ══════════════════════════ make ══════════════════════════

def cmd_make(args):
    if not os.path.isfile(args.record):
        print(f'🔴 找不到檔案：{args.record}', file=sys.stderr)
        return 2

    with open(args.record, encoding='utf-8') as f:
        lines = f.read().split('\n')

    try:
        groups, samples = parse_record(lines)
    except ValueError as e:
        print(f'🔴 記錄表格式錯誤：{e}', file=sys.stderr)
        return 1

    if not groups:
        print('🔴 找不到任何 "## 組別：" 區塊，請確認記錄表格式。', file=sys.stderr)
        return 1

    # 前置檢查一：metadata 沒填就大聲失敗。
    # 靜默通過的話對照表會是空的，而盲評版看起來完全正常。
    missing = [g['title'] for g in groups if not g['meta'].get('模型')]
    if missing:
        print('🔴 下列組別沒有填「模型：」，對照表會失去意義：', file=sys.stderr)
        for t in missing:
            print(f'   組別：{t}', file=sys.stderr)
        return 1

    # ── 題目與判準（外部評審模型專用）─────────────────────────────
    # 邪狼與 Claude 評分時不需要這兩塊（題目本來就在手上）；
    # 外部模型不認識奧狐、也沒有題本，不注入就等於讓它憑印象打分。
    # 🔴 兩者省略時本函式行為與注入前完全相同（向後相容）。
    # 🔴 題本走**白名單**：只取 `<!-- 給評審 -->` 內的題目情境，
    #    標準答案與判讀重點留在題本裡，不進評分表（見 parse_sections）。
    questions = parse_sections(args.questions, require_marker=True) \
        if args.questions else None
    rubric = parse_sections(args.rubric) if args.rubric else None

    # 缺題必須大聲失敗：靜默輸出一份缺題目的評分表時，
    # 評審模型仍然會給出分數，而那個分數看起來與正常的一模一樣。
    # 實際檢查排在 qids 算出來之後（見下方），這裡只放解析結果。

    include = {c.strip().upper() for c in args.include.split(',') if c.strip()}
    # 計分範圍：判準檔優先（單一來源），沒給才退回字母白名單。
    if rubric:
        scored_set = set(rubric)
        qids = [q for q in samples if q in scored_set or q[0] in include]
    else:
        scored_set = None
        qids = [q for q in samples
                if q[0] in SCORED_PREFIXES or q[0] in include]
    if not qids:
        print('🔴 沒有任何題目符合輸出條件（--include 目前是 '
              f'"{args.include}"）。', file=sys.stderr)
        return 1

    # 前置檢查一之二：題本／判準缺題就大聲失敗。
    # 靜默輸出一份缺題目的評分表時，評審模型仍然會給分，
    # 而那個分數看起來與正常的一模一樣。
    for label, src, hint in (
            ('題本', questions,
             '　（題本走白名單：每題的題目情境要用 <!-- 給評審 --> … '
             '<!-- /給評審 --> 包起來，沒包＝不給）'),
            ('判準', rubric, '')):
        if src is None:
            continue
        lack = [q for q in qids if not src.get(q)]
        if lack:
            print(f'🔴 {label}檔缺下列題號，或該題區塊是空的：{hint}',
                  file=sys.stderr)
            for q in lack:
                print(f'   {q}', file=sys.stderr)
            return 1

    # 前置檢查二：空輸出。
    # 空的 <<< >>> 靜默進盲評版時，那一題無法評分，而評分者看不出是漏貼還是模型沒回。
    holes = []
    for q in qids:
        for gi, g in enumerate(groups):
            body = samples[q].get(gi)
            if body is None or not any(l.strip() for l in body):
                holes.append(f'{q} / 組別：{g["title"]}')
    if holes:
        print('🔴 下列題目的原始輸出是空的，貼完再跑：', file=sys.stderr)
        for h in holes:
            print(f'   {h}', file=sys.stderr)
        return 1

    seed = args.seed if args.seed is not None else random.randrange(1, 10 ** 9)
    rng = random.Random(seed)

    head = [
        '# AOFOS 施測結果（盲評版）',
        '',
        '> 每一題的各組輸出**各自獨立打亂**，代號在題與題之間不對應——',
        '> `B1-1` 與 `B2-1` 不是同一組。',
        '> 模型名／量化／LoRA 版本／`prompt_tokens`／既有評分與備註皆未輸出。',
        '> `<<<` 與 `>>>` 之間為原始輸出，**未經任何修改**。',
        '',
    ]
    if questions or rubric:
        head += [
            '> 每題附**題目**與**判分界線**，依界線給 0／1／2 分；'
            '界線沒寫到的情況請就近類推，不要自創尺度。',
            '> 只評本題指定的那一輪。前面幾輪答得再好，指定輪沒答到就是沒答到。',
            '',
        ]
    else:
        head += [
            f'> B/C/D/E 每題 {SCORE_MIN:g}～{SCORE_MAX:g} 分，'
            f'依題本所附的通過條件評分。',
            '',
        ]
    head += ['---', '']

    body, mapping = [], {}
    for q in qids:
        order = list(range(len(groups)))
        rng.shuffle(order)
        body.append(f'## {q}')
        body.append('')
        if questions:
            body.append('<details open><summary>題目</summary>')
            body.append('')
            body.extend(questions[q])
            body.extend(['', '</details>', ''])
        if rubric:
            body.append('<details open><summary>判分界線</summary>')
            body.append('')
            body.extend(rubric[q])
            body.extend(['', '</details>', ''])
        for n, gi in enumerate(order, 1):
            code = f'{q}-{n}'
            mapping[code] = gi
            body.append(f'### {code}')
            body.append(RAW_OPEN)
            body.extend(samples[q][gi])
            body.append(RAW_CLOSE)
            body.append('')
        body.extend(['---', ''])

    # 填答區直接附在盲評版底部：評分者不必自己造格式，merge 也只認這一種。
    body.extend(['## 評分填寫區', '',
                 f'> 一行一個，格式 `代號: 分數`。尺度 {SCORE_MIN:g}～{SCORE_MAX:g}，'
                 f'可用 {SCORE_STEP:g} 分（例：`1.5`），**只到 {SCORE_STEP:g}，不要更細**——'
                 f'再細的數字沒有判準支撐。',
                 '> 介於兩格之間才用半分；不確定的留空不要猜。', ''])
    for q in qids:
        if (q in scored_set) if scored_set is not None else (q[0] in SCORED_PREFIXES):
            for n in range(1, len(groups) + 1):
                body.append(f'{q}-{n}: ')
    body.append('')

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.record))
    stem = os.path.splitext(os.path.basename(args.record))[0]
    key_path = os.path.join(outdir, f'{stem}.對照表.json')

    # 每位評審一份獨立檔案（邪狼 2026-09-04 指定：不共用同一份）。
    # ⚠️ 內容與 seed 相同、只是檔名不同——**分開的是填答與收回，不是題序**。
    #    刻意不為每位評審各洗一次牌：那會變成一位評審一張對照表，
    #    而 merge 目前設計為單一對照表；多表還原的錯誤不會有任何徵兆。
    raters = [r.strip() for r in args.raters.split(',') if r.strip()] if args.raters else []
    blind_text = '\n'.join(head + body)

    # 🔴 標準答案殘留檢查，**排在寫檔之前**。
    # 白名單已經擋掉絕大部分，這是最後一道；而它與下方的 collect_secrets
    # 不同——那邊的字串可能同時是評分依據，所以只定位不阻擋；
    # 標準答案則是一定不該在，且**一旦寫出去，磁碟上就躺著一份可以被誤送的檔案**。
    # ⇒ 先擋再產生（同《現況》0之4「先確認結果有效，再做不可逆的那一步」）。
    if questions or rubric:
        marks = [w for w in ('標準答案', '判讀重點', '參考答案', '正解')
                 if w in blind_text]
        if marks:
            print('🔴 評分表裡出現下列字眼，'
                  '**很可能是不該外流的段落漏在白名單內**：', file=sys.stderr)
            for w in marks:
                print(f'   {w}', file=sys.stderr)
            print('   評審模型看到標準答案會往它收斂——答得同樣好但寫法不同的'
                  '回應會被扣分，而那個扣分看起來完全合理。', file=sys.stderr)
            print('   ⚠️ 未產生任何檔案。', file=sys.stderr)
            return 1

    os.makedirs(outdir, exist_ok=True)   # 排在檢查之後：擋下時連目錄都不建
    written = []
    if raters:
        for r in raters:
            p = os.path.join(outdir, f'{stem}.盲評版.{r}.md')
            with open(p, 'w', encoding='utf-8', newline='\n') as f:
                f.write(blind_text)
            written.append(p)
    else:
        p = os.path.join(outdir, f'{stem}.盲評版.md')
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(blind_text)
        written.append(p)

    key = {
        'source': os.path.basename(args.record),
        'seed': seed,
        'groups': [{'title': g['title'], 'meta': g['meta']} for g in groups],
        # 計分題清單。merge 沒有 --rubric，**必須靠這裡才知道哪些題進總分**，
        # 否則它只能回頭猜字母白名單，而 v2 新增的類別不在那個白名單裡。
        'scored': sorted(scored_set) if scored_set is not None else None,
        'map': mapping,
    }
    with open(key_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    for p in written:
        print(f'✅ {p}')
    print('   → 一位評分者一份，各自填、各自收回'
          if raters else '   → 給 Claude')
    print(f'✅ {key_path}')
    print('   → 自己留著，交分數前不要打開給評分者看')
    print()
    print(f'共 {len(groups)} 組 × {len(qids)} 題，逐題各自打亂。seed={seed}')
    if raters:
        print(f'評審 {len(raters)} 位：{"、".join(raters)}'
              '（內容與題序相同，分開的是填答與收回）')
    if questions or rubric:
        got = [n for n, v in (('題目', questions), ('判分界線', rubric)) if v]
        print(f'已注入：{"、".join(got)}')
    skipped = [q for q in samples if q not in qids]
    if skipped:
        why = ('不在判準檔的計分題清單內' if scored_set is not None
               else 'F/A 類預設不盲評')
        print(f'（未輸出：{"、".join(skipped)}——{why}，'
              f'要放進來用 --include）')

    # 殘留檢查：只定位、不自動改。
    # 模型自稱（「我是 Qwen」）既是識別線索、也是 A/D 類的判讀材料，
    # 自動遮蔽會刪掉評分依據——工具負責枚舉與定位，判斷交給人。
    leaked = sorted(x for x in collect_secrets(groups) if x in blind_text)
    if leaked:
        print()
        print('⚠️ 盲評版的原始輸出裡出現下列字串，**工具不自動改**：')
        for x in leaked:
            print(f'   {x}')
        print('   純識別線索 → 手動刪；同時是評分依據（模型自稱等）→ 留著，')
        print('   並知道該題對評分者而言等於未盲。')
        return 1

    print('✅ 殘留檢查：未出現 metadata 中的模型名字串。')
    print('⚠️ 但風格特徵去不掉——emoji 密度、中英夾雜、回應長度都會洩漏身分。')
    print('   風格差異大的組別，去識別化實際上無效。')

    # 外發自我審查：本工具無法判斷什麼算實作細節，只負責提醒一次。
    # 形狀同上方的殘留檢查——工具枚舉、判斷交給人。
    if raters or questions or rubric:
        print()
        print('⚠️ 這份會送到專案外的模型手上。送出前自己過目一次：')
        print('   注入的題目與判準裡不得出現實作細節（模組名、檔名、參數、架構）。')
        print('   判準是「這段文字會不會流到第三方手上」，不是「有沒有幫忙」。')
    return 0


# ══════════════════════════ merge ══════════════════════════

def parse_scores(path):
    """讀填好的盲評版或純分數檔，回傳 {代號: 分數}。空值略過。

    ⚠️ 尺度不合法時**不靜默略過**——略過的話那一題會變成「沒填」，
    而「填錯」與「沒填」在下游長得一模一樣。
    """
    out, bad = {}, []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            m = SCORE_LINE_RE.match(line.rstrip('\n'))
            if m:
                v, err = parse_score_value(
                    m.group(2), f'{os.path.basename(path)} 第 {i} 行 {m.group(1)}')
                if err:
                    bad.append(err)
                else:
                    out[m.group(1)] = v
    return out, bad


def parse_record_scores(path, titles):
    """
    直接從施測記錄表讀邪狼那欄，回傳 {group_index: {qid: score}}。

    ⚠️ 為什麼要有這條路徑：他的分數本來就填在記錄表的 `**評分**：__/2` 裡，
    若要求他再手抄成 `代號: 分數`，等於一邊查對照表一邊轉 24 行——
    **抄錯不會有任何警告**，而且同一份分數會存在兩個地方。
    這裡讓記錄表維持唯一來源。
    """
    score_re = re.compile(r'^\s*\*\*評分\*\*\s*[：:]\s*(\d+(?:\.\d+)?)\s*/\s*2')
    flag_re = re.compile(r'\[([RXNP])\]')
    out, flags = defaultdict(dict), defaultdict(lambda: defaultdict(int))
    bad_scores = []
    gi, qid = -1, None
    order = {t: i for i, t in enumerate(titles)}
    unknown_titles = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = GROUP_RE.match(line)
            if m:
                t = m.group(1).strip()
                gi = order.get(t, -1)
                if gi < 0:
                    unknown_titles.append(t)
                qid = None
                continue
            m = QID_RE.match(line)
            if m:
                qid = m.group(1)
                continue
            if gi >= 0 and qid:
                for f in flag_re.findall(line):
                    flags[gi][f] += 1
            m = score_re.match(line)
            if m and gi >= 0 and qid:
                v, err = parse_score_value(
                    m.group(1), f'記錄表 {titles[gi]} / {qid}')
                if err:
                    bad_scores.append(err)
                else:
                    out[gi][qid] = v
    return out, unknown_titles, flags, bad_scores


# 門檻的成立條件。🔴 **這三個值與下方的分數門檻是綁在一起的**，
# 原門檻是在「8 個計分題、每題滿分 2、配對 t 檢定 df=7」之下算出來的。
# ⚠️ 題數一變，df 與滿分都變，而原門檻不會跟著變——**它會安靜地指向錯的數字**。
#    這正是零章第十一條原則要防的形狀（不由本處維護的數字過期時無聲），
#    所以這裡不重算統計（要重算得有標準差），而是**條件不符就宣告門檻失效**。
THRESHOLD_BASIS = {'n_items': 8, 'per_item': 2.0}
THRESHOLD_TABLE = {          # 評分者人數 → (90% 信心, 95% 信心)
    1: (5.0, 6.0),
    2: (4.0, 5.0),
}
THRESHOLD_CHERRYPICK = 7.6   # 跑完才挑差距最大的兩組時的門檻


def print_threshold(df, full, n_raters):
    """印出可用的門檻，或在成立條件不符時明說它已失效。"""
    basis_n = THRESHOLD_BASIS['n_items']
    basis_full = basis_n * THRESHOLD_BASIS['per_item']
    ok = (df + 1 == basis_n) and abs(full - basis_full) < 1e-9

    if not ok:
        print(f'🔴 門檻已失效，不要沿用舊數字。')
        print(f'   既有門檻是在「{basis_n} 個計分題、滿分 {basis_full:g}、df='
              f'{basis_n - 1}」下算出來的；')
        print(f'   本輪是「{df + 1} 個計分題、滿分 {fmt_score(full)}、df={df}」。')
        print('   ⚠️ 題數或尺度變動後必須重算配對 t 的門檻，本工具不自行外推——')
        print('      外推出來的門檻看起來與實算的一模一樣。')
        return

    lo, hi = THRESHOLD_TABLE.get(
        n_raters, THRESHOLD_TABLE[2 if n_raters >= 2 else 1])
    who = '單人' if n_raters == 1 else f'{n_raters} 位平均'
    print(f'⚠️ 門檻（配對 t、df={df}、滿分 {fmt_score(full)}）：'
          f'{who} ≥{lo:g} 分為 90% 信心、≥{hi:g} 分為 95%。')
    if n_raters > 2:
        print(f'   ⚠️ 三位以上沿用兩位的門檻是**保守估計**，不是實算值——'
              f'評分者變多時真實門檻會更低，這裡刻意不調鬆。')
    print('   未達門檻一律判「無法區分」，不是「略優」。')
    print(f'   ⛔ 比較對必須開跑前就指定完畢——跑完才挑差距大的兩組，'
          f'門檻升至 {THRESHOLD_CHERRYPICK:g}。')
    print('   本工具只算數字，不下結論。')


def cmd_merge(args):
    if not os.path.isfile(args.key):
        print(f'🔴 找不到對照表：{args.key}', file=sys.stderr)
        return 2
    with open(args.key, encoding='utf-8') as f:
        key = json.load(f)

    titles = [g['title'] for g in key['groups']]
    models = [g['meta'].get('模型', '(未填)') for g in key['groups']]
    qids = sorted({c.rsplit('-', 1)[0] for c in key['map']})
    # 計分範圍優先讀對照表（make 當時以判準檔決定的）；
    # 舊對照表沒有這個欄位，退回字母白名單。
    if key.get('scored'):
        scored_set = set(key['scored'])
        scored_qids = [q for q in qids if q in scored_set]
    else:
        scored_qids = [q for q in qids if q[0] in SCORED_PREFIXES]

    # {rater: {gi: {qid: score}}}
    table, unknown, missing = OrderedDict(), [], []
    bad_scores = []   # 尺度不合法的分數，最後一次全部列出再失敗

    # 來源一：記錄表（未盲，邪狼那欄）。組別標題直接對應，不經代號。
    if args.record:
        if not os.path.isfile(args.record):
            print(f'🔴 找不到記錄表：{args.record}', file=sys.stderr)
            return 2
        rec, bad_titles, flags, bad = parse_record_scores(args.record, titles)
        bad_scores += bad
        if bad_titles:
            print('🔴 記錄表裡的組別標題不在對照表中，兩者不是同一輪：', file=sys.stderr)
            for t in bad_titles:
                print(f'   組別：{t}', file=sys.stderr)
            return 1
        name = os.path.splitext(os.path.basename(args.record))[0] + '（記錄表）'
        table[name] = rec

    else:
        flags = None

    # 來源二：盲評版填答區或純分數檔（每位盲評者一份）
    for p in args.scores or []:
        if not os.path.isfile(p):
            print(f'🔴 找不到分數檔：{p}', file=sys.stderr)
            return 2
        r = os.path.splitext(os.path.basename(p))[0]
        scores, bad = parse_scores(p)
        bad_scores += bad
        table[r] = defaultdict(dict)
        for code, val in scores.items():
            gi = key['map'].get(code)
            if gi is None:
                unknown.append(f'{r}: {code}')
                continue
            table[r][gi][code.rsplit('-', 1)[0]] = val

    if not table:
        print('🔴 沒有給任何分數來源（--record / --scores 至少要有一個）。',
              file=sys.stderr)
        return 1

    raters = table  # 下游沿用同一個名字
    for r in table:
        for q in scored_qids:
            for gi in range(len(titles)):
                if table[r][gi].get(q) is None:
                    missing.append(f'{r}: {titles[gi]} / {q}')

    if bad_scores:
        print(f'🔴 下列分數不合尺度（0～{SCORE_MAX:g}，{SCORE_STEP:g} 的倍數）：',
              file=sys.stderr)
        for x in bad_scores:
            print(f'   {x}', file=sys.stderr)
        print('   ⚠️ 刻意不自動四捨五入——猜出來的值與有依據的值在表上長得一樣。',
              file=sys.stderr)
        return 1

    if unknown:
        print('🔴 下列代號不在對照表裡（打錯或對到別輪的對照表）：', file=sys.stderr)
        for x in unknown:
            print(f'   {x}', file=sys.stderr)
        return 1
    if missing:
        print('⚠️ 下列題目沒有分數，總分不完整：')
        for x in missing:
            print(f'   {x}')
        print()

    full = SCORE_MAX * len(scored_qids)
    df = len(scored_qids) - 1
    print(f'來源：{key["source"]}　seed={key["seed"]}　'
          f'計分題 {len(scored_qids)} 題　滿分 {fmt_score(full)}')
    print()

    header = f'{"題號":<6}' + ''.join(f'{t:<14}' for t in titles)
    for r in raters:
        print(f'── 評分者：{r} ──')
        print(header)
        for q in scored_qids:
            row = f'{q:<6}'
            for gi in range(len(titles)):
                v = table[r][gi].get(q)
                row += f'{fmt_score(v):<14}'
            print(row)
        row = f'{"總分":<6}'
        for gi in range(len(titles)):
            vals = [table[r][gi].get(q) for q in scored_qids]
            got = sum(v for v in vals if v is not None)
            row += f'{f"{fmt_score(got)}/{fmt_score(full)}":<14}'
        print(row)
        print()

    if len(raters) >= 2:
        print(f'── {len(raters)} 位評分者的平均與差距 ──')
        avg = {}
        for gi, t in enumerate(titles):
            tot = []
            for r in raters:
                vals = [table[r][gi].get(q) for q in scored_qids]
                tot.append(sum(v for v in vals if v is not None))
            avg[gi] = sum(tot) / len(tot)
            print(f'{t:<14}{avg[gi]:.2f}/{fmt_score(full)}　（{models[gi]}）')
        print()
        for a in range(len(titles)):
            for b in range(a + 1, len(titles)):
                d = abs(avg[a] - avg[b])
                print(f'{titles[a]} vs {titles[b]}：差距 {d:.2f} 分')
        print()
        print_threshold(df, full, len(raters))

        # 逐題一致率：兩人時才有意義，多於兩人不自動算（配對怎麼取要人決定）
        if len(raters) == 2:
            r1, r2 = list(raters)
            same = tot_n = 0
            for gi in range(len(titles)):
                for q in scored_qids:
                    v1, v2 = table[r1][gi].get(q), table[r2][gi].get(q)
                    if v1 is not None and v2 is not None:
                        tot_n += 1
                        same += (v1 == v2)
            if tot_n:
                print()
                print(f'逐題一致率：{same}/{tot_n} = {same / tot_n:.0%}')
                print('⚠️ 一致率下降不代表變糟——盲評解的是偏差，'
                      '它會讓原本靠共同預期撐起來的「一致」消失。')
    else:
        print_threshold(df, full, 1)

    if flags:
        names = {'R': '角色反轉', 'X': '語法破碎', 'N': '稱謂疊用', 'P': '無情境第三人稱'}
        if any(flags[gi] for gi in range(len(titles))):
            print()
            print('── 語言瑕疵標記（不計分，來自記錄表備註）──')
            print(f'{"組別":<14}' + ''.join(f'{v:<12}' for v in names.values()))
            for gi, t in enumerate(titles):
                print(f'{t:<14}' + ''.join(f'{flags[gi].get(k, 0):<12}' for k in names))
            print('⚠️ 四種是不同的病，不要相加。`[P]` 需判斷情境，是唯一不能純字面判定的。')

    print()
    print('── 對照表 ──')
    for gi, t in enumerate(titles):
        desc = '　'.join(f'{k}={v}' for k, v in key['groups'][gi]['meta'].items())
        print(f'{t}：{desc}')
    return 0


# ══════════════════════════ CLI ══════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='AOFOS 施測記錄表盲評工具（與專案程式碼零耦合）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    mk = sub.add_parser('make', help='記錄表 → 盲評版 ＋ 對照表')
    mk.add_argument('record', help='填好的施測記錄表 .md')
    mk.add_argument('--seed', type=int, default=None,
                    help='固定打亂順序。省略則自動產生一個並寫進對照表')
    mk.add_argument('--outdir', default=None, help='輸出目錄，預設與記錄表同目錄')
    mk.add_argument('--include', default='',
                    help='額外納入盲評的類別，逗號分隔（例：F,A）。'
                         '預設只出 B/C/D/E')
    mk.add_argument('--raters', default=None,
                    help='評審名單，逗號分隔（例：GPT,Claude,Gemini）。'
                         '每位產一份獨立檔案；內容與題序相同，'
                         '分開的是填答與收回。省略則產單一份（原行為）')
    mk.add_argument('--questions', default=None,
                    help='題本 .md 路徑。給了就把每題的題目注入評分表——'
                         '外部評審模型沒有題本，不注入等於讓它憑印象打分')
    mk.add_argument('--rubric', default=None,
                    help='判準 .md 路徑。給了就把每題的判分界線注入評分表')
    mk.set_defaults(func=cmd_make)

    mg = sub.add_parser('merge', help='對照表 ＋ 分數檔 → 開盲、合併、算差距')
    mg.add_argument('--key', required=True, help='make 產出的 .對照表.json')
    mg.add_argument('--record', default=None,
                    help='填好的施測記錄表 .md，直接讀其中的 `**評分**：n/2`'
                         '（未盲那一欄，免手抄）')
    mg.add_argument('--scores', nargs='*', default=[],
                    help='填好的盲評版或純分數檔，可給多個（每位盲評者一份）')
    mg.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BrokenPipeError:
        os._exit(0)
    except KeyboardInterrupt:
        print('\n已中止，未產生任何檔案。', file=sys.stderr)
        sys.exit(130)
