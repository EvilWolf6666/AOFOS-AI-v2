#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blind_eval.py — AOFOS 施測記錄表的盲評工具（make / merge）

═══════════════════════════════════════════════════════════════════
本檔與 AOFOS 程式碼庫**完全不耦合**，這是刻意的：

  · 不 import 專案任何模組，特別是 **不 import config.py**
  · 只用 Python 標準庫，無第三方依賴
  · 獨立指令觸發，不掛進 pipeline._commands，不進 Agent 生命週期

理由同 `tools/chat_strata.py` 的判定：`tools/` 底下的腳本共用 config.py
唯一入口，而本工具不需要任何設定值，硬接會讓一個純文字處理工具綁上
`HF_HUB_OFFLINE` 那條啟動路徑。施測時 LM Studio 可能正載著別的模型、
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
SCORE_LINE_RE = re.compile(r'^\s*([A-Z]\d+-\d+)\s*[：:]\s*([0-2])\s*$')

RAW_OPEN, RAW_CLOSE = '<<<', '>>>'

# metadata 區塊裡要抽出來做對照表的欄位
META_KEYS = ('模型', '量化', 'LoRA', 'GPU 卸載層數', 'GPU 卸載層數 / LM Studio 估計 VRAM',
             '施測日期')

# 計分類別。F（指紋）與 A（語氣）不計分，預設也不進盲評版。
SCORED_PREFIXES = ('B', 'C', 'D', 'E')


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
            for k in META_KEYS:
                mm = re.match(rf'^\s*{re.escape(k)}\s*[：:]\s*(.*)$', line)
                if mm and mm.group(1).strip():
                    groups[gi]['meta'][k] = mm.group(1).strip()

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

    include = {c.strip().upper() for c in args.include.split(',') if c.strip()}
    qids = [q for q in samples if q[0] in SCORED_PREFIXES or q[0] in include]
    if not qids:
        print('🔴 沒有任何題目符合輸出條件（--include 目前是 '
              f'"{args.include}"）。', file=sys.stderr)
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
        '> 評分依《AOFOS-固定評測題組_定稿後不得修改》第三章的通過條件，'
        'B/C/D/E 每題 0～2 分。',
        '',
        '---',
        '',
    ]

    body, mapping = [], {}
    for q in qids:
        order = list(range(len(groups)))
        rng.shuffle(order)
        body.append(f'## {q}')
        body.append('')
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
                 '> 一行一個，格式 `代號: 分數`（0～2）。不確定的留空不要猜。', ''])
    for q in qids:
        if q[0] in SCORED_PREFIXES:
            for n in range(1, len(groups) + 1):
                body.append(f'{q}-{n}: ')
    body.append('')

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.record))
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.record))[0]
    blind_path = os.path.join(outdir, f'{stem}.盲評版.md')
    key_path = os.path.join(outdir, f'{stem}.對照表.json')

    blind_text = '\n'.join(head + body)
    with open(blind_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(blind_text)

    key = {
        'source': os.path.basename(args.record),
        'seed': seed,
        'groups': [{'title': g['title'], 'meta': g['meta']} for g in groups],
        'map': mapping,
    }
    with open(key_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    print(f'✅ {blind_path}')
    print('   → 給 Claude')
    print(f'✅ {key_path}')
    print('   → 自己留著，Claude 交分數前不要打開給他看')
    print()
    print(f'共 {len(groups)} 組 × {len(qids)} 題，逐題各自打亂。seed={seed}')
    skipped = [q for q in samples if q not in qids]
    if skipped:
        print(f'（未輸出：{"、".join(skipped)}——F/A 類預設不盲評，'
              f'要放進來用 --include F,A）')

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
        print('   並知道該題對 Claude 而言等於未盲。')
        return 1

    print('✅ 殘留檢查：未出現 metadata 中的模型名字串。')
    print('⚠️ 但風格特徵去不掉——emoji 密度、中英夾雜、回應長度都會洩漏身分。')
    print('   風格差異大的組別，去識別化實際上無效。')
    return 0


# ══════════════════════════ merge ══════════════════════════

def parse_scores(path):
    """讀填好的盲評版或純分數檔，回傳 {代號: 分數}。空值略過。"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = SCORE_LINE_RE.match(line.rstrip('\n'))
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def cmd_merge(args):
    if not os.path.isfile(args.key):
        print(f'🔴 找不到對照表：{args.key}', file=sys.stderr)
        return 2
    with open(args.key, encoding='utf-8') as f:
        key = json.load(f)

    titles = [g['title'] for g in key['groups']]
    models = [g['meta'].get('模型', '(未填)') for g in key['groups']]
    qids = sorted({c.rsplit('-', 1)[0] for c in key['map']})
    scored_qids = [q for q in qids if q[0] in SCORED_PREFIXES]

    raters = OrderedDict()
    for p in args.scores:
        if not os.path.isfile(p):
            print(f'🔴 找不到分數檔：{p}', file=sys.stderr)
            return 2
        raters[os.path.splitext(os.path.basename(p))[0]] = parse_scores(p)

    if not raters:
        print('🔴 沒有給任何分數檔。', file=sys.stderr)
        return 1

    # {rater: {gi: {qid: score}}}
    table = {r: defaultdict(dict) for r in raters}
    unknown, missing = [], []
    for r, scores in raters.items():
        for code, val in scores.items():
            gi = key['map'].get(code)
            if gi is None:
                unknown.append(f'{r}: {code}')
                continue
            table[r][gi][code.rsplit('-', 1)[0]] = val
        for q in scored_qids:
            for n in range(1, len(titles) + 1):
                if f'{q}-{n}' not in scores:
                    missing.append(f'{r}: {q}-{n}')

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

    full = 2 * len(scored_qids)
    print(f'來源：{key["source"]}　seed={key["seed"]}　滿分 {full}')
    print()

    header = f'{"題號":<6}' + ''.join(f'{t:<14}' for t in titles)
    for r in raters:
        print(f'── 評分者：{r} ──')
        print(header)
        for q in scored_qids:
            row = f'{q:<6}'
            for gi in range(len(titles)):
                v = table[r][gi].get(q)
                row += f'{("_" if v is None else v):<14}'
            print(row)
        row = f'{"總分":<6}'
        for gi in range(len(titles)):
            vals = [table[r][gi].get(q) for q in scored_qids]
            got = sum(v for v in vals if v is not None)
            row += f'{f"{got}/{full}":<14}'
        print(row)
        print()

    if len(raters) >= 2:
        print('── 雙人平均與差距 ──')
        avg = {}
        for gi, t in enumerate(titles):
            tot = []
            for r in raters:
                vals = [table[r][gi].get(q) for q in scored_qids]
                tot.append(sum(v for v in vals if v is not None))
            avg[gi] = sum(tot) / len(tot)
            print(f'{t:<14}{avg[gi]:.1f}/{full}　（{models[gi]}）')
        print()
        for a in range(len(titles)):
            for b in range(a + 1, len(titles)):
                d = abs(avg[a] - avg[b])
                print(f'{titles[a]} vs {titles[b]}：差距 {d:.1f} 分')
        print()
        print('⚠️ 門檻（配對 t、df=7、滿分 16）：雙人平均 ≥4 分為 90% 信心、'
              '≥5 分為 95%；單人 5／6。')
        print('   未達門檻一律判「無法區分」，不是「略優」。')
        print('   ⛔ 比較對必須開跑前就指定完畢——跑完才挑差距大的兩組，門檻升至 7.6。')
        print('   本工具只算數字，不下結論。')

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
        print('⚠️ 只有一位評分者，門檻升一級：≥5 分為 90% 信心、≥6 分為 95%。')

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
    mk.set_defaults(func=cmd_make)

    mg = sub.add_parser('merge', help='對照表 ＋ 分數檔 → 開盲、合併、算差距')
    mg.add_argument('--key', required=True, help='make 產出的 .對照表.json')
    mg.add_argument('--scores', nargs='+', required=True,
                    help='填好的盲評版或純分數檔，可給多個（每人一份）')
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
