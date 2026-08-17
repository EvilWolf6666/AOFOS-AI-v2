#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blind_eval.py — 把填好的 AOFOS 施測記錄表轉成盲評版

═══════════════════════════════════════════════════════════════════
本檔與 AOFOS 程式碼庫**完全不耦合**，這是刻意的：

  · 不 import 專案任何模組，特別是 **不 import config.py**
  · 只用 Python 標準庫，無第三方依賴
  · 獨立指令觸發，不掛進 pipeline._commands，不進 Agent 生命週期

理由同三之35 對 `chat_strata.py` 的判定：`tools/` 底下的腳本共用
`config.py` 唯一入口，而本工具不需要任何設定值，硬接會讓一個純文字
處理工具綁上 `HF_HUB_OFFLINE` 那條啟動路徑。施測時 LM Studio 可能正
載著別的模型、config.py 可能改到一半，本工具不該因此跑不起來。

⚠️ 日後若有人想「順手」讓它讀 config 取路徑——那就是耦合的入口。
   路徑一律由命令列給。
═══════════════════════════════════════════════════════════════════

為什麼要有這支工具
  去識別化如果靠手動，遲早會漏一次，而漏的那次**不會有任何警告**
  ——輸出看起來完全正常，只是 Claude 那欄的分數帶著期望偏差。
  （形狀同「靠自覺運作的規則會失效」，見 aofos-docs/cases.md。）

用法
  python blind_eval.py <記錄表.md>
  python blind_eval.py <記錄表.md> --seed 42        # 要重現同一組打亂順序
  python blind_eval.py <記錄表.md> --outdir <目錄>   # 預設與記錄表同目錄

產出（預設寫在記錄表所在目錄）
  <記錄表名>.盲評版.md    → 給 Claude
  <記錄表名>.對照表.txt   → 自己留著，Claude 交分數前不公布

離開碼
  0 = 正常
  1 = 有殘留字串需人工確認 / 記錄表格式不對
  2 = 找不到檔案
"""
import argparse
import os
import random
import re
import sys

# Windows 終端預設 cp950，直接 print emoji 會 UnicodeEncodeError。
# 這支工具在 VSCode 內建終端與 cmd 都要能跑，所以先把 stdout 轉成 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

GROUP_RE = re.compile(r'^##\s*組別[：:]\s*(.*)$')

# 會洩漏組別身分的行：評分（錨定）、prompt_tokens（Llama 612~628、Qwen 必然不同）
DROP_LINE_RE = re.compile(r'^\s*(\*\*評分\*\*|`prompt_tokens=|判讀[：:])')

# metadata 區塊裡要抽出來做對照表的欄位
META_KEYS = ('模型', '量化', 'LoRA', 'GPU 卸載層數', '施測日期')


def parse_groups(lines):
    """切出 [(原標題, 起始行, 結束行)]，只取 '## 組別：' 區塊。"""
    marks = [i for i, l in enumerate(lines) if GROUP_RE.match(l)]
    groups = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        # '## 測後' / '## 評分程序' 這類尾端章節要排除在組別之外
        for j in range(start + 1, end):
            if lines[j].startswith('## ') and not GROUP_RE.match(lines[j]):
                end = j
                break
        groups.append((GROUP_RE.match(lines[start]).group(1).strip(), start, end))
    return groups


def extract_meta(block):
    meta = {}
    for l in block:
        for k in META_KEYS:
            m = re.match(rf'^\s*{re.escape(k)}\s*[：:]\s*(.*)$', l)
            if m and m.group(1).strip():
                meta[k] = m.group(1).strip()
    return meta


def scrub(block):
    """移除 metadata / 評分 / prompt_tokens。<<< >>> 之間一字不動。"""
    out, in_meta, in_raw = [], False, False
    for l in block:
        s = l.strip()

        if s == '<<<':
            in_raw = True
            out.append(l)
            continue
        if s == '>>>':
            in_raw = False
            out.append(l)
            continue
        if in_raw:
            out.append(l)          # 原始輸出是評分依據，一個字都不改
            continue

        if s == '```':             # metadata 圍籬
            in_meta = not in_meta
            continue
        if in_meta:
            continue
        if DROP_LINE_RE.match(l):
            continue

        out.append(l)

    cleaned, blank = [], 0
    for l in out:
        if l.strip():
            blank = 0
            cleaned.append(l)
        else:
            blank += 1
            if blank <= 1:
                cleaned.append(l)
    return cleaned


def collect_secrets(metas):
    """組別身分字串，用於事後殘留檢查（不用於自動取代）。"""
    secrets = set()
    for m in metas:
        for k in ('模型', 'LoRA'):
            v = m.get(k, '')
            if v:
                secrets.add(v)
                secrets.update(re.findall(r'[A-Za-z0-9_.\-]{4,}', v))
    return {s for s in secrets if s}


def main():
    ap = argparse.ArgumentParser(
        description='把填好的 AOFOS 施測記錄表轉成盲評版（與專案程式碼零耦合）')
    ap.add_argument('record', help='填好的施測記錄表 .md')
    ap.add_argument('--seed', type=int, default=None,
                    help='固定打亂順序。省略則每次都不同')
    ap.add_argument('--outdir', default=None,
                    help='輸出目錄，預設與記錄表同目錄')
    args = ap.parse_args()

    if not os.path.isfile(args.record):
        print(f'🔴 找不到檔案：{args.record}', file=sys.stderr)
        return 2

    with open(args.record, encoding='utf-8') as f:
        lines = f.read().split('\n')

    groups = parse_groups(lines)
    if not groups:
        print('🔴 找不到任何 "## 組別：" 區塊，請確認記錄表格式。', file=sys.stderr)
        return 1

    metas = [extract_meta(lines[s:e]) for _, s, e in groups]

    # 前置檢查：metadata 沒填就大聲失敗。
    # 讓它靜默通過的話，對照表會是空的，而盲評版看起來完全正常。
    missing = [t for (t, _, _), m in zip(groups, metas) if not m.get('模型')]
    if missing:
        print('🔴 下列組別沒有填「模型：」，對照表會失去意義：', file=sys.stderr)
        for t in missing:
            print(f'   組別：{t}', file=sys.stderr)
        print('   填完再跑一次。', file=sys.stderr)
        return 1

    order = list(range(len(groups)))
    random.Random(args.seed).shuffle(order)

    head = [
        '# AOFOS 施測結果（盲評版）',
        '',
        '> 組別已重新指派並打亂；模型名／量化／LoRA 版本／`prompt_tokens`／既有評分皆已移除。',
        '> `<<<` 與 `>>>` 之間為原始輸出，**未經任何修改**。',
        '> 評分依《AOFOS-固定評測題組_定稿後不得修改》第三章的通過條件，B/C/D/E 每題 0～2 分。',
        '',
        '---',
        '',
    ]
    body_all, mapping = [], []
    for new_idx, gi in enumerate(order, 1):
        title, s, e = groups[gi]
        code = f'G{new_idx}'
        body = scrub(lines[s + 1:e])
        while body and not body[0].strip():
            body.pop(0)
        while body and (not body[-1].strip() or body[-1].strip() == '---'):
            body.pop()
        body_all.append(f'## 組別：{code}')
        body_all.extend(body)
        body_all.extend(['', '---', ''])

        desc = '　'.join(f'{k}={metas[gi][k]}' for k in META_KEYS if metas[gi].get(k))
        mapping.append(f'{code}  ←  原標題「{title}」　{desc}')

    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.record)))
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.record))[0]
    blind_path = os.path.join(outdir, f'{stem}.盲評版.md')
    map_path = os.path.join(outdir, f'{stem}.對照表.txt')

    blind_text = '\n'.join(head + body_all)
    with open(blind_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(blind_text)
    with open(map_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('AOFOS 施測盲評對照表\n')
        f.write(f'來源：{os.path.basename(args.record)}\n')
        f.write(f'seed：{args.seed if args.seed is not None else "(隨機，未固定)"}\n')
        f.write('⚠️ Claude 交出分數之前不要公布本檔。\n\n')
        f.write('\n'.join(mapping) + '\n')

    print(f'✅ {blind_path}')
    print(f'   → 給 Claude')
    print(f'✅ {map_path}')
    print(f'   → 自己留著')
    print()
    print(f'共 {len(groups)} 組，已打亂。')

    # 殘留檢查：只定位、不自動改。
    # 模型自稱（「我是 Qwen」）既是識別線索、也是 A/D 類的判讀材料，
    # 自動遮蔽會刪掉評分依據——工具負責枚舉與定位，判斷交給人。
    leaked = sorted(x for x in collect_secrets(metas) if x in blind_text)
    if leaked:
        print()
        print('⚠️ 盲評版的原始輸出裡出現下列字串，**工具不自動改**：')
        for x in leaked:
            print(f'   {x}')
        print('   純識別線索 → 手動刪；同時是評分依據（模型自稱等）→ 留著，')
        print('   並知道該組對 Claude 而言等於未盲。')
        return 1

    print('✅ 殘留檢查：未出現 metadata 中的模型名字串。')
    print('⚠️ 但風格特徵去不掉——emoji 密度、中英夾雜、回應長度都會洩漏身分。')
    print('   風格差異大的組別（例如 Qwen vs 套了 LoRA 的），去識別化實際上無效。')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BrokenPipeError:
        os._exit(0)
    except KeyboardInterrupt:
        print('\n已中止，未產生任何檔案。', file=sys.stderr)
        sys.exit(130)
