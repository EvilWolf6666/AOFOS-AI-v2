# core/text_utils.py

import opencc

from config import TEXT_CONFIG
from core.short_time_test import log_opencc_diff   # 診斷腳手架，清理時連同下方呼叫一併移除

# ⚠️ s2tw 而非 s2t：兩者都只轉字形、不動詞彙，差別在採用哪套繁體字形標準——
#    s2t 走大陸標準（才→纔、吃→喫、床→牀、为→爲），s2tw 走台灣教育部標準。
#
#    ⚠️ 換過來的理由是**對齊目標語系**，不是修 bug。原本掛著的假設是「輸出裡的
#    異體字來自 opencc s2t」，該假設已於 2026-08-06 證偽——log_opencc_diff()
#    掛載期間每一次抓到的都是簡體字（劳→勞之類），異體字一次都沒出現過。
#    所以換與不換都不影響那個問題，別誤以為它解決了什麼。
#
# ⚠️ 絕對不要用 s2twp。那個模式會連詞彙一起換（软件→軟體、内存→記憶體），
#    等於在權重之外偷偷覆寫奧狐的用詞，而且改動不會出現在任何 log 裡——
#    所有 LoRA 版本比較都會被污染，你以為在比權重，實際上比的是權重加一層
#    看不見的詞彙替換。
_converter = opencc.OpenCC("s2tw")


def fix_traditional(text: str) -> str:
    """把輸出裡的簡體字轉成繁體。

    ⚠️ 這一層是在補模型的缺陷，不是本質需求——它存在的唯一原因是現行底模
    （Llama-3.1 系）會漏出簡體字。**換模型後必須重測是否還需要它**：換成台灣
    在地化模型（TAIDE、Breeze2 等）後前提可能不成立，而後處理層會**遮掩模型
    本身的輸出品質**——若模型其實已經不吐簡體字，留著只是多一道無效轉換，
    還會讓人誤以為「有在把關」。

    🔴 **量測方式已於 2026-08-16 改為「直接關掉這一層看輸出」**，
    由 TEXT_CONFIG["opencc_enabled"] 控制（現為 False）。

    原本的量測方式是「開著 DIAGNOSTICS['log_opencc_changes'] 看它有沒有動作」，
    但那條判讀規則（有 diff = opencc 有價值）**本身沒有被驗證過**：s2* 系列對
    「已經是繁體」的輸入不是 no-op，diff 有可能正是它把本來就對的字改壞。
    直接看關掉後的輸出裡有沒有簡體字，才是在量真正要量的東西。見《歸檔》一之14。

    ⚠️ 關閉走早退，不是移除——opencc 的 import 與 _converter 建立一律保留在
    模組層級。恢復只需把 config 那個布林改回 True，不必回頭動三個檔案。
    ⚠️ 早退時 log_opencc_diff() 一併不執行（沒有轉換就沒有 diff 可記），
    這是預期行為，不是腳手架失效。
    """
    if not TEXT_CONFIG["opencc_enabled"]:
        return text

    converted = _converter.convert(text)
    log_opencc_diff(text, converted)   # 診斷腳手架
    return converted