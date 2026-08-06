# core/text_utils.py

import opencc

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

    量測方式：開著 DIAGNOSTICS["log_opencc_changes"] 跑一輪固定評測題組，
    看 opencc 是否完全不動作。完全不動作即代表這一層可以整個拿掉，
    少一層後處理就少一個變數。
    """
    converted = _converter.convert(text)
    log_opencc_diff(text, converted)   # 診斷腳手架
    return converted