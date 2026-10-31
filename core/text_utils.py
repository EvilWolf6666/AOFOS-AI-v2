# core/text_utils.py

import opencc

from config import TEXT_CONFIG
from core.short_time_test import log_opencc_diff   # 診斷腳手架，清理時連同下方呼叫一併移除

# ⚠️ s2tw 而非 s2t：兩者都只轉字形、不動詞彙，差別在採用哪套繁體字形標準——
#    s2t 走大陸標準（才→纔、吃→喫、床→牀、为→爲），s2tw 走台灣教育部標準。
#    用 s2tw 是為了對齊目標語系，不是修 bug。
#
# ⚠️ 絕對不要用 s2twp。那個模式會連詞彙一起換（软件→軟體、内存→記憶體），
#    等於在權重之外偷偷覆寫奧狐的用詞，而且改動不會出現在任何 log 裡——
#    所有 LoRA 版本比較都會被污染，你以為在比權重，實際上比的是權重加一層
#    看不見的詞彙替換。
_converter = opencc.OpenCC("s2tw")


def fix_traditional(text: str) -> str:
    """把輸出裡的簡體字轉成繁體。

    ⚠️ 這一層補的是模型缺陷（現行底模會漏出簡體字），換模型後要重測是否還需要。
    🔴 目前不可拿掉：opencc 關閉下輸出仍有簡體字（《現況》第五章 #8、第八章 C13）；
       每輪施測都要記 opencc_enabled 的值，兩組對照必須同值。
    ⚠️ 要不要恢復看「關掉後的輸出有沒有簡體字」，不看 log_opencc_diff() 的計數：
       s2* 對已是繁體的輸入不是 no-op，diff 可能正是它把對的字改壞（《歸檔》I0042）。
    ⚠️ 關閉走早退，不是移除——opencc 的 import 與 _converter 建立保留在模組層級，
       恢復只改 config 的布林。早退時 log_opencc_diff() 一併不執行，這是預期行為。
    """
    if not TEXT_CONFIG["opencc_enabled"]:
        return text

    converted = _converter.convert(text)
    log_opencc_diff(text, converted)   # 診斷腳手架
    return converted
