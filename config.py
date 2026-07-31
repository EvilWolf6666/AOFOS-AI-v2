# config.py
# 統一設定檔 — 之後換模型/API 只改這裡

# ─────────────────────────────────────────────────────────────
# LLM 連線與推理參數
# ─────────────────────────────────────────────────────────────
# 所有影響輸出的參數一律在此明確指定並隨 request 送出。
#
# 原因：LM Studio 的預設參數綁在「各個模型」而非全域共用。未明確帶入的參數，
# 在切換模型時會各自吃到不同的預設值，使跨模型比較靜默失效且不會有任何警告。
#
# ⚠️ 下方數值請對照 LM Studio 目前載入模型的參數面板逐一抄寫。
#    抄一致 = 行為不變但變數鎖住；抄錯 = 行為改變且不易察覺。

LLM_CONFIG = {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",

    # 模型識別碼。填明確 id 之後，換模型只要改這一行，LM Studio 會自動載入對應模型，
    # 不必去 GUI 點選；紀錄與實際也才會一致。
    #
    # 目前本機可用（2026-07-31 由 LLMClient 啟動時列出）：
    #   "aofos_v3"                               ← 實驗組（LoRA v3 合併版）
    #   "aofos_v2"                               ← LoRA v2
    #   "aofos_v1"                               ← LoRA v1
    #   "meta-llama-3.1-8b-instruct-abliterated" ← 對照組 A（無 LoRA 底模）
    #   "meta-llama-3.1-8b-instruct"             ← 對照組 B（官方版）
    #
    # ⚠️ 切換模型後仍須到 GUI 確認載入層參數（上下文長度 8192／GPU 卸載 17），
    #    LM Studio 會依模型檔案大小自動調整，且不會警告。
    "model": "aofos_v3",

    # ── 標準 OpenAI 參數（直接以具名參數送出）──
    "temperature": 0.8,
    "max_tokens": 300,

    # 不同用途的輸出長度上限，未列出的用途沿用上方 max_tokens。
    #
    # 【2026-07-30 新增】_compress() 要輸出「摘要最多 8 句 + 淘汰清單」兩段內容。
    # 中文在 Llama tokenizer 下約 1 token/字，8 句摘要本身就逼近 300 token，
    # 「淘汰：」那一段極可能被截斷在輸出之外。
    #
    # 這個失敗模式沒有任何徵兆：_parse_compress_result() 找不到「淘汰：」時
    # 不會報錯，會安靜地回傳 discarded="無"，於是淘汰內容永遠不會寫進 RAG。
    # 若 rag_stats.json 的 writes 長期為 0 或遠低於壓縮次數，優先懷疑這裡。
    "max_tokens_by_purpose": {
        "compress": 800,
    },
    "top_p": 0.95,
    "frequency_penalty": 0.0,   # 0.0 = 不改變現有行為
    "presence_penalty": 0.0,    # 0.0 = 不改變現有行為

    # 固定 seed 可消除「同一顆模型自己的隨機波動」，
    # 對照組比較期間務必固定。設 None 表示不固定。
    "seed": 42,

    # ── LM Studio 專屬參數（透過 extra_body 送出）──
    # 若某個參數不想指定，把值設為 None，該欄位就不會被送出。
    "top_k": 40,
    "repeat_penalty": 1.1,
    "min_p": 0.05,
}


# ─────────────────────────────────────────────────────────────
# LM Studio 載入層參數（紀錄用，非設定用）
# ─────────────────────────────────────────────────────────────
# ⚠️ 這些值「無法」透過 OpenAI 相容 API 設定，只能在 LM Studio GUI 手動設。
#    這裡寫下來的目的有二：
#      1. 讓環境設定跟程式碼一起進版控，換模型時有東西可對照
#      2. 啟動時寫進 log，日後回看任何一次跑測的紀錄都知道當時環境是什麼
#
# ⚠️ 切換模型時 LM Studio 會依模型檔案大小「自動」建議 GPU 卸載層數，
#    若放著不管，對照組兩顆會跑在不同卸載層數上——這會同時影響速度與
#    數值精度，且不會有任何警告。載入新模型後務必手動改回下列數值。

LMSTUDIO_LOAD_PARAMS = {
    # ── Context and Offload ──
    "context_length": 8192,              # 模型標稱支援 131072，此處刻意限制（VRAM）
    "gpu_offload_layers": 17,
    # 參考：Aofos-v3 在上述設定下，LM Studio 估計 GPU 2.99 GB / 合計 5.32 GB。
    # 顯存已接近上限，因此 8192 與 17 層都是硬體約束下的結果，不是任意選的。
    # 換模型後若這兩個數字被自動改掉，記憶體估計值也會跟著變，可用來反查是否被改。

    # ── Advanced（載入層）──
    "cpu_thread_pool": 3,
    "evaluation_batch_size": 2048,
    "physical_batch_size": 512,
    # 【2026-07-31 由 4 改為 1】兩個理由：
    #   1. 連續批次處理不保證逐位一致，是輸出無法重現的來源之一
    #   2. llama.cpp 的平行槽位會「切分」總 context——若 LM Studio 沿用此行為，
    #      設 8192 而 parallel=4 時，單次實際可用僅 2048。目前 prompt 約 700 tokens
    #      撞不到，但世界觀寫入 RAG、對話變長後會遠比預期更早開始截斷，
    #      且截斷模式是「截斷中間」，消失的正是摘要與早期對話。
    # 單人使用無平行需求，設 1 沒有任何損失。
    "max_concurrent_predictions": 1,
    "unified_kv_cache": True,
    "context_checkpoints": 32,           # prompt 快取；會影響重複前綴的延遲量測
    "rope_freq_base": "自動",
    "rope_freq_scale": "自動",
    "kv_cache_offload_to_gpu": True,
    "keep_model_in_memory": True,
    "try_mmap": True,
    "gui_seed": 42,                      # 已於 GUI 固定，但實測無效（見 TEST_CONFIG 說明）
    "speculative_decoding": "off",
    "flash_attention": True,
    "kv_cache_quantization": None,       # K/V 快取量化類型皆未啟用

    # ── Inference（推理層，僅紀錄；實際一律由 LLM_CONFIG 覆蓋）──
    "gui_system_prompt": "",             # 必須留空，人格一律由專案端注入
    "reasoning_budget": "Unrestricted",
    "context_overflow": "截斷中間",       # 溢出時被砍的是「中間」，即摘要與早期對話
    "stop_strings": [],
    "structured_output": False,
}


# ─────────────────────────────────────────────────────────────
# 測試模式
# ─────────────────────────────────────────────────────────────
# 用於《AOFOS-固定評測題組》施測與 LoRA 對照組實驗。
#
# ⚠️ 這裡只是「旗標」，本檔案不會自己產生任何效果。
#    必須在下列位置實際讀取並分支，否則開了等於沒開：
#      - core/rag_manager.py   → TEST_MODE 為 True 時不查詢、不寫入
#      - core/memory_manager.py→ TEST_MODE 為 True 時改用 TEST_MEMORY_PATH
#
# 存在理由：手動清空記憶遲早會漏掉一次，而漏掉的那次不會有任何徵兆，
# 只會讓那批分數悄悄失效。

TEST_MODE = False

TEST_CONFIG = {
    # 測試模式下 Compressed Memory 改寫到這裡，不碰正式記憶
    "memory_path": "data/test_compressed_memory.txt",
    # 測試模式下完全停用 RAG（不查詢、不寫入）
    "disable_rag": True,

    # 【2026-07-31】評測專用推理參數：貪婪解碼（greedy decoding）
    #
    # 背景：對照組實驗需要可重現的輸出，原本打算靠固定 seed 達成，但實測失敗——
    # API 送 seed=42 無效，GUI 勾選 seed=42 也無效，Max Concurrent Predictions
    # 從 4 改為 1 之後仍然無效。相同 prompt（token 數完全一致）連續兩次輸出不同。
    # 推測與 17/32 層拆分 GPU/CPU 造成的浮點運算順序差異有關，但未進一步查證——
    # 因為不需要：換個角度就能完全繞開。
    #
    # temperature=0 搭配 top_k=1 等於直接取機率最高的 token，完全不經過取樣，
    # 隨機性從源頭消失，不依賴 seed。這對評測反而更正確：要量的是「模型最可能
    # 給出的答案」，而不是「隨機抽到的某一個樣本」。多數 benchmark 皆採此做法。
    #
    # 代價：貪婪解碼下的語氣可能比日常（temperature=0.8）平淡，因此評測題組
    # A 類（語氣）的觀察結果不能直接拿來描述日常使用體驗。但由於各組條件一致，
    # 組間比較仍然成立。
    # 【2026-07-31 追加 frequency_penalty】貪婪解碼的已知代價：退化迴圈。
    #
    # 實測（對照組 B3）：模型輸出一個 emoji 後，接連生成數百個 U+FE0F 變體選擇符，
    # 一路撞到 max_tokens 才停。temperature=0.8 時取樣有機率跳出，top_k=1 時無路可逃。
    #
    # repeat_penalty=1.1 擋不住，因為它對「出現過的 token」只施加一次固定懲罰。
    # frequency_penalty 則隨出現次數累加（logit -= count × penalty），
    # 重複越多壓得越狠，數個 token 內即可脫離迴圈；
    # 而正常文字不會有同一 token 連續出現數十次的情形，故對一般輸出影響極小。
    #
    # ⚠️ 此參數對所有受測組一律相同，故組間比較仍然成立；
    #    但變更前已完成的題目必須重跑，否則就是拿不同參數的結果互相比較。
    "llm_overrides": {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "frequency_penalty": 0.4,
    },
}


# ─────────────────────────────────────────────────────────────
# 延遲量測
# ─────────────────────────────────────────────────────────────
# 專案核心目標之一為端到端 <500ms，但在此之前沒有任何實測數字。
# 量測本身屬「既有需求的缺口」，不是預先建置。

TIMING_CONFIG = {
    # 關閉後 timed() 變成幾乎零成本的空殼，不需要移除既有的 with 區塊
    "enabled": True,
    # 額外輸出機器可讀的 jsonl，方便事後算平均。
    # 只靠終端機 log 人工加總，跑十幾題就會放棄。
    "jsonl": True,
}


# ─────────────────────────────────────────────────────────────
# 診斷腳手架（暫時性，用完即刪）
# ─────────────────────────────────────────────────────────────
# ⚠️ 這一整段以及下列位置的對應程式碼，都是為了釐清特定問題而臨時加入的，
#    問題釐清後（不論結論為何）應**整段刪除**，不保留、不註解掉。
#    保留無用的診斷程式碼會讓核心邏輯越來越難讀，而且日後沒人記得它為什麼在那裡。
#
# 目前掛著的腳手架清單（刪除時逐項清乾淨）：
#
#   1. log_opencc_changes
#      位置：core/text_utils.py 的 _log_conversion_diff() 及其呼叫點
#      目的：釐清「纔／喫／牀／爲」等罕見異體字是模型輸出的，還是 opencc s2t 轉的
#      刪除條件：確認來源後（若為 opencc，改用 s2tw 即可）
#
#   2. RAG Write/Search 計數器
#      位置：core/rag_manager.py 的 _load_stats / _save_stats / _stats 相關全部
#      輸出：data/rag_stats.json
#      目的：追蹤淘汰內容轉存 RAG 後是否查得到
#      刪除條件：命中率問題釐清後
#
#   3. （評測結束後一併檢視）TIMING_CONFIG
#      位置：core/logger.py 的 timed()
#      註：此項與上兩者不同——延遲量測屬長期需求，不一定要刪，
#          但評測結束後應決定是「保留常駐」還是「預設關閉」。

DIAGNOSTICS = {
    "log_opencc_changes": True,
}


PERSONA_CONFIG = {
    "name": "奧狐",
    "language": "繁體中文",
}