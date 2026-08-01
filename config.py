# config.py
# 統一設定檔 — 換模型 / 換 API / 調參數只改這裡

# ─────────────────────────────────────────────────────────────
# LLM 連線與推理參數
# ─────────────────────────────────────────────────────────────
# 所有影響輸出的參數一律在此明確指定並隨 request 送出。
#
# ⚠️ LM Studio 的預設參數綁在「各個模型」而非全域共用。未明確帶入的參數，
#    切換模型時會各自吃到不同預設值，使跨模型比較靜默失效且不會有任何警告。
#    下方數值請對照 LM Studio 目前載入模型的參數面板逐一抄寫。

LLM_CONFIG = {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",

    # 模型識別碼。本機可用清單由 LLMClient 啟動時列出：
    #   "aofos_v3"                               ← 實驗組（LoRA v3 合併版）
    #   "aofos_v2" / "aofos_v1"                  ← LoRA v2 / v1
    #   "meta-llama-3.1-8b-instruct-abliterated" ← 對照組 A（無 LoRA 底模）
    #   "meta-llama-3.1-8b-instruct"             ← 對照組 B（官方版）
    #
    # ⚠️ 換模型流程：Eject 舊模型 → 載入新模型 → 於 GUI 重設載入層參數
    #    （見 LMSTUDIO_LOAD_PARAMS）→ 才改這一行。JIT 已關閉，順序顛倒會直接報錯。
    "model": "aofos_v3",

    # ── 標準 OpenAI 參數（以具名參數送出）──
    "temperature": 0.8,
    "max_tokens": 300,

    # 各用途的輸出長度上限，未列出的用途沿用上方 max_tokens。
    #
    # ⚠️ compress 需一次輸出「摘要 8 句 + 淘汰清單」兩段。中文在 Llama tokenizer 下
    #    約 1 token/字，300 上限會讓「淘汰：」整段被截斷，而
    #    _parse_compress_result() 找不到該標記時不報錯、安靜回傳「無」——
    #    淘汰內容因此永遠寫不進 RAG，且無任何徵兆。
    #    若 rag_stats.json 的 writes 長期為 0 或遠低於壓縮次數，優先懷疑這裡。
    "max_tokens_by_purpose": {
        "compress": 800,
    },
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,

    # 固定 seed 可消除同一顆模型自身的隨機波動；設 None 表示不固定。
    # 註：實測本環境下 seed 無效，評測改用貪婪解碼，見 TEST_CONFIG。
    "seed": 42,

    # ── LM Studio 專屬參數（透過 extra_body 送出）──
    # 值設為 None 的欄位不會被送出。
    "top_k": 40,
    "repeat_penalty": 1.1,
    "min_p": 0.05,
}


# ─────────────────────────────────────────────────────────────
# LM Studio 載入層參數（紀錄用，非設定用）
# ─────────────────────────────────────────────────────────────
# ⚠️ 這些值無法透過 OpenAI 相容 API 設定，只能在 LM Studio GUI 手動設。
#    寫在這裡的目的：讓環境設定隨程式碼進版控，並於啟動時寫進 log，
#    使日後回看任何一次跑測的紀錄都知道當時環境是什麼。
#
# ⚠️ 載入新模型時 LM Studio 會依模型檔案大小「自動」調整 GPU 卸載層數與 context，
#    且不會警告。對照組因此可能跑在不同卸載層數上（同時影響速度與數值精度）。
#    每次換模型後務必手動改回下列數值。

LMSTUDIO_LOAD_PARAMS = {
    # ── Context and Offload ──
    # 8192 與 17 層皆為 VRAM 約束下的結果（v3 於此設定下 GPU 2.99GB / 合計 5.32GB），
    # 非任意選定；模型標稱支援 131072。若這兩個數字被自動改掉，
    # GUI 的記憶體估計值也會跟著變，可用來反查是否被改。
    "context_length": 8192,
    "gpu_offload_layers": 17,

    # ── Advanced（載入層）──
    "cpu_thread_pool": 3,
    "evaluation_batch_size": 2048,
    "physical_batch_size": 512,
    # ⚠️ 必須為 1。連續批次處理不保證逐位一致（輸出無法重現的來源之一）；
    #    且 llama.cpp 的平行槽位會「切分」總 context，parallel=4 時
    #    設 8192 實際單次僅 2048，且溢出時截掉的是中間（摘要與早期對話）。
    #    單人使用無平行需求，設 1 沒有損失。
    "max_concurrent_predictions": 1,
    "unified_kv_cache": True,
    "context_checkpoints": 32,           # prompt 快取；會影響重複前綴的延遲量測
    "rope_freq_base": "自動",
    "rope_freq_scale": "自動",
    "kv_cache_offload_to_gpu": True,
    "keep_model_in_memory": True,
    "try_mmap": True,
    "gui_seed": 42,                      # 已於 GUI 固定，但實測無效
    "speculative_decoding": "off",
    "flash_attention": True,
    "kv_cache_quantization": None,

    # ── Inference（推理層，僅紀錄；實際一律由 LLM_CONFIG 覆蓋）──
    "gui_system_prompt": "",             # 必須留空，人格一律由專案端注入
    "reasoning_budget": "Unrestricted",
    "context_overflow": "截斷中間",
    "stop_strings": [],
    "structured_output": False,
}


# ─────────────────────────────────────────────────────────────
# 對話行為參數
# ─────────────────────────────────────────────────────────────

# 主動對話的靜默觸發閾值（秒）。暫定值，尚未實測校準。
SILENCE_THRESHOLD_SECONDS = 30

# 記憶
MAX_WORKING_MEMORY = 10      # Working Memory 保留輪數（超過 ×2 則觸發壓縮）
MAX_SUMMARY_SENTENCES = 8    # Compressed Memory 上限句數，避免無限累積
COMPRESS_BATCH_SIZE = 12     # 觸發壓縮時一次取出的訊息則數（user/assistant 各算一則）

# RAG
RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.7   # 低於此分數的查詢結果直接丟棄

# 語音輸入
VOICE_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "vad_threshold": 0.5,      # 語音判定信心值門檻
    "silence_duration": 1.5,   # 靜音多久（秒）判定為說完
    "rms_gate": 0.02,          # 音量門檻，過濾底噪；換麥克風需一併校準
    "chunk_size": 512,         # Silero VAD 要求固定 512 samples
    "debug_print_rms": False,  # 校準麥克風時開啟，印出即時 rms 數值
}


# ─────────────────────────────────────────────────────────────
# 測試模式
# ─────────────────────────────────────────────────────────────
# 用於《AOFOS-固定評測題組》施測與 LoRA 對照組實驗。
#
# ⚠️ 這裡只是旗標，本檔案不會自己產生任何效果，必須在下列位置實際分支：
#      - core/rag_manager.py    → True 時不查詢、不寫入、不初始化
#      - core/memory_manager.py → True 時改用 TEST_CONFIG["memory_path"]
#      - core/llm_client.py     → True 時套用 llm_overrides
#
# 存在理由：手動清空記憶遲早會漏掉一次，而漏掉的那次不會有任何徵兆，
# 只會讓那批分數悄悄失效。

TEST_MODE = False

TEST_CONFIG = {
    "memory_path": "data/test_compressed_memory.txt",
    "disable_rag": True,

    # 評測專用推理參數：貪婪解碼（greedy decoding）。
    #
    # temperature=0 + top_k=1 直接取機率最高的 token，不經取樣，隨機性從源頭消失，
    # 不依賴 seed（本環境 seed 實測無效，詳見《決策脈絡歸檔》十六章）。
    # frequency_penalty 是貪婪解碼的必要配套：實測會出現退化迴圈
    # （同一 token 連續生成至撞上 max_tokens），而 repeat_penalty 只施加一次固定
    # 懲罰擋不住，frequency_penalty 隨出現次數累加才壓得下來。
    #
    # ⚠️ 代價：貪婪解碼下語氣較日常（temperature=0.8）平淡，A 類（語氣）觀察結果
    #    不能直接用來描述日常體驗；但各組條件一致，組間比較仍然成立。
    # ⚠️ 變更此區任何數值後，先前已完成的題目必須全部重跑。
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
# 專案核心目標之一為端到端 <500ms。量測屬「既有需求的缺口」，非預先建置。

TIMING_CONFIG = {
    # 關閉後 timed() 變成近乎零成本的空殼，不需移除既有的 with 區塊
    "enabled": True,
    # 額外輸出機器可讀的 jsonl；只靠終端機 log 人工加總，跑十幾題就會放棄
    "jsonl": True,
}


# ─────────────────────────────────────────────────────────────
# 診斷腳手架（暫時性，用完即刪）
# ─────────────────────────────────────────────────────────────
# 實作集中於 core/diagnostics.py，該檔頂部列有每項的目的與刪除條件。
# 清理時：刪除 core/diagnostics.py + 移除各呼叫點的 import 與單行呼叫 + 刪除本區塊。
#
# 註：TIMING_CONFIG 不屬於腳手架（延遲量測是長期需求），但評測結束後
#     應決定它是「保留常駐」還是「預設關閉」。

DIAGNOSTICS = {
    "log_opencc_changes": True,   # 釐清異體字來自模型輸出還是 opencc s2t
    "rag_stats": True,            # 追蹤淘汰內容轉存 RAG 後的命中率
}


PERSONA_CONFIG = {
    "name": "奧狐",
    "language": "繁體中文",
}