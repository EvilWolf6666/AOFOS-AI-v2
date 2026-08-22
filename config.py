# config.py
# 統一設定檔 — 換模型 / 換 API / 調參數只改這裡

# ── 環境變數：必須在任何 huggingface / chromadb 相關 import 之前生效 ──
import os

from dotenv import load_dotenv

# 放這裡不放 main.py：tools/ 的腳本直接 import config、不經 main.py，
# config 是兩者唯一的共同入口。
load_dotenv()

# embedding 模型已在本機快取，啟動時不連網。用 setdefault 而非指派：換模型時
# 命令列設 HF_HUB_OFFLINE=0 即可恢復下載。
# ⚠️ 快取不存在時直接報錯、不靜默改走下載——刻意如此，失敗要大聲。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ─────────────────────────────────────────────────────────────
# LLM 連線與推理參數
# ─────────────────────────────────────────────────────────────
# 所有影響輸出的參數一律在此明確指定並隨 request 送出。
# ⚠️ LM Studio 預設參數綁在「各模型」而非全域。未明確帶入者，換模型時會各自吃到
#    不同預設值，跨模型比較靜默失效且無警告。

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
    "model": "aofos_v4_colab",

    # ── 標準 OpenAI 參數（以具名參數送出）──
    "temperature": 0.8,
    "max_tokens": 300,

    # 各用途的輸出長度上限，未列出的用途沿用上方 max_tokens。
    #
    # ⚠️ compress 需一次輸出「摘要 8 句 + 淘汰清單」兩段。中文在 Llama tokenizer 下
    #    約 1 token/字，300 上限會讓「淘汰：」整段被截斷，而
    #    _parse_compress_result() 找不到該標記時不報錯、安靜回傳「無」——
    #    淘汰內容因此永遠寫不進 RAG，且無任何徵兆。
    #    徵兆在 memory_manager._compress() 的 log：若長期只出現「本次壓縮無淘汰
    #    內容」而從不出現「本次壓縮淘汰內容（將轉存 RAG）」，優先懷疑這裡。
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

    # ── 串流輸出 ──
    # True 時逐 token 印出。不影響 KPI（端到端 <500ms），只影響等待期間的體感。
    # ⚠️ 只作用於有接串流回呼的呼叫點（目前僅 handle_turn / handle_observation）；
    #    _compress() 一律走非串流。施測時由 TEST_CONFIG["disable_stream"] 強制關閉。
    "stream": True,
}


# ─────────────────────────────────────────────────────────────
# LM Studio 載入層參數（紀錄用，非設定用）
# ─────────────────────────────────────────────────────────────
# ⚠️ 這些值無法經 OpenAI 相容 API 設定，只能在 LM Studio GUI 手動設；此處僅供
#    版控與啟動 log 存證。
# 🔴 載入新模型時 LM Studio 會依檔案大小「自動」改掉 GPU 卸載層數與 context 且不
#    警告——對照組會跑在不同卸載層數上。每次換模型後務必手動改回下列數值。

LMSTUDIO_LOAD_PARAMS = {
    # ── Context and Offload ──
    # 兩者皆為 VRAM 約束下的結果，非任意選定（模型標稱支援 131072）。
    # 被自動改掉時 GUI 的記憶體估計值會跟著變，可據此反查。
    "context_length": 8192,
    "gpu_offload_layers": 17,

    # ── Advanced（載入層）──
    "cpu_thread_pool": 3,
    "evaluation_batch_size": 2048,
    "physical_batch_size": 512,
    # ⚠️ 必須為 1。批次處理不保證逐位一致；且平行槽位會「切分」總 context
    #    （parallel=4 時 8192 實際單次僅 2048），溢出時截掉的是中間。
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

# 記憶
MAX_WORKING_MEMORY = 10      # Working Memory 保留輪數（超過 ×2 則觸發壓縮）
MAX_SUMMARY_SENTENCES = 8    # Compressed Memory 上限句數，避免無限累積
COMPRESS_BATCH_SIZE = 12     # 觸發壓縮時一次取出的訊息則數（user/assistant 各算一則）

# RAG
RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.55   # 低於此分數的查詢結果直接丟棄

# 輸出後處理
TEXT_CONFIG = {
    # opencc 繁簡轉換（s2tw）是否啟用。目前 False。
    #
    # 判讀方式：關閉後看輸出**有沒有簡體字**——有就改回 True。
    # 🔴 不可改用 log_opencc_diff() 的計數判斷：s2* 對已是繁體的輸入不是 no-op，
    #    diff 有可能正是它把本來就對的字改壞。
    # ⚠️ 關閉走早退、不是移除，import 與 converter 一律保留，恢復只改這個布林。
    # 🔴 s2twp 永遠不可用：它替換的是**用詞**（软件→軟體），而用詞是 LoRA 訓出來的。
    #    後處理再改一次等於在權重之外覆寫說話方式，且不會出現在任何 log 裡。
    "opencc_enabled": False,
}

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
# 事件驅動
# ─────────────────────────────────────────────────────────────
# observation 事件（環境狀態變化，沒有人在說話）的注入參數。
# 對應 core/agent/pipeline.py 的 handle_observation()。

EVENT_CONFIG = {
    # observation 輪次的 user 欄位內容。
    #
    # ⚠️ 訓練端與推理端的結構必須一致。若 LLaMA-Factory 無法以空 prompt 訓練、
    #    訓練資料改用最小開場白，這裡要同步換成同一句話，否則模型學到的是
    #    「看到某句話 → 主動說話」，而推理時那句話根本不存在。
    #    此為目前唯一需要改動的位置。
    "user_placeholder": "",

    # 附在情境描述之後的引導句。⚠️ 過渡期產物，應維持空字串——訓練資料不含
    #    引導句，推理端結構須與訓練端一致。非空值會把情境框成「有人在跟妳講話」。
    "observation_prompt_suffix": "",

    # 情境訊息的 role 名稱。🔴 待定，值暫時不動（《現況》P18）。
    #
    # ⚠️ 兩套 role 定義交錯，交集裡沒有兩邊都正確的選項：
    #      "ipython" 過不了 LM Studio 白名單（400，實測）
    #      "tool"    非 Llama-3.1 合法 role，且 v4 的 template 把它原樣插進 header
    #                （渲染出 <|start_header_id|>tool，模型沒見過）
    #    v3 曾有 tool → ipython 映射分支，v4 沒有——依 v3 時代的推理做決定會出錯。
    # ⚠️ 讀完 P18 之前不要改這個值。
    "observation_role": "tool",

    # 註："test_observations" 腳手架已移除（C4）。日後若再需要手動注入情境，
    #     正確做法是新增 producer 而非加回腳手架——路徑必須與真實事件相同。
}


# ─────────────────────────────────────────────────────────────
# 事件仲裁
# ─────────────────────────────────────────────────────────────
# 所有事件在 Agent 的唯一取用點（event_queue.get()）之後、分流之前經此判斷。
# ⚠️ 仲裁邏輯必須集中於單一位置；下放到各 producer 會造成多來源同時輸出。

ARBITRATION_CONFIG = {
    # 奧狐發言後多久內不再處理環境事件（秒）。冷卻只擋 observation，不擋人說話。
    # 下限來自一次推理的耗時，短於此則佇列積壓。
    "observation_cooldown_seconds": 30,

    # ⚠️ TTS 接入後才存在的兩項（插話 QUEUE/INTERRUPT、中斷與靜音通道）位置已
    #    預留於 Agent._arbitrate()，一律加在該處、不得下放到 producer。
}


# ─────────────────────────────────────────────────────────────
# 事件來源：前景視窗監看
# ─────────────────────────────────────────────────────────────
# 第一個真實 producer：成本與錯誤率近乎為零，且不依賴模型路線。
# ⚠️ 「畫面上發生什麼」需要 VL，屬另一個 producer，兩者互補不重疊。

WINDOW_WATCHER_CONFIG = {
    "enabled": True,

    # 輪詢間隔（秒）。Win32 API 呼叫成本極低，但過密無意義——
    # 人切換視窗的頻率遠低於每秒一次。
    "poll_interval_seconds": 2.0,

    # 同一程式停留多久後推一次「久無切換」事件（秒）。
    # 推過一次後需再次達到此門檻才會再推，不會每輪都推。
    "idle_threshold_seconds": 600,

    # 不值得產生事件的程式（小寫比對）。使用者在這些程式之間切換屬於
    # 操作噪音，不是「他改去做別的事」。
    "ignored_processes": [
        "explorer.exe",
        "searchhost.exe",
        "shellexperiencehost.exe",
        "textinputhost.exe",
        "lockapp.exe",
    ],

    # 程式名 → 人類可讀名稱。找不到時退回程式名本身（不隱藏未知程式，
    # 否則新遊戲會靜默變成「未知」而沒人發現需要補這張表）。
    "process_display_names": {
        "code.exe": "VS Code",
        "chrome.exe": "瀏覽器",
        "msedge.exe": "瀏覽器",
        "firefox.exe": "瀏覽器",
        "javaw.exe": "Minecraft",
        "steam.exe": "Steam",
        "lmstudio.exe": "LM Studio",
        "windowsterminal.exe": "終端機",
        "powershell.exe": "終端機",
    },
}


# ─────────────────────────────────────────────────────────────
# 測試模式
# ─────────────────────────────────────────────────────────────
# 用於《AOFOS-固定評測題組》施測與 LoRA 對照組實驗。
#
# ⚠️ 這裡只是旗標，實際分支在三處：
#      core/rag_manager.py    → True 時不查詢、不寫入、不初始化
#      core/memory_manager.py → True 時改用 TEST_CONFIG["memory_path"]
#      core/llm_client.py     → True 時套用 llm_overrides

TEST_MODE = False

TEST_CONFIG = {
    "memory_path": "data/test_compressed_memory.txt",
    "disable_rag": True,

    # 施測期間強制關閉串流。
    # ⚠️ 「串流與非串流在貪婪解碼下逐字相同」尚未驗證，施測不賭這個假設。
    # 驗證方式（做完即可刪本項）：同一題以 stream=True / False 各跑一次，
    #    貪婪解碼下應逐字相同。
    "disable_stream": True,

    # 評測專用：貪婪解碼。temperature=0 + top_k=1 直接取機率最高的 token，
    # 隨機性從源頭消失，不依賴 seed（本環境 seed 實測無效）。
    # frequency_penalty 是必要配套——貪婪解碼會出現退化迴圈，而 repeat_penalty
    # 只施加一次固定懲罰擋不住，需隨出現次數累加才壓得下來。
    #
    # ⚠️ 代價：語氣較日常平淡，A 類觀察不可直接用來描述日常體驗；組間比較仍成立。
    # 🔴 變更此區任何數值後，先前已完成的題目必須全部重跑。
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
# 實作集中於 core/short_time_test.py，該檔頂部列有每項的目的與刪除條件。
# 清理時：刪該檔 + 移除各呼叫點的 import 與單行呼叫 + 刪本區塊。
# 註：TIMING_CONFIG 不屬腳手架（延遲量測是長期需求），評測結束後再決定去留。

DIAGNOSTICS = {
    # ⚠️ 現行目的：換模型後判斷 opencc 這一層還需不需要（舊目的「釐清異體字來源」
    #    已達成且證偽）。見 C1。
    "log_opencc_changes": True,
    "dump_observation_messages": True, # 印出 observation 輪次送出的完整 messages
}


PERSONA_CONFIG = {
    "name": "奧狐",
    "language": "繁體中文",
}