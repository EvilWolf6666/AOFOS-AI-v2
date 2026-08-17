# config.py
# 統一設定檔 — 換模型 / 換 API / 調參數只改這裡

# ── 環境變數：必須在任何 huggingface / chromadb 相關 import 之前生效 ──
import os

from dotenv import load_dotenv

# 放在 config.py 而非 main.py 的理由：tools/ 底下的腳本直接 import config，
# 完全不經過 main.py。原本 load_dotenv() 只寫在 main.py，導致 seed_rag 與
# rag_probe 執行時 .env 從未被讀入、HF_TOKEN 一直是空的（2026-08-06 查證）。
# config 是 main.py 與所有 tools 唯一的共同入口，放這裡才是單一來源。
load_dotenv()

# embedding 模型已在本機快取，啟動時不需要連網查版本。施測協定要求每題重啟程式，
# 每次啟動都打一次 HuggingFace 是累積的延遲，網路不順時還會拖慢甚至卡住。
#
# 用 setdefault 而非直接指派：要換 embedding 模型時，命令列臨時設
# HF_HUB_OFFLINE=0 就能恢復下載，不必回頭改這裡。
#
# ⚠️ 快取不存在時會直接報錯，而不是靜默改走下載——這是刻意的，失敗要大聲。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    # True 時逐 token 印出。KPI（端到端 <500ms）不因此達成——3 tok/s 的生成速度下
    # 第二個字就已超過——但等待期間的體感差異極大：關閉時整段生成完才印，
    # 實測 4,748～15,474ms 之間畫面完全空白。
    #
    # ⚠️ 只影響「有接串流回呼的呼叫點」（目前僅 handle_turn / handle_observation）。
    #    _compress() 沒有消費者要看它逐字吐，一律走非串流路徑。
    #
    # ⚠️ 施測時由 TEST_CONFIG["llm_overrides_stream"] 強制關閉，見該處說明。
    "stream": True,
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

# 記憶
MAX_WORKING_MEMORY = 10      # Working Memory 保留輪數（超過 ×2 則觸發壓縮）
MAX_SUMMARY_SENTENCES = 8    # Compressed Memory 上限句數，避免無限累積
COMPRESS_BATCH_SIZE = 12     # 觸發壓縮時一次取出的訊息則數（user/assistant 各算一則）

# RAG
RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.55   # 低於此分數的查詢結果直接丟棄

# 輸出後處理
TEXT_CONFIG = {
    # opencc 繁簡轉換（s2tw）是否啟用。
    #
    # ⚠️ 2026-08-16 設為 False：暫時關閉以觀察「不經這層時模型原生輸出長什麼樣」。
    #    這一層存在的前提是「現行模型會吐簡體字」，而該前提從未在關閉狀態下被驗過。
    #
    # 判讀方式（🔴 直接看輸出，不要只看 diff 計數）：
    #    關閉後跑一輪，輸出**有沒有簡體字**。
    #      沒有 → 這層在現行模型上已無作用，換模型時可正式評估整層移除
    #      有   → 這層還在做事，改回 True
    #
    # 🔴 為什麼不能只看 log_opencc_diff() 的計數：
    #    「有 diff = opencc 有價值」這條判讀規則本身沒被驗證過。s2* 系列對
    #    「已經是繁體」的輸入**不是 no-op**，diff 有可能正是它把本來就對的字改壞。
    #    見《歸檔》一之14。
    #
    # ⚠️ 關閉是早退，不是移除——opencc 的 import 與 converter 建立一律保留。
    #    後期要恢復只需把這裡改回 True，不必回頭動三個檔。
    #
    # 🔴 s2twp 永遠不可用：它會替換**用詞**（软件→軟體），而用詞是 LoRA 訓出來的。
    #    後處理再改一次等於在權重之外偷偷覆寫她的說話方式，且不會出現在任何 log 裡，
    #    所有 LoRA 版本比較都會被污染。
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

    # 附在情境描述之後的引導句。
    #
    # ⚠️ 過渡期產物，最終應維持空字串——LoRA 訓練資料的格式是 system 只放情境、
    #    A 直接反應，不含引導句，推理端結構需與訓練端一致。
    # ⚠️ 2026-08-02 起設為空字串進行對照。先前值「妳可以針對這件事說點什麼。」
    #    疑似把情境框成「有人在跟妳講話」——情境為「邪狼切去看影片」時，回應為
    #    「又來干擾我了」，把邪狼自己的行為誤讀成對她的互動。若空字串下方向轉
    #    正常，此項可直接刪除；舊值與三則對照結果記於《決策脈絡歸檔》二十二之5 供還原。
    "observation_prompt_suffix": "",

    # 情境訊息的 role 名稱。現行值 "tool"。
    #
    # 🔴 2026-08-16：本項的原始理由已在 v4 上失效，值暫時不動、待定（《現況》P18）。
    #
    #   原理由：「模型的 chat_template 會把 role == "tool" 渲染成
    #            <|start_header_id|>ipython，與訓練端 LLaMA-Factory 一致」
    #   實測：  該敘述只在 aofos_v3 成立。v3 的 template 含 tool → ipython 映射分支；
    #          v4 帶的是 Llama-3.0 風格簡化版，把 role 字串**原樣**插進 header，
    #          因此 v4 上渲染出的是 <|start_header_id|>tool。
    #
    # ⚠️ "tool" 並不是 Llama-3.1 的合法 role（官方僅 system/user/assistant/ipython）。
    #    它能送出去只因為 LM Studio 的白名單用的是 OpenAI 的 role 集合。
    #    兩套定義交錯，交集裡沒有任何一個是兩邊都正確的：
    #      "ipython" 過不了 LM Studio 白名單（400，2026-08-03 實測）
    #      "tool"    過不了 v4 的 chat_template（渲染成模型沒見過的 header）
    #
    # ⚠️ 不要在讀完《歸檔》六之11 與 P18 之前改這個值。
    #    改動前也不要照 v3 時代的推理（六之5）做決定——那份推理的前提已不成立。
    "observation_role": "tool",

    # 註：原有 "test_observations" 腳手架（/proactive test 的循環取用清單）已於
    #     2026-08-16 移除，刪除條件（第一個真實 producer 建立）已由
    #     WINDOW_WATCHER_CONFIG 於 2026-08-02 滿足。
    #     手動注入情境的需求若日後再現，正確做法是新增一個 producer，
    #     不是把腳手架加回來——走的路徑必須與真實事件相同。
}


# ─────────────────────────────────────────────────────────────
# 事件仲裁
# ─────────────────────────────────────────────────────────────
# 所有事件在 Agent 的唯一取用點（event_queue.get()）之後、分流之前經此判斷。
#
# ⚠️ 仲裁邏輯必須集中於單一位置。分散到各 producer 內部各自決定要不要發言，
#    會造成多來源同時輸出（Lumi_Nox 失效案例，見《決策脈絡歸檔》十一之1、十一之3）。

ARBITRATION_CONFIG = {
    # 奧狐發言後多久內不再處理環境事件（秒）。
    #
    # 使用者輸入不受此限制——冷卻只擋 observation，不擋人說話。
    # 下限來自實測：一次推理 8～15 秒，冷卻若短於此，佇列會積壓。
    "observation_cooldown_seconds": 30,

    # ⚠️ 尚未實作、位置已預留的兩項（① 類，TTS 接入主迴圈後才存在）：
    #      · 奧狐正在說話時使用者插話 → QUEUE / INTERRUPT 的選擇
    #      · 中斷與靜音通道
    #    目前文字模式為同步阻塞，輸出期間不可能有新輸入，故兩者皆不存在。
    #    實作時一律加在 Agent._arbitrate()，不得下放到 producer。
}


# ─────────────────────────────────────────────────────────────
# 事件來源：前景視窗監看
# ─────────────────────────────────────────────────────────────
# 第一個真實 producer。選它而非螢幕畫面（VL）的理由見《決策脈絡歸檔》二十之7 之8：
# 成本近乎為零、錯誤率為零、不依賴模型路線，且它知道的事（他在用什麼程式）
# 永遠不會被 VL 取代。
#
# ⚠️ 「畫面上發生什麼」需要 VL，屬另一個 producer，待模型路線定案。兩者互補不重疊。

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

    # 施測期間強制關閉串流。
    #
    # ⚠️ 「串流與非串流在貪婪解碼下逐字相同」是一個**尚未驗證的假設**——理論上
    #    兩者走同一條 forward pass，差別只在傳輸方式；但本專案已有多次「理論上
    #    成立、實際上另一層在動手腳」的紀錄（見《歸檔》一之14）。
    #    在驗過之前，施測一律走已知的那條路徑，不拿 v1 題組去賭這個假設。
    #
    # 驗證方式（做完即可移除本項）：同一題以 stream=True / False 各跑一次，
    # 貪婪解碼下應逐字相同。相同 → 本項可刪；不同 → 找出差在哪再說。
    "disable_stream": True,

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
# 實作集中於 core/short_time_test.py，該檔頂部列有每項的目的與刪除條件。
# 清理時：刪除 core/short_time_test.py + 移除各呼叫點的 import 與單行呼叫 + 刪除本區塊。
#
# 註：TIMING_CONFIG 不屬於腳手架（延遲量測是長期需求），但評測結束後
#     應決定它是「保留常駐」還是「預設關閉」。
#
# 註：EVENT_CONFIG["test_observations"] 已於 2026-08-16 移除（C4 結案）。

DIAGNOSTICS = {
    # ⚠️ 目的已於 2026-08-06 更新：舊目的（釐清異體字來源）已達成且證偽，
    #    現行目的是換模型後判斷 opencc 這一層還需不需要（《歸檔》三十二之9）。
    "log_opencc_changes": True,
    "dump_observation_messages": True, # 印出 observation 輪次送出的完整 messages
}


PERSONA_CONFIG = {
    "name": "奧狐",
    "language": "繁體中文",
}