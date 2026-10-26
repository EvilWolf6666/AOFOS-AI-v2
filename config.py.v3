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
# 專案根目錄 — 全專案唯一的解析點
# ─────────────────────────────────────────────────────────────
# 🔴 任何模組都不得自己用 os.path.dirname(__file__) + ".." 推算專案根。
#    那個寫法把「這個檔在第幾層」寫死進程式碼，**檔案一搬就指到別的地方，
#    而且不會報錯**——os.makedirs(exist_ok=True) 與 get_or_create_collection
#    會乖乖在錯的位置建一套空的，啟動訊息印「無歷史記憶，從零開始」（那是一句
#    正常的訊息），而 .gitignore 的 `data/` 不帶前導斜線、任何層級都忽略，
#    連版控都看不到。⇒ 三層遮蔽疊在一起，症狀是「搬檔成功了，但成功的是另一件事」。
#
# config.py 固定在專案根，深度不會變，故由它解析一次、其餘模組一律引用。
# 依《現況》零章第八條（資料要收斂到單一來源）與本檔既有的「唯一設定入口」定位。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# 推理後端：llama-server（llama.cpp 官方伺服器）
# ─────────────────────────────────────────────────────────────
# 🔺 2026-10-10 由 LM Studio 換成 llama-server（舊的 LMSTUDIO_LOAD_PARAMS 已移除，
#    LM Studio 時期的 GUI 設定值見《歸檔》B0013）。
# 本區是載入層參數的**唯一來源**：tools/start_llama_server.py 讀它組啟動指令、
#    LLMClient 讀它當 context 預算分母、backend_check 啟動時拿它比對實際載入狀態。
# ⚠️ 改了本區任何值 ⇒ 重開 llama-server（start\啟動推理後端.bat）才生效；
#    改了載入參數，延遲量測要重量（《現況》〈📐 規則〉三）。
# 換模型：改 "model" 與 "alias" → 重開 llama-server → 重開 AOFOS。

LLAMA_SERVER = {
    # 必須與執行檔一致，啟動腳本會跑 --version 比對，不符即停（文件記的版號才是真的）。
    "build": "b11541",
    "exe": r"C:\Users\EvilWolf\Desktop\AI-project\AOFOS-AI\llama.cpp\llama-server.exe",
    # v3 GGUF（沿用 LM Studio 下載的檔案，不另存一份）。
    "model": r"C:\Users\EvilWolf\.lmstudio\models\EvilWolf1956\Aofos_v3_Q4_K_M\Aofos_v3_Q4_K_M.gguf",
    # 模型識別碼：server 回應裡的 model 欄位＝這個值，LLMClient 拿它比對。
    #   沿用 LM Studio 時期的命名，舊 log 與施測紀錄才對得上：
    #   "aofos_v3"                               ← 實驗組（LoRA v3 合併版）
    #   "aofos_v2" / "aofos_v1"                  ← LoRA v2 / v1
    #   "meta-llama-3.1-8b-instruct-abliterated" ← 對照組 A（無 LoRA 底模）
    #   "meta-llama-3.1-8b-instruct"             ← 對照組 B（官方版）
    "alias": "aofos_v3",
    "host": "127.0.0.1",
    "port": 8080,

    # ── 載入參數。🔺 2026-10-11 依 tools/chat_probe 四組單一變數實驗定（GTX 1650，n=1／組）──
    # ⚠️ 換設備整組重量（〈📐 規則〉三：延遲結論不可跨設備轉移）。
    # context 與卸載層數皆為 VRAM 約束下的結果（GTX 1650 4GB），模型標稱支援 131072。
    "ctx": 8192,
    "ngl": 17,            # 16GB 級顯卡到位後改 99（整顆上 GPU）
    # ⚠️ 必須為 1。平行槽位會切分總 context（4 槽時單次只剩 2048），且批次不保證逐位一致。
    "parallel": 1,
    # CPU 執行緒（-t；處理 prompt 的 -tb 沒另設時跟著它）。3 → 4（＝實體核心數）：
    #   少於約 32 token 的 prefill 走 CPU 那幾層，每輪 3.4 → 2.8 秒；生成速度不變（約 5 tok/s，
    #   推測卡在記憶體頻寬，非 CPU）。≥32 token 時 llama.cpp 改把 CPU 層的計算搬上 GPU，快一個量級。
    "threads": 4,
    # 邏輯批次。2048 → 512：server 似乎做完一批才檢查取消（推測，log 時間點吻合），
    #   背景判斷被打斷後讓出推理槽 12～13 → 4.8 秒；大段 prefill 速度不變（664 tok 都是 7.9 秒）。
    "batch": 512,
    "ubatch": 512,        # 實體批次；不得大於 batch
    "flash_attn": "on",   # on／off／auto
    # 附加參數。-cram＝RAM 提示詞快取上限（MiB，server 預設 8192）：對話與背景判斷輪流用同一推理槽，
    #   靠它把彼此的前綴存在 RAM、切回來不必整段重算（2026-10-11 log 確認有在作用）。
    #   壓到 1024：系統 RAM 只有 8GB 級時，預設上限可能吃光記憶體而開始換頁。
    # ⚠️ 單一變數實驗也加在這裡；--cache-reuse 屬片段式重用，可能有偏差 ⇒ 施測時不得加（《現況》11之2 己）。
    "extra_args": [],

    # ── 其餘推理後端參數（2026-10-24 邪狼：先全部列上，哪怕還用不到）──
    # 規則：值為 None／False ＝不送（用推理後端預設）；True ＝只送旗標本身；其他值 ＝「旗標 值」。
    # ⚠️ 旗標名取自 llama.cpp 官方 server README（master，2026-10-24 查）；版本間會改名。
    #    啟動腳本會先跑 --help，**啟用中的旗標在本機版本找不到就停下**，不會帶著錯參數開機。
    # ⚠️ 取樣參數（溫度、top_k、懲罰…）不在這裡：一律由 LLM_CONFIG 隨每個請求送出（見下方說明）。
    # 不列：模型下載（-hf 等）、多模態、router、Web UI／工具／MCP、embedding／rerank、SSL、內建預設模型。
    "optional_args": {
        # ── 記憶體與快取 ──
        # RAM 提示詞快取上限（MiB，預設 8192）。對話與背景判斷輪流用同一推理槽，靠它存彼此前綴、
        #   切回來不必整段重算（2026-10-11 log 確認有作用）；壓到 1024：系統 RAM 只有 8GB 級，預設可能吃光而換頁。
        "-cram": 1024,   # 用短旗標：與 10-11 起實際跑的指令逐字相同（長名 --cache-ram）
        "--cache-reuse": None,          # 片段式 KV 重用的最小區塊（預設 0＝關）。⚠️ 可能有偏差，施測時不得開（《現況》11之2 己）
        "--no-cache-prompt": False,     # True＝關掉提示詞快取（只在量「不吃快取」的成本時用）
        "--no-cache-idle-slots": False, # True＝新任務時不把閒置槽存進 RAM 快取（預設會存）
        "--kv-unified": None,           # True＝多槽共用一塊 KV（槽數為 auto 時預設開）；parallel 實驗時再看
        "--kv-unified-per-slot": None,  # 每槽 context 上限（N）；不設 -c 時總量＝parallel×N
        "--ctx-checkpoints": None,      # 每槽最多幾個 context 檢查點（預設 32）
        "--cache-type-k": None,         # KV 精度：f16（預設）／q8_0／q4_0…；省 VRAM 但可能影響輸出，要量
        "--cache-type-v": None,
        "--no-kv-offload": False,       # True＝KV 不放 GPU（省 VRAM、變慢）
        "--slot-prompt-similarity": None,  # 請求要多像某槽的 prompt 才沿用該槽（預設 0.10）
        "--slot-save-path": None,       # 存槽 KV 到磁碟的目錄（預設關）
        "--context-shift": False,       # True＝超出 context 時丟掉舊 token 繼續生成（預設關）
        "--keep": None,                 # context shift 時保留開頭幾個 token（-1＝全保留）

        # ── 載入與裝置 ──
        "--fit": None,                  # on／off：自動調整「沒設的參數」以塞進顯存（預設 on；我們有設的不受影響）
        "--fit-target": None,           # 每張卡保留的顯存餘裕（MiB，預設 1024）
        "--load-mode": None,            # auto（預設）／none／mmap…：載入方式
        "--override-tensor": None,      # 指定張量放哪個裝置（例："ffn_.*=CPU"），VRAM 不夠時的細部調整
        "--device": None,               # 指定用哪些裝置（--list-devices 查）
        "--main-gpu": None,
        "--tensor-split": None,         # 多卡時各卡比例
        "--split-mode": None,           # none／layer／row／tensor
        "--no-repack": False,
        "--check-tensors": False,       # True＝載入時檢查權重有沒有壞值（慢）
        "--lora": None,                 # 額外掛 LoRA adapter 檔（現行用合併好的 GGUF，不需要）
        "--lora-init-without-apply": False,

        # ── CPU 與執行緒 ──
        "--threads-batch": None,        # 處理 prompt 的執行緒（預設跟 threads 相同）
        "--threads-http": None,         # HTTP 處理執行緒
        "--prio": None,                 # 行程優先序：-1 低／0 一般（預設）／1／2／3 即時
        "--poll": None,                 # 等待工作時的輪詢程度 0～100（預設 50）
        "--numa": None,

        # ── 伺服器行為 ──
        "--cont-batching": None,        # 連續批次（預設開）；False 不送、要關請用 extra_args 加 --no-cont-batching
        "--timeout": None,              # 讀寫逾時秒數（預設 3600）
        "--sse-ping-interval": None,    # 串流保活 ping 秒數（預設 30，-1 關）
        "--sleep-idle-seconds": None,   # 閒置多久後休眠（預設 -1＝不休眠）
        "--no-warmup": False,           # True＝開機不做空跑預熱（開機快一點、第一句慢一點）
        "--props": False,               # True＝允許 POST /props 改全域設定
        "--no-slots": False,            # True＝關掉 /slots 監看端點
        "--api-key": None,              # 只聽本機，不需要
        "--log-prompts-dir": None,      # 除錯用：把 prompt 寫成檔
        "--log-verbosity": None,        # 0～4
        "--log-timestamps": None,

        # ── 模板與思考 ──
        "--chat-template-file": None,   # 覆寫聊天模板。⚠️ 必須與訓練端一致，否則推理與訓練格式不同
        "--chat-template-kwargs": None,
        "--reasoning": None,            # on／off／auto：思考模式（Llama 3.1 沒有，維持預設）
        "--no-prefill-assistant": False,

        # ── 加速（推測解碼；需要另一顆小模型或 n-gram，未試）──
        "--spec-type": None,            # 例：ngram-simple／draft-simple…
        "--model-draft": None,          # 草稿模型路徑
        "--gpu-layers-draft": None,
    },
}


# ─────────────────────────────────────────────────────────────
# LLM 連線與推理參數
# ─────────────────────────────────────────────────────────────
# 所有影響輸出的參數一律在此明確指定並隨 request 送出。
# ⚠️ 沒帶的參數會吃 server 預設值，或 GGUF 內嵌的取樣值（general.sampling.*，《現況》第五章 #5）
#    ——換模型時會各自不同，跨模型比較靜默失效且無警告。

LLM_CONFIG = {
    # 由 LLAMA_SERVER 組出，host／port 只存一份。
    "base_url": f"http://{LLAMA_SERVER['host']}:{LLAMA_SERVER['port']}/v1",
    # llama-server 沒開 --api-key 時不檢查；OpenAI SDK 要求非空才填。
    "api_key": "local",

    # 換模型改 LLAMA_SERVER，這裡跟著走（不得另寫一份）。
    "model": LLAMA_SERVER["alias"],

    # ── 標準 OpenAI 參數（以具名參數送出）──
    "temperature": 0.8,
    "max_tokens": 300,

    # 各用途的輸出長度上限，未列出的用途沿用上方 max_tokens（llm_client 依 purpose 查）。
    # 🔺 2026-10-09 移除 v1 的 "compress": 800（Compressed Memory 已於 2026-09-26 移出，無呼叫點）。
    "max_tokens_by_purpose": {},
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,

    # 日常一律 None（每次請求隨機）。
    # 🔺 2026-10-11 由 42 改 None：LM Studio 時期 seed 實測看起來無效；llama-server 會照 seed 執行，
    #    固定 seed 讓每次請求用同一串亂數，內容不同的輸入連續 5 輪抽出同一句「你好，哥！」（2026-10-11 實測）。
    #    施測不靠 seed：TEST_CONFIG 用貪婪解碼，結果本來就固定。
    "seed": None,

    # ── 非 OpenAI 標準的取樣參數（llama-server 原生支援；SDK 要經 extra_body 送出）──
    # 值設為 None 的欄位不會被送出。
    "top_k": 40,
    "repeat_penalty": 1.1,
    "min_p": 0.05,

    # ── 串流輸出 ──
    # True 時逐 token 印出。不影響 KPI（端到端 <500ms），只影響等待期間的體感。
    # ⚠️ 只作用於有接串流回呼的呼叫點（目前僅 handle_turn / handle_observation）。
    #    施測時由 TEST_CONFIG["disable_stream"] 強制關閉。（背景判斷另走 BackgroundLLM，一律串流。）
    "stream": True,
}


# ─────────────────────────────────────────────────────────────
# 對話行為參數
# ─────────────────────────────────────────────────────────────

# 記憶
# Working Memory 滑動視窗（core/memory/manager.py 的 _slide）。
# 🔺 2026-10-04：10 輪／每次丟 1 輪 → 20 輪／每次丟 10 輪。
#    理由（實測，Windows，LM Studio 時期）：每輪丟 1 輪會讓送給模型的開頭每輪都變 ⇒ 後端的提示詞快取對不上、
#    整段重讀，滿 10 輪後**每輪** TTFT 多 2.5～3 秒。一次丟一批 ⇒ 快取每 WM_DROP_ROUNDS 輪才斷一次。
#    剛丟完時她看得到 MAX＋1−DROP 輪（現為 11，不少於舊上限 10）。
# ⚠️ 上限受 context 8192 約束：人格約 550＋回覆保留 300＋記憶區塊約 200 token；她回覆變長時每輪
#    可到 300 token 以上 ⇒ 20 輪是安全值。超過 8192 時 LM Studio 時期是「截斷中間」**靜默**切掉對話；
#    llama-server 超過時是報錯還是截斷，未實測（換後端後待驗）。
MAX_WORKING_MEMORY = 20      # 最多保留幾輪（一輪＝一問一答）
WM_DROP_ROUNDS = 10          # 超過上限時一次丟掉最舊的幾輪（1＝舊行為，每輪都斷快取）

# ── 記憶（core/memory/manager.py；準 v4，《現況》〈🔒 隔離區〉記憶模塊〈準 v4〉）──────────
# 🔺 2026-10-09 準 v3 讀取路徑淘汰：撈原句（top_k／neighbor_n／RRF）、原句向量、開場的
#    原句（carry_rounds）與 v3_recall／v3_opening 開關的參數已刪。開場的狀態與上一場關鍵詞留下。
# 🔺 2026-10-09 一併移除 v1 遺留、已無讀者的參數：MAX_SUMMARY_SENTENCES、COMPRESS_BATCH_SIZE、
#    RAG_TOP_K／RAG_SIMILARITY_THRESHOLD／RAG_INCLUDE_SEED、TEST_CONFIG 的 memory_path／disable_rag。
MEMORY_CONFIG = {
    # 日常區與測試區是兩個獨立檔。/test on 時「讀取共用、寫入分流」（第八章 舊C8（已結案））：
    # 讀兩個、寫只寫測試區；進出測試模式時整包丟棄測試區。
    "db_path": os.path.join(PROJECT_ROOT, "data", "memory.db"),
    "test_db_path": os.path.join(PROJECT_ROOT, "data", "memory_test.db"),
    "max_query_terms": 64,   # 主題關鍵字那一路：查詢最多取幾個 bigram（長輸入防 OR 爆量）
    # 說話者稱呼（審核工具印原句時用）。⚠️ 只影響顯示文字，不寫進資料庫——改了不需遷移。
    "speaker_labels": {"user": "邪狼", "assistant": "奧狐"},

    # ── 注入格式（prompt 版：不訓練，靠區塊自帶說明；2026-09-27 邪狼裁定）──
    # 撈到的記憶以一則 system 訊息、用下面兩個符號包夾，緊接在本輪使用者輸入之前。
    # ⚠️ 只用一般文字符號，不加進 tokenizer（加新 token 要訓 embed／lm_head）。
    "block_open": "⟦記憶⟧",
    "block_close": "⟦/記憶⟧",
    # 區塊裡兩段的標題（2026-10-22 邪狼裁定對齊人格 prompt V3）：記得＝開場（上次聊天的狀態與上一場關鍵詞），
    #    想起＝依本輪輸入撈到的主題。🔴 字串必須與 prompts/aofos_persona.txt〈記憶誠實原則〉逐字相同——
    #    人格規定「只有這兩段裡的才算知道」，改了一邊沒改另一邊，記憶會被她當成不知道而不報錯。
    "label_remember": "妳記得的事",
    "label_recall": "妳想起的事",
    # 區塊第一行的自我說明。⚠️ 刻意放在區塊裡、不改人格 prompt——人格 prompt 是
    #    施測變數（SPT 在比較序列開跑後不得改動），記憶說明跟著記憶走才不互相牽動。
    "block_note": "（以下是你記得的過去，不是對方現在說的話；和現在的話題有關才用，用自己的話帶出來，不要照抄原句。）",
    # 說明放哪（2026-10-11 邪狼裁定試 "fixed"；單一變數，改回 "inline" 即完全回到舊行為）：
    #   "inline"＝block_note 放在每輪的區塊裡（舊做法）。區塊每輪不同又不留在歷史，
    #            說明連同區塊每輪都要重算（實測每輪重算 222～314 token、約 3 秒，GTX 1650）。
    #   "fixed" ＝改送 fixed_note：一則永遠不變的 system 訊息，緊接在人格之後 ⇒ 每輪吃得到快取；
    #            區塊只留標記與會變的那幾行。⚠️ 說明離區塊變遠，「記憶≠對方現在說的話」的效果要重測。
    # ⚠️ 兩組對照施測必須同值（它改變 prompt 結構）。
    "note_position": "fixed",
    # 區塊放哪（2026-10-23 邪狼裁定改 "in_user"；改回 "tail_system" 即完全回到舊行為）：
    #   "tail_system"＝一則 system 訊息，夾在上一句回覆與本輪輸入之間（舊做法）。
    #   "in_user"    ＝併進本輪 user 訊息的開頭（區塊＋換行＋輸入）。
    # 依據（2026-10-23 實測，記憶開場、llama-server，V3 人格）：tail_system 時區塊一出現在上一句回覆後面，
    #    她就傾向把上一句原樣再講一次（日常取樣 4/5 場、固定 seed 3/3、貪婪解碼每場都有）；
    #    in_user 與「放在對話開頭」各 0 次。in_user 的快取成本與舊做法相近，放開頭則開場消失時整段歷史重算。
    # ⚠️ 兩組對照施測必須同值（它改變 prompt 結構）。
    "block_position": "in_user",
    # {open}／{close}／{remember}／{recall} 由上面四個鍵代入（標記與標題只存一份）。⚠️ 用詞是 Claude 擬的草稿，可直接改；
    #    稱呼用「妳」跟人格 prompt 一致（舊 block_note 用「你」，inline 模式照舊不動）。
    "fixed_note": "對話裡用 {open} 和 {close} 包起來的段落，是妳的記憶，不是對方現在說的話："
                  "「{remember}」是妳本來就記得的，「{recall}」是現在的話題讓妳想起的。"
                  "和現在的話題有關才用，用自己的話帶出來，不要照抄原句。"
                  "「{recall}」每一條前面括號裡的，是那件事發生的時候。",

    # ── 開場（程式算、不經 LLM）：上次聊天多久以前／今天聊過幾次／那次幾輪＋上一場關鍵詞 ──
    "opening_turns": 3,       # 本場前幾輪帶開場；之後不帶（避免每輪同一段字被照唸）

    # ── 事件層（程式切段、程式挑關鍵詞與代表句，不經 LLM；代表句照存、不注入）──
    "episode_gap_minutes": 30,   # 兩句相隔超過這麼久就切成新的一段
    "episode_keywords": 6,       # 每段挑幾個關鍵詞
    "episode_reps": 2,           # 每段挑幾句代表句（只挑使用者的話；存進事件層，目前不注入）

    # ── 主題向量的 embedding 模型 ──
    # 2026-09-27 Windows 實測：import 14.5s＋載入 3.6s（只在開程式時、背景進行）、
    # 單句 33 ms、常駐約 800 MB；掛著時 LLM 總耗時差約 1.7%（prompt 743 token）。
    # ⚠️ 對話變長時記憶體壓力會變，屆時要重量。
    "embed_enabled": True,
    "embed_model": "paraphrase-multilingual-MiniLM-L12-v2",

    # ══ 準 v4（2026-09-30，《現況》〈🔒 隔離區〉記憶模塊〈準 v4〉）══════════════
    # 寫入端：原句全存（同準 v3）＋背景判斷「這段有沒有值得記的事」→ 有就建主題（transcript 的索引）。
    # 讀取端：主題兩路檢索 → 注入最新有效版本的主題句（重寫＝第 4 步，尚未實作）。
    # 🔺 2026-10-14 關閉（《現況》第八章 P1 第 ② 點）：換判斷模型之前不建主題——v3 判斷不可靠，
    #    清零後的記憶會再長出垃圾主題。關閉期間原句照存、開場照常；向量模型不載（只有主題用得到）；
    #    獨立終端的背景判斷（start/背景判斷.bat）也不啟動。重開後，關閉期間存下的原句會被補判。
    "v4_enabled": False,         # 關掉＝只存原句、不判不撈

    # 說話者 ID（寫進 transcript.speaker_id）。觀眾之後用平台 ID，不在這裡列。
    # ⚠️ 顯示名只影響送給背景模型與注入的文字，改了不需遷移。
    "speaker_ids": {"user": "evilwolf", "assistant": "aofos"},
    "speaker_names": {"evilwolf": "邪狼", "aofos": "奧狐"},

    # 「什麼值得記」：類型、判準、例子全部在這個檔（2026-10-09 邪狼定，直接排版改這個檔即可，不用碰程式）。
    #    類型名稱的唯一來源是檔內【類型】段；模型回了清單外的類型 ⇒ 照存並記 WARNING。
    #    ⚠️ 例子越多，每段背景判斷的 prefill 越長（規則在前、連續判好幾段時吃得到快取）。
    "judge_prompt_path": os.path.join(PROJECT_ROOT, "prompts", "memory_judge.txt"),

    # 背景判斷（乙：同一顆對話模型，閒置時跑、一開口就取消）
    "judge_in_test_mode": False,  # 施測時不跑：日常區在施測中必須靜止，且背景呼叫會搶推論槽
    # 最後一次輪次收尾後閒置多久才開始背景判斷。🔺 2026-10-09 8 → 60（邪狼）：最近 20 輪在
    #    Working Memory 裡、同場本來就記得，主題晚點建不影響；拉長＝少被開口取消、少搶推理槽。
    "judge_idle_s": 60.0,
    "judge_segment_max": 12,      # 一段最多幾句（原句數，雙方合計）
    "judge_segment_gap_minutes": 30,  # 兩句相隔超過這麼久就切段（同事件層）
    "judge_segment_close_s": 60,  # 最後一句距今超過這麼久，這段才算「講完了」可以判
    "judge_max_topics": 5,        # 一段最多建幾件事
    "judge_max_tokens": 220,      # 一次呼叫：第一行判斷＋最多 judge_max_topics 行主題句
    # 建不建怎麼定（judge.py decide）：
    #   None ＝看模型第一行的字（「建」「不確定」、看不懂都當建；只有「不建」不建）。P(建) 照算、照寫 log。
    #   數值 ＝看機率：P(建)＝第一行是「建」或「不確定」的機率（logprobs 算，judge.py p_build）≥ 這個值就建；
    #          模型答「不建」但過門檻 ⇒ 補一次呼叫要它寫主題句。低於 0.5＝寧可多建。
    # 🔺 2026-10-24 邪狼裁：0.3 → None。依據：llama-server 16 段實測（貪婪，3 次），門檻只改變了
    #    「外星人派來的」一段（P≈0.24～0.35，應不建）的結果，而且是改錯；沒有一段靠門檻救回漏建。
    #    等實跑出現「答不建、其實該建」的樣本再考慮恢復數值（log 裡的 P(建) 就是校準資料）。
    #    舊值 0.3 的由來：2026-09-30「寧可多建」；2026-10-09 探測「我是你哥也是你的創造者」P(建)≈0.37。
    "judge_build_threshold": None,
    "judge_top_logprobs": 5,      # 每個 token 要幾個候選（算 P(建) 只看前兩個 token）
    # 背景一律貪婪（可重現）。🔴 取樣參數全部明帶、不靠後端預設值（沒帶的會套 server 預設或
    #    GGUF 內嵌值，換模型就靜默改變背景行為）。重複懲罰在貪婪下仍會改變選字（主題句要照抄人名）⇒ 1.0。
    #    ⭐ 單一來源在這裡（對話用的 LLM_CONFIG 同樣全部明帶）。
    "judge_params": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                     "repeat_penalty": 1.0},
    "judge_backlog_warn": 200,    # 待判原句超過這麼多句 ⇒ WARNING（背景跟不上或死掉）
    "judge_pause_timeout_s": 300, # 暫停超過這麼久沒被恢復 ⇒ 自動恢復並 WARNING（收尾掛點漏呼叫的保險）

    # 讀取：向量＋關鍵字兩路一律並行，各取自己的配額、聯集去重、不重排（2026-10-09 邪狼裁）。
    #    注入最多 topic_vec_k＋topic_bm25_k 件（兩路撈到同一件時只算一件、空位不遞補）。
    #    每多一件約多 20～30 token prefill。三個值都是 🟨 起始值，待量測校準。
    "topic_vec_k": 3,
    "topic_bm25_k": 1,
    "topic_min_sim": 0.45,        # 向量那一路的下限（主題句短、語意集中，先比 v1 RAG 原句的 0.55 低）
}

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

    # 情境怎麼送（2026-10-23 邪狼裁 P8 改選項 D）：
    #   "in_user"       ＝併進本輪 user 訊息（「[目前情境]\n描述」；有記憶區塊時接在區塊後面）。
    #   "separate_role" ＝舊做法：一則 observation_role 訊息放在空 user 之前。改回即完全回到舊行為。
    # 依據（2026-10-23 實測，《現況》22之84）：separate_role（"tool"）在模板裡被渲染成 ipython header、內容包成
    #    JSON 字串（引號與 \n 都是字面字元），扣住情境最差，貪婪解碼連續觸發會回同一句；
    #    in_user 扣住情境最好、照抄 0 次，邪狼說話的輪次不受影響。⛔ 不可改成尾端 system（24/24 照講上一句）。
    # ⚠️ in_user 時 user_placeholder 不送給模型，只用在寫回 Working Memory 與「observation 輪不做檢索」。
    # ⚠️ 已知限制：一對一才成立；聊天室恢復後 user 位置多人共用，到時重看（邪狼 2026-10-23）。
    "observation_delivery": "in_user",

    # 情境要不要留在之後幾輪的對話歷史裡（方案 α，2026-10-24 邪狼裁走 α、選 all；《現況》第八章 P20）：
    #   "none"   ＝方案 β（舊做法）：情境只存在記憶 db 的欄位，之後幾輪模型看不到（歷史裡是空 user）。
    #   "latest" ＝只留最近一則：更早的收回成空 user（落選）。
    #   "all"    ＝每則 observation 都顯示情境。
    # 依據（2026-10-24 瀏覽器直打，各 6 場日常＋1 場貪婪，《現況》22之84 六）：all 與 latest 說話輪插話都是 0/12、
    #    「現在在幹嘛」都 6/6；all「剛剛在玩什麼」完整答出 4 vs 2；all 的 observation 輪只重算 38～53 token，
    #    latest 因為要把舊那則收回、快取斷掉，要 101～161。時間感標記不加（她會開始講時間）。
    # ⚠️ all 的代價：情境文字留在 WM（每則約 30～40 token，受 WM 輪數上限約束）；偶爾把舊情境當成現在（6 場 1 次）。
    # ⚠️ 只在 observation_delivery="in_user" 時有效（歷史要和當時送出的一致），否則啟動時大聲失敗。
    "situation_in_history": "all",

    # 情境送給模型的樣子（observation 輪與歷史共用這一份）。
    "situation_template": "[目前情境]\n{description}",

    # 情境訊息的 role 名稱（只在 observation_delivery="separate_role" 時用）。
    #
    # ⚠️ 兩套 role 定義交錯，交集裡沒有兩邊都正確的選項：
    #      "ipython" 過不了 LM Studio 白名單（400，實測）
    #      "tool"    非 Llama-3.1 合法 role，且 v4 的 template 把它原樣插進 header
    #                （渲染出 <|start_header_id|>tool，模型沒見過）
    #    v3 曾有 tool → ipython 映射分支，v4 沒有——依 v3 時代的推理做決定會出錯。
    # ⚠️ 讀完《現況》第八章 P8 之前不要改這個值。
    # ⚠️ 上述兩條實測都是 LM Studio 時期的。換 llama-server 後（2026-10-11，tools/chat_probe）：
    #    "tool" 不被拒（沒有 400、有正常回應）。2026-10-23 /apply-template 實查：v3 模板把 "tool" 與 "ipython"
    #    都渲染成 ipython header、內容包成 JSON 字串，兩者輸出逐字相同（選項 A 等於現值）。
    "observation_role": "tool",

    # 註："test_observations" 腳手架已移除（《現況》第八章 舊C4（已結案））。日後若再需要手動注入情境，
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
# ⚠️ 這裡只是旗標，實際分支在兩處（2026-09-26 起）：
#      core/memory/manager.py  → True 時寫入改走 MEMORY_CONFIG["test_db_path"]
#                                （讀取共用、寫入分流，第八章 舊C8（已結案））
#      core/llm_client.py      → True 時套用 llm_overrides

TEST_MODE = False

TEST_CONFIG = {
    # 施測期間強制關閉串流。
    # ⚠️ 「串流與非串流在貪婪解碼下逐字相同」尚未驗證，施測不賭這個假設。
    # 驗證方式（做完即可刪本項）：同一題以 stream=True / False 各跑一次，
    #    貪婪解碼下應逐字相同。
    "disable_stream": True,

    # 評測專用：貪婪解碼。temperature=0 + top_k=1 直接取機率最高的 token，
    # 隨機性從源頭消失，不依賴 seed（日常 seed 為 None，見 LLM_CONFIG["seed"]）。
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
    # ⚠️ 保留理由已於 2026-08-28 從假設轉為排程（《現況》第八章 C13）：2026-08-26
    #    三組 36 題實測在 opencc 關閉下仍抓到簡體字，恢復已成定局、只剩何時
    #    （2026-09-03 裁定等換設備後再議），而恢復時本項是唯一能看見轉換內容的
    #    工具。舊目的「釐清異體字來源」已於 2026-08-06 達成且證偽。
    # ⚠️ 施測期間不要開——它是新變數，兩組對照必須同值。
    "log_opencc_changes": True,
}


PERSONA_CONFIG = {
    "name": "奧狐",
    "language": "繁體中文",
}

