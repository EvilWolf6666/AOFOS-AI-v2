# core/memory_manager.py
# 記憶系統：Working Memory（近期對話）+ Compressed Memory（摘要）+ RAG（長期知識）

import json
import os
from datetime import datetime

from config import (
    COMPRESS_BATCH_SIZE,
    MAX_SUMMARY_SENTENCES,
    MAX_WORKING_MEMORY,
    TEST_CONFIG,
    TEST_MODE,
)
from core.llm_client import LLMClient
from core.logger import setup_logger, timed
from core.rag_manager import RAGManager

logger = setup_logger("MemoryManager")

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
NORMAL_MEMORY_PATH = os.path.join(_PROJECT_ROOT, "data", "memory.json")
TEST_MEMORY_PATH = os.path.join(_PROJECT_ROOT, TEST_CONFIG["memory_path"])


class MemoryManager:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.working_memory: list[dict] = []
        self.compressed_memory: str = ""
        self.rag = RAGManager()

        # 測試模式改用獨立記憶檔，避免施測污染正式記憶。
        # 起始值取自 config，執行期可由 /test 指令切換（見 set_test_mode）。
        self.test_mode = TEST_MODE
        self.memory_path = TEST_MEMORY_PATH if TEST_MODE else NORMAL_MEMORY_PATH
        if TEST_MODE:
            logger.warning(f"TEST_MODE 啟用：記憶檔改用 {self.memory_path}")

        self._load_memory()

    def set_test_mode(self, on: bool) -> None:
        """
        執行期切換測試模式：存回目前記憶 → 切換路徑與 RAG 開關 →
        清空 Working Memory 並從新路徑重新載入。

        Working Memory 必須清空（切換的目的就是隔離，殘留對話會污染下一次施測的
        context），這是對呼叫端可見的副作用，指令回應中會明講。

        進入測試模式時一併清掉上一次的測試記憶檔：施測協定為「每題重開程式 +
        /test on」，測試記憶若跨題殘留，第 N 題會被前 N-1 題壓縮出的摘要污染。
        自動清空以避免依賴人工每題手動刪檔（12 題 × 2 組共 24 次，漏一次整批作廢）。

        本切換刻意不寫回 config：不持久化才不會有「測完忘記關」的風險。
        """
        if on == self.test_mode:
            return

        self.save_memory()  # 先把現況存回切換前的檔案

        self.test_mode = on
        self.memory_path = TEST_MEMORY_PATH if on else NORMAL_MEMORY_PATH
        self.rag.set_enabled(not (on and TEST_CONFIG.get("disable_rag", True)))

        if on and os.path.exists(TEST_MEMORY_PATH):
            try:
                os.remove(TEST_MEMORY_PATH)
                logger.info("已清除上一次的測試記憶檔")
            except Exception as e:
                logger.warning(f"清除測試記憶檔失敗，本次施測可能受前次殘留污染：{e}")

        self.working_memory.clear()
        self.compressed_memory = ""
        self._load_memory()

        logger.warning(
            f"測試模式{'開啟' if on else '關閉'}：記憶檔切換為 {self.memory_path}，"
            f"Working Memory 已清空"
        )

    # ── 讀取 / 儲存 ──────────────────────────────

    def _load_memory(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.compressed_memory = data.get("compressed_memory", "")
            logger.info("記憶載入完成。")
        else:
            logger.info("無歷史記憶，從零開始。")

    def save_memory(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump({"compressed_memory": self.compressed_memory}, f, ensure_ascii=False, indent=2)
        logger.debug("記憶已存檔。")

    def _backup_memory(self, reason: str) -> str | None:
        """
        在即將以不可信內容覆寫 compressed_memory 前，先另存一份現有內容。

        存在理由：_parse_compress_result() 解析失敗時會直接用 LLM 原始輸出當摘要，
        等於用一段格式不明的文字覆蓋掉正常的長期記憶，且原本不可逆。
        """
        if not self.compressed_memory:
            return None
        try:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(
                os.path.dirname(self.memory_path), f"memory_backup_{ts}_{reason}.json"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"compressed_memory": self.compressed_memory},
                    f, ensure_ascii=False, indent=2
                )
            return path
        except Exception as e:
            logger.warning(f"記憶備份失敗（不影響主流程）：{e}")
            return None

    # ── 對話管理 ─────────────────────────────────

    def add_turn(self, user_input: str, assistant_reply: str):
        """只追加進 Working Memory，不觸發壓縮（壓縮時機交給呼叫端，見 maybe_compress）。"""
        self.working_memory.append({"role": "user", "content": user_input})
        self.working_memory.append({"role": "assistant", "content": assistant_reply})

    def needs_compress(self) -> bool:
        """僅回報是否將要壓縮，不執行。供呼叫端在壓縮前先給使用者提示。"""
        return len(self.working_memory) > MAX_WORKING_MEMORY * 2

    def maybe_compress(self) -> bool:
        """
        視 Working Memory 長度決定是否壓縮，回傳是否真的執行了壓縮。

        ⚠️ 應在回應已經輸出之後呼叫，避免壓縮耗時落在使用者等待的路徑上。
        目前仍是同步執行，尖峰只是被移到「下一次輸入之前」，並未消失。
        若改為背景執行，需一併處理併發寫入 compressed_memory 的風險
        （本欄位是整份覆寫而非追加，競態的後果是記憶被寫壞且無徵兆）。
        """
        if self.needs_compress():
            self._compress_oldest()
            return True
        return False

    def get_context(self, user_input: str = "") -> list[dict]:
        """組裝 context：Compressed Memory → Working Memory → RAG 查詢結果。

        ⚠️ 順序依「變動頻率由低到高」排列，不可任意調換：prompt cache 比對的是
        最長共同前綴，任何一段內容變動，其後全部要重新 prefill。RAG 每輪依
        user_input 重查、內容每輪都不同，排在 Working Memory 之前會使 WM 永遠
        落在斷點之後、永遠無法重用。同一原理見 pipeline.handle_observation()。
        """
        with timed("記憶組裝"):
            context = []

            if self.compressed_memory:
                context.append({
                    "role": "system",
                    "content": f"[過去對話摘要]\n{self.compressed_memory}"
                })

            context.extend(self.working_memory)

            if user_input:
                rag_results = self.rag.query(user_input)
                if rag_results:
                    rag_text = "\n".join(f"- {r}" for r in rag_results)
                    context.append({
                        "role": "system",
                        "content": f"[相關記憶]\n{rag_text}"
                    })
                    logger.debug(f"注入 {len(rag_results)} 筆 RAG 記憶。")
                else:
                    logger.debug("無相關 RAG 記憶，跳過注入。")

        return context

    # ── 記憶壓縮 ─────────────────────────────────

    def _compress(self, turns: list[dict]):
        """
        將指定對話輪次與既有摘要整合覆寫（非疊加），長度受 MAX_SUMMARY_SENTENCES 限制。

        舊摘要中沒出現在新摘要裡的內容一律轉存 RAG，不區分原因（被取代或排不進
        句數上限）——區分只增加 LLM 誤判風險，兩者最終都存進同一個地方。

        prompt 的三條規則各自對應一個實測踩過的失敗模式，改動前請先看
        《歸檔》D0001：
          規則一 未提及內容預設保留 → 防資訊無聲消失
          規則二 禁止逐字複製對話原文 → 防記憶變成自我強化的複述錨點
          規則三 禁止推測填充 → 防取捨順序被讀成必填欄位模板

        內部 LLM 呼叫以 purpose="compress" 標記，才能與回應使用者的對話呼叫
        在計時紀錄中區分開來。
        """
        with timed("記憶壓縮(整體)"):
            dialogue_text = "\n".join(
                f"{'邪狼' if m['role'] == 'user' else '奧狐'}：{m['content']}"
                for m in turns
            )

            existing_block = f"[之前的摘要]\n{self.compressed_memory}\n\n" if self.compressed_memory else ""

            prompt = [{
                "role": "user",
                "content": (
                    "你要幫忙整理「奧狐」的近期記憶摘要。\n\n"
                    f"{existing_block}[新對話]\n{dialogue_text}\n\n"
                    "請完成兩件事，並嚴格遵守以下規則：\n"
                    "規則一：「之前的摘要」中的每一條資訊，預設都要「保留」在新摘要裡，"
                    "除非新對話明確更新、推翻、或讓它過時（例如某件事後來解決了、狀態變了），"
                    "或受限於新摘要的句數上限而排不進去。"
                    "新對話沒有提到的舊資訊，不代表它不重要，原則上不可以省略，"
                    "只有在真的擠不進新摘要時才可以不保留。\n"
                    "規則二：新摘要必須是改寫過的客觀概括描述，即使某句對話內容很簡短，"
                    "也一樣要改寫成第三人稱敘述句，不可以逐字複製對話原文"
                    "（不論是之前摘要裡的句子，還是新對話裡奧狐或邪狼說的原話，"
                    "都不可以整句照抄）。\n"
                    "規則三：只能寫「之前的摘要」或「新對話」裡實際出現過的內容，"
                    "不可以補充、推測或為了湊長度而填充。"
                    "如果某類資訊完全沒有出現過（例如沒談到關係、沒做出決定），"
                    "就完全不要提到那類資訊——"
                    "不要寫「關係沒有改變」「最近沒有新的決定」這種句子，"
                    "沒有資訊時正確的做法是「不寫」，不是「寫沒有變化」。"
                    "新摘要的長度取決於實際有多少內容，只有一兩句也完全可以。\n\n"
                    f"1. 輸出整合後的新摘要，不超過 {MAX_SUMMARY_SENTENCES} 句話，"
                    "包含：未被取代的舊資訊（照規則一保留）+ 新對話中重要的新資訊，"
                    "全部以改寫過的概括句呈現（依規則二）。"
                    "句數上限是上限不是目標，有多少寫多少。"
                    "只有在內容確實超過上限、必須取捨時，才依此順序保留："
                    "人名 > 明確做出的決定 > 近期狀態 > 其他；"
                    "這是取捨順序，不是必填欄位清單。"
                    "每一句話結束後換行，一行一句，不要把多句擠在同一行。\n"
                    "2. 列出「之前的摘要」中，沒有出現在新摘要裡的所有舊資訊"
                    "（無論原因是被新對話取代、還是因為句數上限排不進去，只要新摘要裡沒有，就列出來；"
                    "若新摘要完整保留了之前摘要的所有內容，就寫「無」）。\n\n"
                    "請只用以下格式輸出，不要其他說明文字：\n"
                    "摘要：<新摘要內容>\n"
                    "淘汰：<沒出現在新摘要裡的舊資訊，或「無」>"
                )
            }]

            result = self.llm.chat(prompt, purpose="compress")

            # ⚠️ llm_client.chat() 在任何例外時回傳 ""（刻意設計，見該檔）。
            # 空字串會讓 _parse_compress_result 的 next() 拋 StopIteration、
            # 落入保底分支並回傳 ("", "無")，接著把 compressed_memory 覆寫成
            # 空字串並寫回磁碟——長期記憶就此消失，而使用者只會看到
            # finalize_turn 印的一行「壓縮失敗」，不會知道記憶被清空了。
            # 故必須在覆寫之前擋掉，且回傳 None 讓呼叫端知道本次沒有結果。
            if not result.strip():
                logger.error("壓縮呼叫回傳空內容，本次壓縮放棄（記憶維持原狀）")
                return None

            new_summary, discarded = self._parse_compress_result(result)

            self.compressed_memory = new_summary
            self.save_memory()

            # 明確記錄淘汰結果：「RAG 沒有新增資料」有兩種完全不同的原因
            # （本來就沒東西被淘汰 / 有東西被淘汰但寫入失敗），沒有這行 log 無法區分
            if discarded and discarded != "無":
                logger.info(f"本次壓縮淘汰內容（將轉存 RAG）：{discarded[:80]}")
                try:
                    self.rag.add(discarded)
                    logger.debug(f"舊記憶已轉存 RAG：{discarded[:50]}...")
                except Exception as e:
                    logger.warning(f"舊記憶轉存 RAG 失敗（不影響主流程）：{e}")
            else:
                logger.info("本次壓縮無淘汰內容，RAG 不會新增資料（此為正常情況之一）")

        return new_summary

    def _parse_compress_result(self, result: str) -> tuple[str, str]:
        """解析 LLM 回傳的『摘要 / 淘汰』結果。"""
        try:
            lines = result.strip().splitlines()
            summary_line_idx = next(
                i for i, l in enumerate(lines) if l.strip().startswith("摘要：")
            )
            discard_line_idx = next(
                (i for i, l in enumerate(lines) if l.strip().startswith("淘汰：")),
                None,
            )

            if discard_line_idx is not None:
                summary = "\n".join(lines[summary_line_idx:discard_line_idx]).replace("摘要：", "", 1).strip()
                discarded = "\n".join(lines[discard_line_idx:]).replace("淘汰：", "", 1).strip()
            else:
                summary = "\n".join(lines[summary_line_idx:]).replace("摘要：", "", 1).strip()
                discarded = "無"

            return summary, discarded
        except Exception as e:
            # 保底：解析失敗時用原始輸出當摘要（避免記憶遺失），不寫入 RAG
            # （避免存入格式錯亂內容）。覆寫前先備份，讓這一步變成可回復的。
            backup_path = self._backup_memory("parse_fail")
            logger.error(
                f"壓縮結果解析失敗：{e}。將以原始輸出覆寫記憶"
                + (f"，覆寫前已備份至 {backup_path}" if backup_path else "（無既有記憶可備份）")
                + f"。原始結果：{result}"
            )
            return result.strip(), "無"

    def _compress_oldest(self):
        """
        ⚠️ 「先壓縮、後裁切」的順序不可對調。先裁切的話，壓縮失敗時那批對話
        已經從 working_memory 移除、而摘要也沒產生——兩邊都不在了，且沒有
        任何徵兆（使用者只看到一行「壓縮失敗」）。

        壓縮失敗時 working_memory 會持續超長，下一次 finalize_turn 會再試一次。
        這是正確的：重試比靜默清空好，而 LLM 連續失敗代表 LM Studio 掛了，
        那本來就是更該被看見的事。
        """
        to_compress = self.working_memory[:COMPRESS_BATCH_SIZE]
        summary = self._compress(to_compress)

        if summary is None:
            logger.error(
                "壓縮未產生有效摘要，Working Memory 不裁切、既有摘要不覆寫"
                "（下次觸發時會重試）"
            )
            return

        self.working_memory = self.working_memory[COMPRESS_BATCH_SIZE:]
        logger.info(f"已壓縮記憶：{summary[:50]}...")

    def compress_on_exit(self):
        if not self.working_memory:
            return

        # 測試模式下不壓縮：評測不需要摘要，而壓縮是一次完整的 LLM 呼叫
        # （實測 17～25 秒）；且壓縮結果會寫進測試記憶檔，成為下一題的污染源。
        if self.test_mode:
            logger.info("測試模式：跳過離開時的記憶壓縮。")
            return

        if self._compress(self.working_memory) is None:
            # ⚠️ 這裡和 _compress_oldest 不同：程式即將結束，沒有「下次重試」。
            # working_memory 只存在記憶體中，此刻壓縮失敗就等於本次對話
            # 完全不會進入長期記憶。必須大聲說明後果，否則使用者會以為
            # 只是「摘要沒更新」。
            logger.error(
                f"離場壓縮失敗，本次對話的 {len(self.working_memory)} 則訊息"
                "不會寫入長期記憶（僅存在記憶體，程式結束後消失）。"
                "既有摘要未被覆寫。"
            )
            return

        logger.info("對話已儲存。")