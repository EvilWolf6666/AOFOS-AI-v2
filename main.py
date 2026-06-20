# main.py

from __future__ import annotations
from typing import Optional
from core.llm_client import LLMClient
from core.persona import Persona
from core.memory_manager import MemoryManager
from core.logger import setup_logger

logger = setup_logger("Main")


def main():
    # ── 初始化核心模組 ────────────────────────────
    llm = LLMClient()
    persona = Persona()
    memory = MemoryManager(llm)

    # ── VTube Studio 串接（可選）─────────────────
    vtube = None
    try:
        from core.vtube_client import VTubeClient
        vtube = VTubeClient()
        vtube.connect()
    except Exception as e:
        logger.warning(f"VTube 未連線，跳過：{e}")

    # ── 啟動訊息 ──────────────────────────────────
    print("=== 奧狐 AOFOS 啟動 ===")
    print("[記憶已載入]" if memory.compressed_memory else "[首次啟動，無歷史記憶]")
    print("模式：t = 文字輸入，v = 語音輸入（預設文字）")
    print("指令：/remember <內容>、/list、/mode t、/mode v、quit\n")

    from core.voice_input import VoiceInput
    voice_input: Optional[VoiceInput] = None
    mode = "t"

    # ── 主對話迴圈 ────────────────────────────────
    while True:

        # 取得輸入
        if mode == "t":
            user_input = input("你：").strip()
        else:
            user_input = voice_input.listen()  # type: ignore
            if not user_input:
                continue

        if not user_input:
            continue

        # ── 系統指令 ──────────────────────────────

        if user_input.lower() == "quit":
            memory.compress_on_exit()
            if vtube:
                vtube.disconnect()
            print("掰掰。")
            break

        if user_input == "/mode t":
            mode = "t"
            print("[切換為文字模式]\n")
            continue

        if user_input == "/mode v":
            if voice_input is None:
                voice_input = VoiceInput(model_size="small", language="zh")
            mode = "v"
            print("[切換為語音模式]\n")
            continue

        if user_input.startswith("/remember "):
            content = user_input[10:].strip()
            memory.rag.add(content)
            print(f"[已記住：{content}]\n")
            continue

        if user_input == "/list":
            items = memory.rag.list_all()
            if not items:
                print("[目前沒有任何長期記憶]\n")
            else:
                print("[長期記憶列表]")
                for item in items:
                    print(f"  [{item['id'][:8]}] {item['text']}")
                print()
            continue

        # ── 正常對話 ──────────────────────────────

        context = memory.get_context(user_input)
        messages = persona.build_messages(context, user_input)
        reply = llm.chat(messages)

        if reply:
            print(f"\n奧狐：{reply}\n")
            logger.debug(f"奧狐回應：{reply[:50]}...")
            memory.add_turn(user_input, reply)

            # VTube 嘴型
            if vtube:
                try:
                    vtube.set_mouth(0.8)
                except Exception as e:
                    logger.warning(f"VTube 嘴型控制失敗：{e}")


if __name__ == "__main__":
    main()