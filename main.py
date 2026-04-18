# main.py

from core.llm_client import LLMClient
from core.persona import Persona
from core.memory_manager import MemoryManager


def main():
    llm = LLMClient()
    persona = Persona()
    memory = MemoryManager(llm)

    print("=== 奧狐 AOFOS 啟動 ===")
    if memory.compressed_memory:
        print("[記憶已載入]")
    else:
        print("[首次啟動，無歷史記憶]")
    print("指令：/remember <內容> 手動存入知識，/list 查看已存知識，quit 離開\n")

    while True:
        user_input = input("你：").strip()

        # 離開
        if user_input.lower() == "quit":
            memory.compress_on_exit()
            print("掰掰。")
            break

        if not user_input:
            continue

        # 手動存入 RAG 知識
        if user_input.startswith("/remember "):
            content = user_input[10:].strip()
            memory.rag.add(content)
            print(f"[已記住：{content}]\n")
            continue

        # 列出所有 RAG 知識
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

        # 正常對話
        context = memory.get_context(user_input)
        messages = persona.build_messages(context, user_input)
        reply = llm.chat(messages)

        if reply:
            print(f"\n奧狐：{reply}\n")
            memory.add_turn(user_input, reply)


if __name__ == "__main__":
    main()