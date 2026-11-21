# core/run_snapshot.py
"""
一次執行「實際用了哪組參數」的快照（給施測、探測，之後也給日常對話）。

為什麼有這支
  施測前六步要求「人格兩組相同、參數全帶入」，但沒有快照時事後驗不了。
  chat_probe 原本自己組一份；eval_runner 沒有。放在這裡讓兩邊（與之後的日常對話）共用同一份，
  不各寫各的（單一來源）。

取值原則
  · 能從執行中的物件取「實際值」就不取 config：例如測試模式會把取樣參數換成貪婪解碼，
    config 寫的是日常值，記 config 會記錯（effective 一欄）。
  · 人格只記雜湊與第一行，不記全文（全文在 prompts/，雜湊足以判斷兩組是否相同）。
  · 不記 api_key。
"""
from __future__ import annotations

import hashlib
import json

import config

_MEM_KEYS = ("block_position", "note_position", "opening_turns", "v4_enabled", "episode_gap_minutes")
_EVT_KEYS = ("observation_delivery", "situation_in_history", "situation_template")


def snapshot(persona, llm) -> dict:
    """persona：core.persona.Persona；llm：core.llm_client.LLMClient（都要是本次實際在用的那個實例）。"""
    sp = persona.system_prompt or ""
    return {
        "persona_sha256": hashlib.sha256(sp.encode("utf-8")).hexdigest(),
        "persona_first_line": sp.splitlines()[0] if sp else "",
        "persona_sent": getattr(persona, "send_system", True),
        "llama_server": {k: config.LLAMA_SERVER.get(k) for k in ("build", "model", "alias")},
        "llm": {k: v for k, v in config.LLM_CONFIG.items() if k != "api_key"},
        "seed_in_client": llm.seed,
        # 本次請求實際送出的取樣參數（測試模式時已換成 TEST_CONFIG 的值）
        "effective": {
            "test_mode": llm.test_mode,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "max_tokens": llm.max_tokens,
            "frequency_penalty": llm.frequency_penalty,
            "presence_penalty": llm.presence_penalty,
            "extra_body": dict(llm.extra_body),
            "stream": llm.stream,
        },
        "memory": {k: config.MEMORY_CONFIG.get(k) for k in _MEM_KEYS},
        "event": {k: config.EVENT_CONFIG.get(k) for k in _EVT_KEYS},
        "wm": {"MAX_WORKING_MEMORY": config.MAX_WORKING_MEMORY, "WM_DROP_ROUNDS": config.WM_DROP_ROUNDS},
    }


def digest(snap: dict, *, drop: tuple[str, ...] = ()) -> str:
    """快照的雜湊（鍵排序後算，與 dict 順序無關）。drop：比對時刻意不比的頂層鍵（例：跨模型比較時的 llama_server）。"""
    body = {k: v for k, v in snap.items() if k not in drop}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
