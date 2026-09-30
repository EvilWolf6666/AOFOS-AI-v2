# core/memory/fake_llm_server.py
"""
自測用的假 OpenAI 相容伺服器（只給 selftest_v4 用，不接任何真服務）。

模擬 LM Studio 的兩段延遲：送出回應標頭前的 prefill，以及逐 token 串流。
記錄每個請求是「送完」還是「被客戶端中途斷線」，用來驗證 BackgroundLLM 的取消。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self) -> None:
        self.prefill_s = 0.0          # 回應標頭前的延遲（模擬 prefill）
        self.token_s = 0.0            # 每個 chunk 之間的延遲
        self.responder = lambda req: "OK"   # req(dict) → 回應文字
        self.log: list[dict] = []     # 每個請求：{purpose, status: done/aborted, body}
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base_url = f"http://127.0.0.1:{self._srv.server_address[1]}/v1"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 安靜
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                entry = {"body": req, "status": "started"}
                fake.log.append(entry)
                time.sleep(fake.prefill_s)
                text = fake.responder(req)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i, ch in enumerate(_chunks(text)):
                        if i:
                            time.sleep(fake.token_s)
                        data = {"id": "x", "object": "chat.completion.chunk", "created": 0,
                                "model": req.get("model", "m"),
                                "choices": [{"index": 0, "delta": {"content": ch},
                                             "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    entry["status"] = "done"
                except (BrokenPipeError, ConnectionResetError, OSError):
                    entry["status"] = "aborted"

        return H


def _chunks(text: str, size: int = 2) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]
