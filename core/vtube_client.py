# core/vtube_client.py
# VTube Studio 串接 — 控制嘴型、表情
# 內部用 asyncio，對外提供同步介面（run_until_complete 包裝）

from __future__ import annotations
from typing import Optional
import asyncio
import json
import os
import websockets
import websockets.asyncio.client   # 只給型別標註用（connect() 回傳的連線類別）
from config import PROJECT_ROOT
from core.logger import setup_logger

logger = setup_logger("VTubeClient")

VTUBE_URL = "ws://localhost:8001"
PLUGIN_NAME = "AOFOS"
PLUGIN_DEVELOPER = "EvilWolf"

# 認證 token 存檔重用。⚠️ 沒存時每次啟動都會重新要 token，VTube Studio 每次都跳「允許外掛」視窗。
#    放 data/（.gitignore 已排除）：token 等同這個外掛的通行證，不進版本控制。
TOKEN_PATH = os.path.join(PROJECT_ROOT, "data", "vtube_token.txt")
# 一般請求等回應的上限（秒）。VTS 沒回應時不卡住啟動。
RECV_TIMEOUT_S = 5.0
# 要新 token 時等使用者在 VTS 按「允許」的上限（秒）。只在第一次或 token 失效時發生。
APPROVE_TIMEOUT_S = 30.0


class VTubeClient:
    def __init__(self):
        self.ws: Optional[websockets.asyncio.client.ClientConnection] = None
        self.auth_token: Optional[str] = None
        self._loop = asyncio.new_event_loop()

    # ── 連線與認證 ────────────────────────────────

    def connect(self):
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        self.ws = await websockets.connect(VTUBE_URL)
        try:
            await self._authenticate()
        except BaseException:
            # 認證途中失敗（逾時等）：關掉已開的連線再往外拋，不留半開的 socket
            await self.ws.close()
            self.ws = None
            raise
        logger.info("連線成功。")

    async def _request(self, message_type: str, data: dict, request_id: str,
                       timeout: float = RECV_TIMEOUT_S) -> dict:
        """送一個請求並等回應；逾時拋 asyncio.TimeoutError（由 Agent 的「VTube 未連線」接住）。"""
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": message_type,
            "data": data,
        }
        await self.ws.send(json.dumps(req))
        return json.loads(await asyncio.wait_for(self.ws.recv(), timeout))

    async def _authenticate(self):
        """VTube Studio 認證為兩步：先取 token，再用 token 認證。

        先試存檔的 token；沒有或被拒才要新的（只有這一步會跳允許視窗），拿到後存檔。
        """
        plugin = {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_DEVELOPER}

        saved = self._load_token()
        if saved and await self._auth_with(saved, plugin):
            self.auth_token = saved
            logger.info("認證成功（沿用已存的 token）。")
            return

        logger.info(f"向 VTube Studio 要新的 token：請在 VTS 視窗按「允許」（最多等 {APPROVE_TIMEOUT_S:.0f} 秒）。")
        resp = await self._request("AuthenticationTokenRequest", plugin, "auth_req",
                                   timeout=APPROVE_TIMEOUT_S)
        token = resp.get("data", {}).get("authenticationToken", "")
        if not token:
            logger.warning(f"沒拿到 token（被拒絕或 VTS 回報錯誤）：{resp.get('data')}")
            return
        if await self._auth_with(token, plugin):
            self.auth_token = token
            self._save_token(token)
            logger.info("認證成功（新 token 已存檔，下次不再跳允許視窗）。")
        else:
            logger.warning("認證失敗，請在 VTube Studio 裡按允許。")

    async def _auth_with(self, token: str, plugin: dict) -> bool:
        resp = await self._request("AuthenticationRequest", {**plugin, "authenticationToken": token}, "auth")
        return bool(resp.get("data", {}).get("authenticated", False))

    @staticmethod
    def _load_token() -> str:
        try:
            with open(TOKEN_PATH, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _save_token(token: str) -> None:
        try:
            os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                f.write(token)
        except OSError as e:
            # 存不了只是下次會再跳一次允許視窗，不影響本次連線
            logger.warning(f"token 存檔失敗（下次啟動會再跳允許視窗）：{e}")

    # ── 嘴型控制 ──────────────────────────────────

    def set_mouth(self, value: float):
        """控制嘴型開合，value 範圍 0.0（閉嘴）～ 1.0（全開）。"""
        if self.ws is None:
            logger.debug("未連線，跳過嘴型控制。")
            return
        self._loop.run_until_complete(self._inject_param("MouthOpen", value))

    async def _inject_param(self, param_id: str, value: float):
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "inject",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": False,
                "mode": "set",
                "parameterValues": [
                    {"id": param_id, "value": value}
                ]
            }
        }
        await self.ws.send(json.dumps(req))
        await asyncio.wait_for(self.ws.recv(), RECV_TIMEOUT_S)

    # ── 表情控制 ──────────────────────────────────

    def play_expression(self, expression_name: str):
        """觸發指定表情。"""
        if self.ws is None:
            logger.debug("未連線，跳過表情控制。")
            return
        self._loop.run_until_complete(self._play_expression(expression_name))

    async def _play_expression(self, expression_name: str):
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "expr",
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": expression_name,
                "active": True,
            }
        }
        await self.ws.send(json.dumps(req))
        await asyncio.wait_for(self.ws.recv(), RECV_TIMEOUT_S)

    # ── 斷線 ─────────────────────────────────────

    def disconnect(self):
        if self.ws:
            self._loop.run_until_complete(self.ws.close())
        logger.info("已斷線。")