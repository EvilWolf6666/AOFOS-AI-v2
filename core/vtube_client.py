# core/vtube_client.py
# VTube Studio 串接模組 — 控制嘴型、表情
# 內部用 asyncio，對外提供同步介面（run_until_complete 包裝）

from __future__ import annotations
from typing import Optional
import asyncio
import json
import websockets
from core.logger import setup_logger

logger = setup_logger("VTubeClient")

VTUBE_URL = "ws://localhost:8001"
PLUGIN_NAME = "AOFOS"
PLUGIN_DEVELOPER = "EvilWolf"


class VTubeClient:
    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.auth_token: Optional[str] = None
        self._loop = asyncio.new_event_loop()

    # ── 連線與認證 ────────────────────────────────

    def connect(self):
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        self.ws = await websockets.connect(VTUBE_URL)
        await self._authenticate()
        logger.info("連線成功。")

    async def _authenticate(self):
        # 第一步：請求 token
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth_req",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
            }
        }
        await self.ws.send(json.dumps(req))
        resp = json.loads(await self.ws.recv())
        self.auth_token = resp["data"].get("authenticationToken", "")

        # 第二步：用 token 認證
        auth = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
                "authenticationToken": self.auth_token,
            }
        }
        await self.ws.send(json.dumps(auth))
        resp = json.loads(await self.ws.recv())
        authenticated = resp["data"].get("authenticated", False)
        if not authenticated:
            logger.warning("認證失敗，請在 VTube Studio 裡按允許。")
        else:
            logger.info("認證成功。")

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
        await self.ws.recv()

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
        await self.ws.recv()

    # ── 斷線 ─────────────────────────────────────

    def disconnect(self):
        if self.ws:
            self._loop.run_until_complete(self.ws.close())
        logger.info("已斷線。")