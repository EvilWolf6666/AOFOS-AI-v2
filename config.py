# config.py
# 統一設定檔 — 之後換模型/API 只改這裡

LLM_CONFIG = {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "local-model",
    "temperature": 0.8,
    "max_tokens": 300,
}

PERSONA_CONFIG = {
    "name": "奧狐",
    "language": "繁體中文",
}