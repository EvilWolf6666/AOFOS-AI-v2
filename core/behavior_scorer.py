# core/behavior_scorer.py
# Behavior Scoring — 檢查奧狐的回應有沒有偏離設定

from core.llm_client import LLMClient
from core.logger import setup_logger

logger = setup_logger("BehaviorScorer")

# 評分標準描述，給 LLM 當參考
SCORING_PROMPT = """你是一個評審，負責檢查「奧狐」的回應是否符合她的人格設定。

奧狐的設定：
- 說話自然口語，不客套
- 不假裝記得沒有的事
- 不自稱AI助手
- 不用 *動作描述* emote
- 不過度正式或長篇大論
- 個性直接，偶爾嗆人但不惡意

請根據以下回應，給出：
1. 分數：0~100（100分代表完全符合設定）
2. 問題：如果有偏離，簡短說明是什麼問題（沒問題就說「無」）

只輸出以下格式，不要其他說明：
分數：XX
問題：XXXX
"""

SCORE_THRESHOLD = 60  # 低於這個分數才介入修正


class BehaviorScorer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, reply: str) -> tuple[int, str]:
        """
        評分奧狐的回應。
        回傳 (分數, 問題描述)
        """
        messages = [
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": f"奧狐的回應：{reply}"},
        ]
        result = self.llm.chat(messages)
        return self._parse(result)

    def _parse(self, result: str) -> tuple[int, str]:
        """解析 LLM 回傳的評分結果"""
        try:
            lines = result.strip().splitlines()
            score_line = next((l for l in lines if l.startswith("分數：")), "分數：100")
            issue_line = next((l for l in lines if l.startswith("問題：")), "問題：無")
            score = int(score_line.replace("分數：", "").strip())
            issue = issue_line.replace("問題：", "").strip()
            return score, issue
        except Exception as e:
            logger.warning(f"評分解析失敗：{e}，原始結果：{result}")
            return 100, "無"

    def check(self, reply: str) -> bool:
        """
        回傳 True 代表通過，False 代表有問題需要注意。
        低於門檻才記錄警告。
        """
        score, issue = self.score(reply)
        if score < SCORE_THRESHOLD:
            logger.warning(f"行為評分偏低（{score}分）：{issue}")
            return False
        else:
            logger.debug(f"行為評分正常（{score}分）")
            return True