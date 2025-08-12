# services/intent_classifier.py
from __future__ import annotations
from typing import Dict

class Intent:
    LEARN = "learn"             # 새로운 지식 학습
    QUESTION = "question"       # 일반 질의응답
    IMAGE_ANALYZE = "image_analyze"
    SEARCH = "search"           # 웹 검색 필요
    CORRECTION = "correction"   # 정정/수정
    OTHER = "other"

class IntentClassifier:
    """
    매우 경량의 규칙 기반 분류기. 외부 의존성 없음.
    """
    def classify(self, text: str, has_image: bool = False) -> str:
        t = (text or "").strip().lower()
        if has_image:
            return Intent.IMAGE_ANALYZE
        if any(k in t for k in ["정정", "수정", "틀렸", "오류", "정답은"]):
            return Intent.CORRECTION
        if any(k in t for k in ["검색", "최신", "뉴스", "가격", "스펙 업데이트"]):
            return Intent.SEARCH
        if any(k in t for k in ["학습", "외워", "기억해", "추가해"]):
            return Intent.LEARN
        if t.endswith("?") or any(k in t for k in ["왜", "어떻게", "무엇", "언제", "어디"]):
            return Intent.QUESTION
        return Intent.OTHER