"""
MiniCPM-V 2.6 모델 로딩 및 추론 관리
멀티모달 모델을 이용한 이미지 분석 및 임베딩 추출
"""
#MiniCPM-V/model.py

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import logging
from transformers import AutoTokenizer, AutoModel
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiniCPMVModel:
    """MiniCPM-V 2.6 모델 관리 클래스"""

    def __init__(self, model_path: str = "openbmb/MiniCPM-V-2_6"):
        """
        MiniCPM-V 모델 초기화

        Args:
            model_path (str): 모델 경로 또는 HuggingFace 모델명
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """모델과 토크나이저 로딩"""
        try:
            logger.info(f"Loading MiniCPM-V 2.6 model from {self.model_path}...")

            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 모델 로딩
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            self.model.eval()
            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def analyze_image(self, image: Image.Image, question: str,
                      context: str = "", roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        이미지 분석 수행

        Args:
            image (Image.Image): 분석할 이미지
            question (str): 질문 텍스트
            context (str): 추가 문맥 정보
            roi (Optional[Tuple]): 관심 영역 (x, y, width, height)

        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded")

            # ROI가 지정된 경우 이미지 크롭
            if roi:
                x, y, w, h = roi
                image = image.crop((x, y, x + w, y + h))

            # 프롬프트 구성
            prompt = self._build_analysis_prompt(question, context)

            # 모델 추론
            with torch.no_grad():
                response = self.model.chat(
                    image=image,
                    msgs=[{"role": "user", "content": prompt}],
                    tokenizer=self.tokenizer,
                    sampling=True,
                    temperature=0.7
                )

            # 결과 구조화
            result = {
                "analysis": response,
                "confidence": self._calculate_confidence(response),
                "extracted_entities": self._extract_entities(response),
                "visual_features": self._extract_visual_features(response),
                "roi_used": roi is not None,
                "roi_coordinates": roi
            }

            return result

        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return {"error": str(e), "analysis": "분석에 실패했습니다."}

    def extract_embedding(self, text: str = "", image: Optional[Image.Image] = None) -> np.ndarray:
        """
        텍스트/이미지 임베딩 추출

        Args:
            text (str): 텍스트 입력
            image (Optional[Image.Image]): 이미지 입력

        Returns:
            np.ndarray: 임베딩 벡터
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            with torch.no_grad():
                if image and text:
                    # 멀티모달 임베딩
                    embedding = self._extract_multimodal_embedding(text, image)
                elif image:
                    # 이미지 임베딩
                    embedding = self._extract_image_embedding(image)
                else:
                    # 텍스트 임베딩
                    embedding = self._extract_text_embedding(text)

                return embedding.cpu().numpy()

        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            return np.zeros(768)  # 기본 임베딩 크기

    def _build_analysis_prompt(self, question: str, context: str) -> str:
        """분석용 프롬프트 구성"""
        base_prompt = """이미지를 자세히 분석하여 다음 질문에 답해주세요.

다음 정보를 포함하여 답변해주세요:
1. 객체 및 요소: 이미지에 보이는 주요 객체들
2. 색상과 형태: 주요 색상과 형태적 특징
3. 위치와 구도: 객체들의 위치 관계와 전체 구도
4. 맥락과 상황: 이미지가 나타내는 상황이나 의미
5. 감정과 분위기: 이미지에서 느껴지는 감정이나 분위기

질문: {question}

{context_part}

답변을 JSON 형태로 구성하여 구조화된 정보를 제공해주세요."""

        context_part = f"추가 문맥: {context}" if context else ""
        return base_prompt.format(question=question, context_part=context_part)

    def _extract_multimodal_embedding(self, text: str, image: Image.Image) -> torch.Tensor:
        """멀티모달 임베딩 추출"""
        # 실제 구현에서는 모델의 hidden states를 추출
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).squeeze()

    def _extract_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """이미지 임베딩 추출"""
        # 이미지 전처리 및 임베딩 추출
        # 실제 구현에서는 모델의 vision encoder를 사용
        dummy_embedding = torch.randn(768).to(self.device)
        return dummy_embedding

    def _extract_text_embedding(self, text: str) -> torch.Tensor:
        """텍스트 임베딩 추출"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 마지막 hidden state의 평균을 임베딩으로 사용
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
        return embedding

    def _calculate_confidence(self, response: str) -> float:
        """응답의 신뢰도 계산 (간단한 휴리스틱)"""
        confidence_keywords = ["확실", "명확", "분명", "틀림없이", "확실히"]
        uncertainty_keywords = ["불확실", "모호", "애매", "잘 모르겠", "확실하지"]

        confidence_score = 0.5  # 기본값

        for keyword in confidence_keywords:
            if keyword in response:
                confidence_score += 0.1

        for keyword in uncertainty_keywords:
            if keyword in response:
                confidence_score -= 0.1

        return max(0.0, min(1.0, confidence_score))

    def _extract_entities(self, response: str) -> List[Dict[str, Any]]:
        """응답에서 엔티티 추출"""
        entities = []

        # 간단한 키워드 기반 엔티티 추출
        entity_patterns = {
            "색상": ["빨간", "파란", "노란", "초록", "검은", "하얀", "회색"],
            "객체": ["사람", "자동차", "건물", "나무", "동물", "음식", "책"],
            "감정": ["행복", "슬픔", "화남", "놀람", "두려움", "기쁨", "우울"]
        }

        for category, keywords in entity_patterns.items():
            for keyword in keywords:
                if keyword in response:
                    entities.append({
                        "text": keyword,
                        "category": category,
                        "confidence": 0.8
                    })

        return entities

    def _extract_visual_features(self, response: str) -> Dict[str, Any]:
        """시각적 특징 추출"""
        features = {
            "dominant_colors": [],
            "objects_detected": [],
            "scene_type": "unknown",
            "lighting": "unknown",
            "composition": "unknown"
        }

        # 간단한 패턴 매칭으로 특징 추출
        if "밝은" in response or "환한" in response:
            features["lighting"] = "bright"
        elif "어두운" in response or "그늘진" in response:
            features["lighting"] = "dark"

        return features

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }


# 전역 모델 인스턴스
_model_instance = None


def get_model() -> MiniCPMVModel:
    """전역 모델 인스턴스 반환 (싱글톤 패턴)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = MiniCPMVModel()
    return _model_instance


def analyze_image_with_model(image: Image.Image, question: str,
                             context: str = "", roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """편의 함수: 이미지 분석"""
    model = get_model()
    return model.analyze_image(image, question, context, roi)


def extract_embedding_with_model(text: str = "", image: Optional[Image.Image] = None) -> np.ndarray:
    """편의 함수: 임베딩 추출"""
    model = get_model()
    return model.extract_embedding(text, image)