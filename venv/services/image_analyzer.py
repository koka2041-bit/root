# services/image_analyzer.py
# 이미지 분석 및 상황 판단 - MiniCPM-V 2.6 기반 멀티모달 분석

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any
import json
import base64
import io
from dataclasses import dataclass
import cv2

from utils.vision_utils import VisionAnalyzer
from debug_logger import get_debugger

debugger = get_debugger()


@dataclass
class ROI:
    """관심 영역(Region of Interest) 정의"""
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    confidence: float = 0.0


@dataclass
class AnalysisResult:
    """이미지 분석 결과"""
    scene_description: str
    objects: List[Dict[str, Any]]
    colors: Dict[str, float]
    composition: Dict[str, Any]
    context: Dict[str, Any]
    rois: List[ROI]
    text_content: List[str]
    emotions: List[str]
    suggested_questions: List[str]


class ImageAnalyzer:
    """
    MiniCPM-V 2.6 기반 이미지 분석기
    - 종합적 이미지 분석
    - ROI 지정 및 분석
    - 텍스트와 이미지 정보 결합
    - 상황별 맞춤 분석
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Args:
            model: MiniCPM-V 모델 인스턴스
            tokenizer: 토크나이저 인스턴스
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vision_analyzer = VisionAnalyzer()

        # 분석 프롬프트 템플릿
        self.analysis_prompts = {
            "comprehensive": "이 이미지를 자세히 분석해주세요. 물체, 색상, 구성, 분위기, 상황을 포함해서 설명해주세요.",
            "objects": "이 이미지에 있는 모든 물체들을 나열하고 각각의 위치와 특징을 설명해주세요.",
            "scene": "이 이미지의 장면과 상황을 설명해주세요. 어떤 일이 일어나고 있나요?",
            "colors": "이 이미지의 주요 색상들과 색감의 특징을 분석해주세요.",
            "composition": "이 이미지의 구성과 레이아웃을 분석해주세요.",
            "emotions": "이 이미지에서 느껴지는 감정이나 분위기를 분석해주세요.",
            "text_ocr": "이 이미지에 있는 모든 텍스트를 읽어주세요.",
            "roi_analysis": "지정된 영역을 중심으로 이미지를 분석해주세요."
        }

    async def analyze_comprehensive(
            self,
            image: Image.Image,
            user_question: str = "",
            roi_list: Optional[List[ROI]] = None
    ) -> AnalysisResult:
        """
        종합적 이미지 분석

        Args:
            image: 분석할 이미지
            user_question: 사용자 질문 (컨텍스트)
            roi_list: 관심 영역 리스트

        Returns:
            종합 분석 결과
        """
        debugger.debug(f"종합 이미지 분석 시작: 이미지 크기 {image.size}", "IMAGE_ANALYSIS")

        try:
            # 기본 이미지 분석
            basic_analysis = await self._analyze_basic_features(image)

            # 장면 및 객체 분석
            scene_analysis = await self._analyze_scene_and_objects(image, user_question)

            # 색상 및 구성 분석
            visual_analysis = await self._analyze_visual_elements(image)

            # ROI 분석 (지정된 경우)
            roi_analysis = []
            if roi_list:
                roi_analysis = await self._analyze_rois(image, roi_list)

            # 텍스트 추출 (OCR)
            text_content = await self._extract_text_content(image)

            # 감정 및 분위기 분석
            emotion_analysis = await self._analyze_emotions(image)

            # 추천 질문 생성
            suggested_questions = self._generate_suggested_questions(
                scene_analysis, basic_analysis, user_question
            )

            # 결과 종합
            result = AnalysisResult(
                scene_description=scene_analysis.get("description", ""),
                objects=scene_analysis.get("objects", []),
                colors=visual_analysis.get("colors", {}),
                composition=visual_analysis.get("composition", {}),
                context={
                    "user_question": user_question,
                    "analysis_confidence": basic_analysis.get("confidence", 0.0),
                    "image_quality": basic_analysis.get("quality", "unknown")
                },
                rois=roi_analysis,
                text_content=text_content,
                emotions=emotion_analysis,
                suggested_questions=suggested_questions
            )

            debugger.success("종합 이미지 분석 완료", "IMAGE_ANALYSIS")
            return result

        except Exception as e:
            debugger.error(f"이미지 분석 중 오류: {e}", "IMAGE_ANALYSIS")
            return AnalysisResult(
                scene_description=f"분석 중 오류 발생: {str(e)}",
                objects=[],
                colors={},
                composition={},
                context={"error": str(e)},
                rois=[],
                text_content=[],
                emotions=[],
                suggested_questions=[]
            )

    async def _analyze_basic_features(self, image: Image.Image) -> Dict[str, Any]:
        """기본 이미지 특징 분석"""

        # 이미지 품질 평가
        width, height = image.size
        aspect_ratio = width / height

        # 색상 모드 확인
        color_mode = image.mode

        # 평균 밝기 계산
        grayscale = image.convert('L')
        avg_brightness = np.array(grayscale).mean()

        return {
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": aspect_ratio,
            "color_mode": color_mode,
            "avg_brightness": avg_brightness,
            "quality": "high" if width > 800 and height > 600 else "medium" if width > 400 else "low",
            "confidence": 0.9
        }

    async def _analyze_scene_and_objects(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        """장면 및 객체 분석"""

        if not self.model or not self.tokenizer:
            debugger.warning("MiniCPM-V 모델이 없어 기본 분석 사용", "IMAGE_ANALYSIS")
            return self._fallback_scene_analysis(image)

        try:
            # 컨텍스트 포함 프롬프트 구성
            if context:
                prompt = f"{self.analysis_prompts['comprehensive']} 특히 '{context}'와 관련된 내용을 중심으로 분석해주세요."
            else:
                prompt = self.analysis_prompts['comprehensive']

            # MiniCPM-V로 이미지 분석
            messages = [{"role": "user", "content": prompt}]

            with torch.no_grad():
                response = self.model.chat(
                    image=image,
                    msgs=messages,
                    tokenizer=self.tokenizer,
                    temperature=0.3
                )

            # 응답 파싱
            objects = self._parse_objects_from_response(response)
            description = self._extract_scene_description(response)

            return {
                "description": description,
                "objects": objects,
                "raw_response": response,
                "confidence": 0.85
            }

        except Exception as e:
            debugger.error(f"MiniCPM-V 장면 분석 실패: {e}", "IMAGE_ANALYSIS")
            return self._fallback_scene_analysis(image)

    def _fallback_scene_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """기본 장면 분석 (모델 없을 때)"""

        # 색상 기반 기본 분석
        colors = image.getcolors(maxcolors=256 * 256 * 256)
        dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5] if colors else []

        # 밝기 기반 분위기 추정
        avg_brightness = np.array(image.convert('L')).mean()

        if avg_brightness > 180:
            mood = "밝고 화사한"
        elif avg_brightness > 120:
            mood = "중간 밝기의"
        else:
            mood = "어둡고 차분한"

        return {
            "description": f"{mood} 분위기의 이미지입니다.",
            "objects": [{"name": "알 수 없는 객체", "confidence": 0.3}],
            "confidence": 0.3
        }

    def _parse_objects_from_response(self, response: str) -> List[Dict[str, Any]]:
        """응답에서 객체 정보 추출"""
        objects = []

        # 간단한 패턴 매칭으로 객체 추출
        import re

        # "사람", "자동차", "건물" 등의 패턴 찾기
        object_patterns = [
            r'([가-힣]+)(?:이|가|을|를|는|은)',
            r'(\w+)(?:\s+(?:보입니다|있습니다|보여집니다))',
        ]

        for pattern in object_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if len(match) > 1 and match not in [obj["name"] for obj in objects]:
                    objects.append({
                        "name": match,
                        "confidence": 0.7,
                        "description": f"{match}가 이미지에서 감지됨"
                    })

        return objects[:10]  # 최대 10개

    def _extract_scene_description(self, response: str) -> str:
        """응답에서 장면 설명 추출"""
        # 첫 번째 문장이나 주요 설명 부분 추출
        sentences = response.split('.')
        if sentences:
            return sentences[0].strip() + '.'
        return response[:100] + "..." if len(response) > 100 else response

    async def _analyze_visual_elements(self, image: Image.Image) -> Dict[str, Any]:
        """시각적 요소 분석 (색상, 구성)"""

        # 색상 분석
        colors_analysis = self._analyze_colors(image)

        # 구성 분석 (OpenCV 사용)
        composition_analysis = self._analyze_composition(image)

        return {
            "colors": colors_analysis,
            "composition": composition_analysis
        }

    def _analyze_colors(self, image: Image.Image) -> Dict[str, float]:
        """색상 분석"""

        # RGB 이미지로 변환
        rgb_image = image.convert('RGB')

        # 색상 히스토그램
        colors = rgb_image.getcolors(maxcolors=256 * 256 * 256)

        if not colors:
            return {}

        # 주요 색상 추출
        total_pixels = sum([count for count, color in colors])
        color_percentages = {}

        # 색상 이름 매핑 (간단화)
        color_names = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128)
        }

        for count, color in colors[:10]:  # 상위 10개 색상
            percentage = (count / total_pixels) * 100

            # 가장 가까운 색상 이름 찾기
            closest_color = min(color_names.items(),
                                key=lambda x: sum(abs(a - b) for a, b in zip(color, x[1])))

            color_name = closest_color[0]
            if color_name in color_percentages:
                color_percentages[color_name] += percentage
            else:
                color_percentages[color_name] = percentage

        return color_percentages

    def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """구성 분석"""

        try:
            # PIL을 OpenCV 형식으로 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 엣지 검출
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # 대비 분석
            contrast = np.std(gray)

            # 대칭성 분석 (간단화)
            height, width = gray.shape
            left_half = gray[:, :width // 2]
            right_half = cv2.flip(gray[:, width // 2:], 1)

            # 크기 맞추기
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]

            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0

            return {
                "edge_density": float(edge_density),
                "contrast": float(contrast),
                "symmetry": float(symmetry),
                "complexity": "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low"
            }

        except Exception as e:
            debugger.warning(f"구성 분석 중 오류: {e}", "IMAGE_ANALYSIS")
            return {"error": str(e)}

    async def _analyze_rois(self, image: Image.Image, roi_list: List[ROI]) -> List[ROI]:
        """관심 영역 분석"""

        analyzed_rois = []

        for roi in roi_list:
            try:
                # ROI 영역 추출
                roi_image = image.crop((roi.x, roi.y, roi.x + roi.width, roi.y + roi.height))

                # ROI 영역 분석
                if self.model and self.tokenizer:
                    roi_analysis = await self._analyze_roi_with_model(roi_image, roi.label)
                else:
                    roi_analysis = self._basic_roi_analysis(roi_image)

                # ROI 정보 업데이트
                analyzed_roi = ROI(
                    x=roi.x,
                    y=roi.y,
                    width=roi.width,
                    height=roi.height,
                    label=roi.label or roi_analysis.get("description", "관심 영역"),
                    confidence=roi_analysis.get("confidence", 0.5)
                )

                analyzed_rois.append(analyzed_roi)

            except Exception as e:
                debugger.error(f"ROI 분석 중 오류: {e}", "IMAGE_ANALYSIS")
                continue

        return analyzed_rois

    async def _analyze_roi_with_model(self, roi_image: Image.Image, context: str = "") -> Dict[str, Any]:
        """모델을 사용한 ROI 분석"""

        prompt = f"{self.analysis_prompts['roi_analysis']} {context}"
        messages = [{"role": "user", "content": prompt}]

        try:
            with torch.no_grad():
                response = self.model.chat(
                    image=roi_image,
                    msgs=messages,
                    tokenizer=self.tokenizer,
                    temperature=0.3
                )

            return {
                "description": response[:100],
                "confidence": 0.8,
                "raw_response": response
            }

        except Exception as e:
            debugger.error(f"ROI 모델 분석 실패: {e}", "IMAGE_ANALYSIS")
            return self._basic_roi_analysis(roi_image)

    def _basic_roi_analysis(self, roi_image: Image.Image) -> Dict[str, Any]:
        """기본 ROI 분석"""

        # 색상 분포
        colors = self._analyze_colors(roi_image)
        dominant_color = max(colors.items(), key=lambda x: x[1])[0] if colors else "unknown"

        # 밝기
        avg_brightness = np.array(roi_image.convert('L')).mean()

        return {
            "description": f"{dominant_color} 계열의 영역",
            "confidence": 0.4,
            "dominant_color": dominant_color,
            "brightness": avg_brightness
        }

    async def _extract_text_content(self, image: Image.Image) -> List[str]:
        """텍스트 추출 (OCR)"""

        try:
            if self.model and self.tokenizer:
                # MiniCPM-V로 텍스트 추출
                messages = [{"role": "user", "content": self.analysis_prompts['text_ocr']}]

                with torch.no_grad():
                    response = self.model.chat(
                        image=image,
                        msgs=messages,
                        tokenizer=self.tokenizer,
                        temperature=0.1
                    )

                # 텍스트 파싱
                text_lines = [line.strip() for line in response.split('\n') if line.strip()]
                return text_lines[:10]  # 최대 10줄

            else:
                # 기본 OCR (pytesseract 등 사용 가능)
                return ["텍스트 추출 기능 제한됨"]

        except Exception as e:
            debugger.error(f"텍스트 추출 중 오류: {e}", "IMAGE_ANALYSIS")
            return []

    async def _analyze_emotions(self, image: Image.Image) -> List[str]:
        """감정 및 분위기 분석"""

        try:
            if self.model and self.tokenizer:
                messages = [{"role": "user", "content": self.analysis_prompts['emotions']}]

                with torch.no_grad():
                    response = self.model.chat(
                        image=image,
                        msgs=messages,
                        tokenizer=self.tokenizer,
                        temperature=0.5
                    )

                # 감정 키워드 추출
                emotion_keywords = self._extract_emotion_keywords(response)
                return emotion_keywords

            else:
                # 색상 기반 감정 추정
                colors = self._analyze_colors(image)
                return self._emotion_from_colors(colors)

        except Exception as e:
            debugger.error(f"감정 분석 중 오류: {e}", "IMAGE_ANALYSIS")
            return []

    def _extract_emotion_keywords(self, response: str) -> List[str]:
        """응답에서 감정 키워드 추출"""

        emotion_words = [
            "행복", "기쁨", "즐거움", "웃음", "밝음",
            "슬픔", "우울", "어둠", "눈물", "고요",
            "화남", "분노", "긴장", "스트레스",
            "평온", "차분", "안정", "편안", "따뜻",
            "신비", "경이", "놀라움", "호기심",
            "로맨틱", "사랑", "애정", "포근"
        ]

        detected_emotions = []
        response_lower = response.lower()

        for emotion in emotion_words:
            if emotion in response_lower:
                detected_emotions.append(emotion)

        return detected_emotions[:5]  # 최대 5개

    def _emotion_from_colors(self, colors: Dict[str, float]) -> List[str]:
        """색상 기반 감정 추정"""

        color_emotions = {
            "red": ["열정", "에너지", "사랑"],
            "blue": ["차분", "평온", "신뢰"],
            "green": ["자연", "안정", "성장"],
            "yellow": ["행복", "밝음", "활력"],
            "orange": ["따뜻", "친근", "활동적"],
            "purple": ["신비", "고급", "창의"],
            "pink": ["부드러움", "사랑", "따뜻"],
            "brown": ["안정", "자연", "편안"],
            "black": ["강함", "세련", "신비"],
            "white": ["순수", "깔끔", "평화"],
            "gray": ["중성", "현대적", "차분"]
        }

        emotions = []
        for color, percentage in colors.items():
            if percentage > 10 and color in color_emotions:  # 10% 이상인 색상만
                emotions.extend(color_emotions[color])

        return list(set(emotions))[:5]  # 중복 제거, 최대 5개

    def _generate_suggested_questions(
            self,
            scene_analysis: Dict,
            basic_analysis: Dict,
            user_question: str
    ) -> List[str]:
        """추천 질문 생성"""

        questions = []

        # 객체 기반 질문
        objects = scene_analysis.get("objects", [])
        if objects:
            obj_names = [obj["name"] for obj in objects[:3]]
            questions.append(f"이 이미지의 {', '.join(obj_names)}에 대해 더 자세히 알려주세요.")

        # 장면 기반 질문
        if scene_analysis.get("description"):
            questions.append("이 상황에서 무슨 일이 일어나고 있나요?")
            questions.append("이 장면의 배경이나 맥락을 설명해주세요.")

        # 감정 기반 질문
        questions.append("이 이미지에서 어떤 감정이나 분위기가 느껴지나요?")

        # 색상/구성 기반 질문
        questions.append("이 이미지의 색감이나 구성에 특별한 의미가 있나요?")

        # 사용자 질문 관련 후속 질문
        if user_question:
            questions.append(f"'{user_question}'와 관련해서 이 이미지에서 놓친 부분이 있을까요?")

        return questions[:5]  # 최대 5개

    async def analyze_with_roi(
            self,
            image: Image.Image,
            x: int, y: int, width: int, height: int,
            question: str = ""
    ) -> Dict[str, Any]:
        """특정 ROI 영역 분석"""

        roi = ROI(x=x, y=y, width=width, height=height, label=question)

        # ROI 이미지 추출
        roi_image = image.crop((x, y, x + width, y + height))

        # 상세 분석
        analysis = await self._analyze_roi_with_model(roi_image, question)

        # 시각화를 위한 이미지 생성
        annotated_image = self._create_roi_visualization(image, roi)

        return {
            "roi_analysis": analysis,
            "roi_coordinates": {"x": x, "y": y, "width": width, "height": height},
            "annotated_image": annotated_image,
            "question": question
        }

    def _create_roi_visualization(self, image: Image.Image, roi: ROI) -> Image.Image:
        """ROI 시각화 이미지 생성"""

        # 이미지 복사
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # ROI 사각형 그리기
        rectangle_coords = [roi.x, roi.y, roi.x + roi.width, roi.y + roi.height]
        draw.rectangle(rectangle_coords, outline="red", width=3)

        # 라벨 추가
        if roi.label:
            try:
                # 기본 폰트 사용
                draw.text((roi.x, roi.y - 20), roi.label, fill="red")
            except:
                # 폰트 로딩 실패 시 기본 텍스트
                pass

        return annotated

    async def compare_images(
            self,
            image1: Image.Image,
            image2: Image.Image,
            comparison_focus: str = ""
    ) -> Dict[str, Any]:
        """두 이미지 비교 분석"""

        try:
            # 각각 분석
            analysis1 = await self.analyze_comprehensive(image1, f"첫 번째 이미지 {comparison_focus}")
            analysis2 = await self.analyze_comprehensive(image2, f"두 번째 이미지 {comparison_focus}")

            # 차이점 분석
            differences = self._find_differences(analysis1, analysis2)

            # 유사점 분석
            similarities = self._find_similarities(analysis1, analysis2)

            return {
                "image1_analysis": analysis1,
                "image2_analysis": analysis2,
                "differences": differences,
                "similarities": similarities,
                "comparison_summary": self._generate_comparison_summary(differences, similarities)
            }

        except Exception as e:
            debugger.error(f"이미지 비교 중 오류: {e}", "IMAGE_ANALYSIS")
            return {"error": str(e)}

    def _find_differences(self, analysis1: AnalysisResult, analysis2: AnalysisResult) -> Dict[str, Any]:
        """두 분석 결과의 차이점 찾기"""

        differences = {}

        # 객체 차이
        objects1 = {obj["name"] for obj in analysis1.objects}
        objects2 = {obj["name"] for obj in analysis2.objects}

        differences["objects"] = {
            "only_in_image1": list(objects1 - objects2),
            "only_in_image2": list(objects2 - objects1)
        }

        # 색상 차이
        colors1 = set(analysis1.colors.keys())
        colors2 = set(analysis2.colors.keys())

        differences["colors"] = {
            "only_in_image1": list(colors1 - colors2),
            "only_in_image2": list(colors2 - colors1)
        }

        # 감정 차이
        emotions1 = set(analysis1.emotions)
        emotions2 = set(analysis2.emotions)

        differences["emotions"] = {
            "only_in_image1": list(emotions1 - emotions2),
            "only_in_image2": list(emotions2 - emotions1)
        }

        return differences

    def _find_similarities(self, analysis1: AnalysisResult, analysis2: AnalysisResult) -> Dict[str, Any]:
        """두 분석 결과의 유사점 찾기"""

        similarities = {}

        # 공통 객체
        objects1 = {obj["name"] for obj in analysis1.objects}
        objects2 = {obj["name"] for obj in analysis2.objects}
        similarities["common_objects"] = list(objects1 & objects2)

        # 공통 색상
        colors1 = set(analysis1.colors.keys())
        colors2 = set(analysis2.colors.keys())
        similarities["common_colors"] = list(colors1 & colors2)

        # 공통 감정
        emotions1 = set(analysis1.emotions)
        emotions2 = set(analysis2.emotions)
        similarities["common_emotions"] = list(emotions1 & emotions2)

        return similarities

    def _generate_comparison_summary(self, differences: Dict, similarities: Dict) -> str:
        """비교 요약 생성"""

        summary_parts = []

        # 유사점
        if similarities.get("common_objects"):
            summary_parts.append(f"두 이미지 모두 {', '.join(similarities['common_objects'][:3])}이 포함되어 있습니다.")

        # 차이점
        if differences.get("objects", {}).get("only_in_image1"):
            summary_parts.append(f"첫 번째 이미지에만 {', '.join(differences['objects']['only_in_image1'][:3])}이 있습니다.")

        if differences.get("objects", {}).get("only_in_image2"):
            summary_parts.append(f"두 번째 이미지에만 {', '.join(differences['objects']['only_in_image2'][:3])}이 있습니다.")

        return " ".join(summary_parts) if summary_parts else "두 이미지는 비슷한 특징을 가지고 있습니다."