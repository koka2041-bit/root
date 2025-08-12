"""
이미지 처리 및 ROI(관심영역) 추출 유틸리티
MiniCPM-V 모델과 연동하여 시각적 분석 지원
"""
#비전 유틸리티 (utils/vision_utils.py)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)


class ROIExtractor:
    """관심 영역(ROI) 추출 및 관리 클래스"""

    def __init__(self):
        """ROI 추출기 초기화"""
        self.roi_history = []
        self.current_image = None

    def extract_roi_auto(self, image: Image.Image, method: str = "saliency") -> List[Dict[str, Any]]:
        """
        자동 ROI 추출

        Args:
            image (Image.Image): 입력 이미지
            method (str): 추출 방법 ('saliency', 'contour', 'edge')

        Returns:
            List[Dict]: ROI 정보 리스트
        """
        try:
            # PIL을 OpenCV 형식으로 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            rois = []

            if method == "saliency":
                rois = self._extract_saliency_roi(cv_image)
            elif method == "contour":
                rois = self._extract_contour_roi(cv_image)
            elif method == "edge":
                rois = self._extract_edge_roi(cv_image)
            else:
                # 기본: 전체 이미지를 ROI로 설정
                rois = [{
                    "x": 0, "y": 0,
                    "width": image.width,
                    "height": image.height,
                    "confidence": 1.0,
                    "type": "full_image"
                }]

            # ROI 이력 저장
            self.roi_history.append({
                "method": method,
                "rois": rois,
                "image_size": (image.width, image.height)
            })

            return rois

        except Exception as e:
            logger.error(f"ROI 추출 실패: {str(e)}")
            return []

    def extract_roi_manual(self, image: Image.Image, coordinates: List[Tuple[int, int, int, int]]) -> List[
        Dict[str, Any]]:
        """
        수동 ROI 지정

        Args:
            image (Image.Image): 입력 이미지
            coordinates (List[Tuple]): ROI 좌표 리스트 (x, y, width, height)

        Returns:
            List[Dict]: ROI 정보 리스트
        """
        rois = []

        for i, (x, y, w, h) in enumerate(coordinates):
            # 이미지 범위 내 좌표 검증
            x = max(0, min(x, image.width - 1))
            y = max(0, min(y, image.height - 1))
            w = min(w, image.width - x)
            h = min(h, image.height - y)

            rois.append({
                "x": x, "y": y,
                "width": w, "height": h,
                "confidence": 1.0,
                "type": "manual",
                "index": i
            })

        return rois

    def _extract_saliency_roi(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        """현저성 기반 ROI 추출"""
        try:
            # 간단한 현저성 맵 생성 (실제로는 더 정교한 알고리즘 사용)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 가우시안 블러와 차이를 이용한 현저성 계산
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            saliency = cv2.absdiff(gray, blur)

            # 임계값 적용
            _, thresh = cv2.threshold(saliency, 30, 255, cv2.THRESH_BINARY)

            # 컨투어 찾기
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rois = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 최소 크기 필터
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(1.0, area / (cv_image.shape[0] * cv_image.shape[1]))

                    rois.append({
                        "x": x, "y": y,
                        "width": w, "height": h,
                        "confidence": confidence,
                        "type": "saliency",
                        "area": area
                    })

            # 신뢰도 순으로 정렬
            rois.sort(key=lambda x: x["confidence"], reverse=True)
            return rois[:5]  # 상위 5개만 반환

        except Exception as e:
            logger.error(f"현저성 ROI 추출 실패: {str(e)}")
            return []

    def _extract_contour_roi(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        """컨투어 기반 ROI 추출"""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # 컨투어 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rois = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    rois.append({
                        "x": x, "y": y,
                        "width": w, "height": h,
                        "confidence": 0.7,
                        "type": "contour",
                        "area": area
                    })

            return rois[:10]

        except Exception as e:
            logger.error(f"컨투어 ROI 추출 실패: {str(e)}")
            return []

    def _extract_edge_roi(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        """엣지 기반 ROI 추출"""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # 형태학적 연산으로 엣지 연결
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 컨투어 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rois = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 800:
                    x, y, w, h = cv2.boundingRect(contour)
                    rois.append({
                        "x": x, "y": y,
                        "width": w, "height": h,
                        "confidence": 0.6,
                        "type": "edge",
                        "area": area
                    })

            return rois[:8]

        except Exception as e:
            logger.error(f"엣지 ROI 추출 실패: {str(e)}")
            return []


class ImageProcessor:
    """이미지 전처리 및 후처리 클래스"""

    @staticmethod
    def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = None) -> Image.Image:
        """
        이미지 전처리

        Args:
            image (Image.Image): 입력 이미지
            target_size (Tuple[int, int]): 목표 크기 (width, height)

        Returns:
            Image.Image: 전처리된 이미지
        """
        try:
            processed = image.copy()

            # 크기 조정
            if target_size:
                processed = processed.resize(target_size, Image.Resampling.LANCZOS)

            # RGB 모드 변환
            if processed.mode != 'RGB':
                processed = processed.convert('RGB')

            return processed

        except Exception as e:
            logger.error(f"이미지 전처리 실패: {str(e)}")
            return image

    @staticmethod
    def enhance_image(image: Image.Image, enhance_type: str = "auto") -> Image.Image:
        """
        이미지 품질 향상

        Args:
            image (Image.Image): 입력 이미지
            enhance_type (str): 향상 타입 ('auto', 'contrast', 'brightness', 'sharpness')

        Returns:
            Image.Image: 향상된 이미지
        """
        try:
            from PIL import ImageEnhance

            enhanced = image.copy()

            if enhance_type == "auto" or enhance_type == "contrast":
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.2)

            if enhance_type == "auto" or enhance_type == "brightness":
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.1)

            if enhance_type == "auto" or enhance_type == "sharpness":
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.1)

            return enhanced

        except Exception as e:
            logger.error(f"이미지 향상 실패: {str(e)}")
            return image

    @staticmethod
    def draw_roi_overlay(image: Image.Image, rois: List[Dict[str, Any]],
                         show_confidence: bool = True) -> Image.Image:
        """
        ROI 오버레이 그리기

        Args:
            image (Image.Image): 기본 이미지
            rois (List[Dict]): ROI 정보 리스트
            show_confidence (bool): 신뢰도 표시 여부

        Returns:
            Image.Image: ROI가 그려진 이미지
        """
        try:
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)

            # 기본 폰트 (시스템에 따라 조정 필요)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

            for i, roi in enumerate(rois):
                color = colors[i % len(colors)]
                x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]

                # ROI 사각형 그리기
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

                # 신뢰도 및 타입 표시
                if show_confidence:
                    confidence = roi.get("confidence", 0.0)
                    roi_type = roi.get("type", "unknown")
                    label = f"{roi_type}: {confidence:.2f}"

                    # 라벨 배경
                    bbox = draw.textbbox((x, y - 20), label, font=font)
                    draw.rectangle(bbox, fill=color, outline=color)
                    draw.text((x, y - 20), label, fill="white", font=font)

            return overlay

        except Exception as e:
            logger.error(f"ROI 오버레이 그리기 실패: {str(e)}")
            return image


class VisualFeatureExtractor:
    """시각적 특징 추출 클래스"""

    @staticmethod
    def extract_color_features(image: Image.Image) -> Dict[str, Any]:
        """색상 특징 추출"""
        try:
            # 이미지를 numpy 배열로 변환
            img_array = np.array(image)

            # 주요 색상 추출 (K-means 클러스터링 사용)
            from sklearn.cluster import KMeans

            # 픽셀 데이터 reshape
            pixels = img_array.reshape(-1, 3)

            # K-means로 주요 색상 추출 (5개)
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)

            colors = kmeans.cluster_centers_.astype(int)

            # 색상별 비율 계산
            labels = kmeans.labels_
            color_ratios = []
            for i in range(5):
                ratio = np.sum(labels == i) / len(labels)
                color_ratios.append(ratio)

            # 색상 이름 매핑
            color_names = []
            for color in colors:
                name = VisualFeatureExtractor._get_color_name(color)
                color_names.append(name)

            return {
                "dominant_colors": [
                    {
                        "rgb": color.tolist(),
                        "name": name,
                        "ratio": ratio
                    }
                    for color, name, ratio in zip(colors, color_names, color_ratios)
                ],
                "color_diversity": len(np.unique(labels)) / len(labels),
                "brightness": np.mean(img_array),
                "contrast": np.std(img_array)
            }

        except Exception as e:
            logger.error(f"색상 특징 추출 실패: {str(e)}")
            return {"dominant_colors": [], "color_diversity": 0.0}

    @staticmethod
    def _get_color_name(rgb: np.ndarray) -> str:
        """RGB 값을 색상 이름으로 변환"""
        r, g, b = rgb

        # 기본적인 색상 분류
        if r > 200 and g > 200 and b > 200:
            return "흰색"
        elif r < 50 and g < 50 and b < 50:
            return "검은색"
        elif r > g and r > b:
            return "빨간색"
        elif g > r and g > b:
            return "초록색"
        elif b > r and b > g:
            return "파란색"
        elif r > 150 and g > 150 and b < 100:
            return "노란색"
        elif r > 150 and g < 100 and b > 150:
            return "보라색"
        elif r > 150 and g > 100 and b < 100:
            return "주황색"
        else:
            return "회색"

    @staticmethod
    def extract_texture_features(image: Image.Image) -> Dict[str, Any]:
        """텍스처 특징 추출"""
        try:
            # 그레이스케일 변환
            gray = image.convert('L')
            img_array = np.array(gray)

            # 엣지 검출
            edges = cv2.Canny(img_array, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # 텍스처 방향성 (간단한 그래디언트 기반)
            grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)

            # 텍스처 복잡도
            texture_complexity = np.std(img_array)

            return {
                "edge_density": float(edge_density),
                "texture_complexity": float(texture_complexity),
                "gradient_magnitude": float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))),
                "smoothness": 1.0 / (1.0 + texture_complexity)
            }

        except Exception as e:
            logger.error(f"텍스처 특징 추출 실패: {str(e)}")
            return {"edge_density": 0.0, "texture_complexity": 0.0}


def analyze_image_composition(image: Image.Image) -> Dict[str, Any]:
    """
    이미지 구도 분석

    Args:
        image (Image.Image): 입력 이미지

    Returns:
        Dict[str, Any]: 구도 분석 결과
    """
    try:
        width, height = image.size

        # 황금비율 분석
        golden_ratio = 1.618
        actual_ratio = width / height
        golden_ratio_score = 1.0 - abs(actual_ratio - golden_ratio) / golden_ratio

        # 삼등분법 분석 (Rule of Thirds)
        third_x = width // 3
        third_y = height // 3

        # 관심점들이 삼등분선에 얼마나 가까운지 측정
        roi_extractor = ROIExtractor()
        rois = roi_extractor.extract_roi_auto(image, method="saliency")

        thirds_alignment_score = 0.0
        if rois:
            for roi in rois[:3]:  # 상위 3개 ROI만 고려
                center_x = roi["x"] + roi["width"] // 2
                center_y = roi["y"] + roi["height"] // 2

                # 삼등분선까지의 거리 계산
                dist_to_thirds = min(
                    abs(center_x - third_x), abs(center_x - 2 * third_x),
                    abs(center_y - third_y), abs(center_y - 2 * third_y)
                )

                # 거리를 점수로 변환 (가까울수록 높은 점수)
                max_dist = min(width, height) // 6
                alignment = max(0, 1.0 - dist_to_thirds / max_dist)
                thirds_alignment_score += alignment

            thirds_alignment_score /= len(rois[:3])

        return {
            "aspect_ratio": actual_ratio,
            "golden_ratio_score": max(0.0, golden_ratio_score),
            "thirds_alignment_score": thirds_alignment_score,
            "composition_balance": _calculate_balance_score(image),
            "visual_weight_distribution": _analyze_visual_weight(image)
        }

    except Exception as e:
        logger.error(f"구도 분석 실패: {str(e)}")
        return {"aspect_ratio": 1.0, "golden_ratio_score": 0.0}


def _calculate_balance_score(image: Image.Image) -> float:
    """이미지의 시각적 균형 점수 계산"""
    try:
        img_array = np.array(image.convert('L'))
        height, width = img_array.shape

        # 이미지를 좌우로 분할
        left_half = img_array[:, :width // 2]
        right_half = img_array[:, width // 2:]

        # 각 반쪽의 시각적 가중치 계산 (밝기 기반)
        left_weight = np.sum(left_half)
        right_weight = np.sum(right_half)

        # 균형 점수 계산 (0~1, 1이 완전 균형)
        total_weight = left_weight + right_weight
        if total_weight == 0:
            return 0.0

        balance_ratio = min(left_weight, right_weight) / max(left_weight, right_weight)
        return balance_ratio

    except Exception as e:
        logger.error(f"균형 점수 계산 실패: {str(e)}")
        return 0.0


def _analyze_visual_weight(image: Image.Image) -> Dict[str, float]:
    """시각적 가중치 분포 분석"""
    try:
        img_array = np.array(image.convert('L'))
        height, width = img_array.shape

        # 이미지를 9등분 (3x3 그리드)
        grid_weights = {}
        for i in range(3):
            for j in range(3):
                y_start = i * height // 3
                y_end = (i + 1) * height // 3
                x_start = j * width // 3
                x_end = (j + 1) * width // 3

                grid_section = img_array[y_start:y_end, x_start:x_end]
                weight = np.mean(grid_section)
                grid_weights[f"grid_{i}_{j}"] = float(weight)

        return grid_weights

    except Exception as e:
        logger.error(f"시각적 가중치 분석 실패: {str(e)}")
        return {}


def create_analysis_report(image: Image.Image, question: str = "") -> Dict[str, Any]:
    """
    종합적인 이미지 분석 보고서 생성

    Args:
        image (Image.Image): 분석할 이미지
        question (str): 분석 질문

    Returns:
        Dict[str, Any]: 종합 분석 결과
    """
    try:
        # 각종 분석 수행
        roi_extractor = ROIExtractor()
        feature_extractor = VisualFeatureExtractor()

        # ROI 추출
        rois = roi_extractor.extract_roi_auto(image, method="saliency")

        # 색상 특징 추출
        color_features = feature_extractor.extract_color_features(image)

        # 텍스처 특징 추출
        texture_features = feature_extractor.extract_texture_features(image)

        # 구도 분석
        composition_analysis = analyze_image_composition(image)

        # 종합 보고서 구성
        report = {
            "image_info": {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format
            },
            "roi_analysis": {
                "total_rois": len(rois),
                "rois": rois[:5],  # 상위 5개만 포함
                "roi_coverage": sum(roi.get("area", 0) for roi in rois) / (image.width * image.height) if rois else 0
            },
            "color_analysis": color_features,
            "texture_analysis": texture_features,
            "composition_analysis": composition_analysis,
            "question": question,
            "analysis_timestamp": str(np.datetime64('now'))
        }

        return report

    except Exception as e:
        logger.error(f"분석 보고서 생성 실패: {str(e)}")
        return {"error": str(e)}


# 편의 함수들
def extract_rois(image: Image.Image, method: str = "saliency") -> List[Dict[str, Any]]:
    """ROI 추출 편의 함수"""
    extractor = ROIExtractor()
    return extractor.extract_roi_auto(image, method)


def draw_rois_on_image(image: Image.Image, rois: List[Dict[str, Any]]) -> Image.Image:
    """ROI 오버레이 편의 함수"""
    processor = ImageProcessor()
    return processor.draw_roi_overlay(image, rois)


def enhance_image_quality(image: Image.Image, enhance_type: str = "auto") -> Image.Image:
    """이미지 품질 향상 편의 함수"""
    processor = ImageProcessor()
    return processor.enhance_image(image, enhance_type)