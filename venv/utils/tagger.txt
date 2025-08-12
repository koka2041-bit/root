"""
태그 추출 및 분류 유틸리티
대화 내용에서 의미있는 태그를 추출하고 카테고리별로 분류
"""
#태그 추출기 (utils/tagger.py)
import json
import re
import os
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import Counter, defaultdict
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class TagExtractor:
    """태그 추출 및 관리 클래스"""

    def __init__(self, data_dir: str = "data"):
        """
        태그 추출기 초기화

        Args:
            data_dir (str): 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.tags_dir = os.path.join(data_dir, "tags")
        self.tag_map_file = os.path.join(data_dir, "tag_map.json")

        # 태그 데이터 로드
        self.tag_categories = self._load_tag_categories()
        self.tag_patterns = self._build_tag_patterns()
        self.custom_tags = {}

        # 태그 매핑 데이터 로드
        self.tag_mappings = self._load_tag_mappings()

    def _load_tag_categories(self) -> Dict[str, Dict[str, Any]]:
        """태그 카테고리 파일들 로드"""
        categories = {}

        if not os.path.exists(self.tags_dir):
            os.makedirs(self.tags_dir, exist_ok=True)
            self._create_default_tag_files()

        # 각 카테고리 파일 로드
        category_files = ["음식.json", "감정.json", "사건.json", "관계.json"]

        for filename in category_files:
            filepath = os.path.join(self.tags_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        category_name = filename.replace('.json', '')
                        categories[category_name] = json.load(f)
                except Exception as e:
                    logger.error(f"태그 파일 로드 실패 {filename}: {str(e)}")
                    categories[filename.replace('.json', '')] = {}
            else:
                categories[filename.replace('.json', '')] = {}

        return categories

    def _create_default_tag_files(self):
        """기본 태그 파일들 생성"""
        default_tags = {
            "음식.json": {
                "한식": {
                    "keywords": ["김치", "불고기", "비빔밥", "된장찌개", "갈비", "삼겹살"],
                    "patterns": [r"(김치|불고기|비빔밥|된장|갈비|삼겹살)"],
                    "confidence": 0.9
                },
                "중식": {
                    "keywords": ["짜장면", "짬뽕", "탕수육", "마파두부", "양장피"],
                    "patterns": [r"(짜장면|짬뽕|탕수육|마파두부|양장피)"],
                    "confidence": 0.9
                },
                "일식": {
                    "keywords": ["초밥", "라멘", "우동", "돈카츠", "텐푸라"],
                    "patterns": [r"(초밥|라멘|우동|돈카츠|텐푸라)"],
                    "confidence": 0.9
                }
            },
            "감정.json": {
                "기쁨": {
                    "keywords": ["기쁘다", "즐겁다", "행복하다", "신나다", "좋다", "만족"],
                    "patterns": [r"(기쁘|즐겁|행복|신나|좋|만족|웃음|환한)"],
                    "confidence": 0.8
                },
                "슬픔": {
                    "keywords": ["슬프다", "우울하다", "힘들다", "아프다", "속상하다"],
                    "patterns": [r"(슬프|우울|힘들|아프|속상|눈물|울|낙담)"],
                    "confidence": 0.8
                },
                "화남": {
                    "keywords": ["화나다", "짜증나다", "분노", "열받다", "악"],
                    "patterns": [r"(화나|짜증|분노|열받|악|빡|미쳐)"],
                    "confidence": 0.8
                },
                "놀람": {
                    "keywords": ["놀라다", "깜짝", "신기하다", "대단하다", "와"],
                    "patterns": [r"(놀라|깜짝|신기|대단|와|헐|오마이갓)"],
                    "confidence": 0.7
                }
            },
            "사건.json": {
                "일상": {
                    "keywords": ["출근", "퇴근", "식사", "잠자리", "쇼핑", "청소"],
                    "patterns": [r"(출근|퇴근|식사|잠|쇼핑|청소|빨래|요리)"],
                    "confidence": 0.8
                },
                "특별한날": {
                    "keywords": ["생일", "기념일", "결혼식", "졸업식", "여행"],
                    "patterns": [r"(생일|기념일|결혼식|졸업식|여행|휴가|파티)"],
                    "confidence": 0.9
                },
                "업무": {
                    "keywords": ["회의", "프로젝트", "발표", "출장", "야근"],
                    "patterns": [r"(회의|프로젝트|발표|출장|야근|업무|일)"],
                    "confidence": 0.8
                }
            },
            "관계.json": {
                "가족": {
                    "keywords": ["부모님", "형제", "자매", "아이", "할머니", "할아버지"],
                    "patterns": [r"(부모|엄마|아빠|형|누나|언니|동생|할머니|할아버지|가족)"],
                    "confidence": 0.9
                },
                "친구": {
                    "keywords": ["친구", "동료", "선후배", "동기", "룸메이트"],
                    "patterns": [r"(친구|동료|선배|후배|동기|룸메이트|동아리)"],
                    "confidence": 0.8
                },
                "연인": {
                    "keywords": ["남친", "여친", "애인", "연인", "커플"],
                    "patterns": [r"(남친|여친|애인|연인|커플|데이트|사귀)"],
                    "confidence": 0.9
                }
            }
        }

        for filename, content in default_tags.items():
            filepath = os.path.join(self.tags_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

    def _build_tag_patterns(self) -> Dict[str, List[re.Pattern]]:
        """태그 패턴 컴파일"""
        patterns = {}

        for category, tags in self.tag_categories.items():
            category_patterns = []
            for tag_name, tag_data in tags.items():
                tag_patterns = tag_data.get("patterns", [])
                for pattern_str in tag_patterns:
                    try:
                        pattern = re.compile(pattern_str, re.IGNORECASE)
                        category_patterns.append((pattern, tag_name, tag_data.get("confidence", 0.5)))
                    except re.error as e:
                        logger.error(f"패턴 컴파일 실패 {pattern_str}: {str(e)}")

            patterns[category] = category_patterns

        return patterns

    def _load_tag_mappings(self) -> Dict[str, Any]:
        """태그 매핑 데이터 로드"""
        if os.path.exists(self.tag_map_file):
            try:
                with open(self.tag_map_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"태그 매핑 로드 실패: {str(e)}")

        return {"dialogue_tags": {}, "statistics": {}}

    def extract_tags(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        텍스트에서 태그 추출

        Args:
            text (str): 분석할 텍스트
            context (Dict[str, Any]): 추가 컨텍스트 정보

        Returns:
            Dict[str, Any]: 추출된 태그 정보
        """
        try:
            extracted_tags = {
                "categories": {},
                "all_tags": [],
                "confidence_scores": {},
                "extraction_metadata": {
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat(),
                    "context": context or {}
                }
            }

            # 카테고리별 태그 추출
            for category, patterns in self.tag_patterns.items():
                category_tags = self._extract_category_tags(text, patterns)
                if category_tags:
                    extracted_tags["categories"][category] = category_tags
                    extracted_tags["all_tags"].extend(category_tags)

            # 키워드 기반 태그 추출
            keyword_tags = self._extract_keyword_tags(text)
            if keyword_tags:
                extracted_tags["categories"]["키워드"] = keyword_tags
                extracted_tags["all_tags"].extend(keyword_tags)

            # 컨텍스트 기반 태그 추출
            if context:
                context_tags = self._extract_context_tags(text, context)
                if context_tags:
                    extracted_tags["categories"]["컨텍스트"] = context_tags
                    extracted_tags["all_tags"].extend(context_tags)

            # 신뢰도 점수 계산
            extracted_tags["confidence_scores"] = self._calculate_tag_confidence(
                text, extracted_tags["all_tags"]
            )

            # 중복 제거
            extracted_tags["all_tags"] = list(set(extracted_tags["all_tags"]))

            return extracted_tags

        except Exception as e:
            logger.error(f"태그 추출 실패: {str(e)}")
            return {"categories": {}, "all_tags": [], "error": str(e)}

    def _extract_category_tags(self, text: str, patterns: List[Tuple]) -> List[str]:
        """카테고리별 패턴 기반 태그 추출"""
        found_tags = []

        for pattern, tag_name, confidence in patterns:
            if pattern.search(text):
                found_tags.append(tag_name)

        return found_tags

    def _extract_keyword_tags(self, text: str) -> List[str]:
        """키워드 기반 태그 추출"""
        keyword_tags = []

        # 모든 카테고리의 키워드 검사
        for category, tags in self.tag_categories.items():
            for tag_name, tag_data in tags.items():
                keywords = tag_data.get("keywords", [])
                for keyword in keywords:
                    if keyword in text:
                        keyword_tags.append(f"{category}_{tag_name}")

        return keyword_tags

    def _extract_context_tags(self, text: str, context: Dict[str, Any]) -> List[str]:
        """컨텍스트 기반 태그 추출"""
        context_tags = []

        # 시간 컨텍스트
        if "timestamp" in context:
            time_tag = self._get_time_based_tag(context["timestamp"])
            if time_tag:
                context_tags.append(time_tag)

        # 사용자 컨텍스트
        if "user_id" in context:
            user_tag = f"사용자_{context['user_id']}"
            context_tags.append(user_tag)

        # 대화 타입 컨텍스트
        if "conversation_type" in context:
            conv