# services/learning_engine.py
# 대화형 학습 엔진 - AI가 모르는 내용을 사용자에게 질문하고 학습하는 핵심 모듈

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import uuid
from PIL import Image

from utils.embedding_utils import EmbeddingManager
from utils.memory_builder import MemoryBuilder
from utils.vision_utils import VisionAnalyzer
from services.action_manager import ActionManager


class LearningEngine:
    """
    대화형 학습 엔진
    - 모르는 내용 감지 및 질문 생성
    - 사용자 답변 기반 지속 학습
    - 정정 요청 처리 및 메모리 업데이트
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Args:
            confidence_threshold: 확신도 임계값 (이하면 질문 생성)
        """
        self.confidence_threshold = confidence_threshold
        self.memory_dir = "data/memory"
        self.entities_file = os.path.join(self.memory_dir, "entities.json")
        self.provenance_file = os.path.join(self.memory_dir, "provenance.json")

        # 필수 디렉토리 생성
        os.makedirs(self.memory_dir, exist_ok=True)

        # 컴포넌트 초기화
        self.embedding_manager = EmbeddingManager()
        self.memory_builder = MemoryBuilder()
        self.vision_analyzer = VisionAnalyzer()
        self.action_manager = ActionManager()

        # 메모리 로드
        self.entities = self._load_entities()
        self.provenance = self._load_provenance()

    def _load_entities(self) -> Dict[str, Any]:
        """엔티티 메모리 로드"""
        if os.path.exists(self.entities_file):
            with open(self.entities_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "concepts": {},  # 개념별 정보
            "labels": {},  # 라벨별 분류
            "synonyms": {},  # 동의어 매핑
            "examples": {}  # 예시 이미지
        }

    def _load_provenance(self) -> List[Dict]:
        """출처 및 정정 이력 로드"""
        if os.path.exists(self.provenance_file):
            with open(self.provenance_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_entities(self):
        """엔티티 메모리 저장"""
        with open(self.entities_file, 'w', encoding='utf-8') as f:
            json.dump(self.entities, f, ensure_ascii=False, indent=2)

    def _save_provenance(self):
        """출처 및 정정 이력 저장"""
        with open(self.provenance_file, 'w', encoding='utf-8') as f:
            json.dump(self.provenance, f, ensure_ascii=False, indent=2)

    async def process_learning_request(
            self,
            user_input: str,
            image: Optional[Image.Image] = None,
            context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        학습 요청 처리

        Args:
            user_input: 사용자 입력 텍스트
            image: 이미지 (선택사항)
            context: 대화 컨텍스트

        Returns:
            학습 결과 및 응답 정보
        """
        try:
            # 1. 입력 분석 및 임베딩 생성
            input_embedding = await self.embedding_manager.get_text_embedding(user_input)
            image_embedding = None
            image_analysis = None

            if image:
                image_embedding = await self.embedding_manager.get_image_embedding(image)
                image_analysis = await self.vision_analyzer.analyze_comprehensive(image, user_input)

            # 2. 기존 지식과 유사도 계산
            knowledge_match = await self._find_similar_knowledge(
                user_input, input_embedding, image_embedding
            )

            # 3. 확신도 판단
            confidence = knowledge_match.get("confidence", 0.0)

            if confidence < self.confidence_threshold:
                # 모르는 내용 - 질문 생성
                return await self._generate_learning_question(
                    user_input, image, image_analysis, knowledge_match
                )
            else:
                # 알고 있는 내용 - 기존 지식 기반 응답
                return await self._provide_known_answer(
                    user_input, knowledge_match, image_analysis
                )

        except Exception as e:
            return {
                "type": "error",
                "message": f"학습 처리 중 오류 발생: {str(e)}",
                "confidence": 0.0
            }

    async def _find_similar_knowledge(
            self,
            text: str,
            text_embedding: np.ndarray,
            image_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """유사한 기존 지식 검색"""

        # 텍스트 기반 검색
        text_matches = await self.embedding_manager.search_similar(
            text_embedding, top_k=5, threshold=0.6
        )

        # 이미지 기반 검색 (있는 경우)
        image_matches = []
        if image_embedding is not None:
            image_matches = await self.embedding_manager.search_similar_images(
                image_embedding, top_k=3, threshold=0.7
            )

        # 매칭 결과 종합
        if text_matches or image_matches:
            best_match = self._select_best_match(text_matches, image_matches)
            return {
                "found": True,
                "confidence": best_match.get("similarity", 0.0),
                "content": best_match.get("content", ""),
                "source": best_match.get("source", ""),
                "entity_id": best_match.get("entity_id", "")
            }

        return {"found": False, "confidence": 0.0}

    def _select_best_match(self, text_matches: List, image_matches: List) -> Dict:
        """텍스트와 이미지 매칭 결과에서 최적 선택"""
        all_matches = text_matches + image_matches

        if not all_matches:
            return {}

        # 유사도 기준 정렬
        sorted_matches = sorted(all_matches, key=lambda x: x.get("similarity", 0), reverse=True)
        return sorted_matches[0]

    async def _generate_learning_question(
            self,
            user_input: str,
            image: Optional[Image.Image],
            image_analysis: Optional[Dict],
            partial_knowledge: Dict
    ) -> Dict[str, Any]:
        """학습을 위한 질문 생성"""

        # 질문 타입 결정
        if image and image_analysis:
            question_type = "multimodal"
            question = self._generate_multimodal_question(user_input, image_analysis)
        else:
            question_type = "text"
            question = self._generate_text_question(user_input, partial_knowledge)

        # 학습 세션 ID 생성
        session_id = str(uuid.uuid4())

        return {
            "type": "learning_question",
            "session_id": session_id,
            "question": question,
            "question_type": question_type,
            "original_input": user_input,
            "context": {
                "has_image": image is not None,
                "image_analysis": image_analysis,
                "partial_knowledge": partial_knowledge
            },
            "confidence": partial_knowledge.get("confidence", 0.0)
        }

    def _generate_multimodal_question(self, text: str, image_analysis: Dict) -> str:
        """멀티모달 학습 질문 생성"""
        objects = image_analysis.get("objects", [])
        scene = image_analysis.get("scene_description", "")

        questions = [
            f"이 이미지에서 '{text}'와 관련된 부분을 설명해줄 수 있을까요?",
            f"사진 속 {', '.join(objects[:3])}에 대해 더 자세히 알려주세요.",
            f"이 상황에서 '{text}'는 어떤 의미인가요?"
        ]

        return questions[hash(text) % len(questions)]

    def _generate_text_question(self, text: str, partial_knowledge: Dict) -> str:
        """텍스트 학습 질문 생성"""
        if partial_knowledge.get("found"):
            return f"'{text}'에 대해 알고 있는 내용이 조금 있는데, 더 자세히 설명해주실 수 있나요?"
        else:
            questions = [
                f"'{text}'에 대해 처음 들어보는데, 어떤 의미인지 설명해주실 수 있나요?",
                f"'{text}'가 무엇인지 알려주시겠어요?",
                f"'{text}'에 대해 배우고 싶어요. 설명해주실 수 있을까요?"
            ]
            return questions[hash(text) % len(questions)]

    async def _provide_known_answer(
            self,
            user_input: str,
            knowledge: Dict,
            image_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """기존 지식 기반 응답 제공"""

        base_answer = knowledge.get("content", "")

        # 이미지가 있다면 시각적 정보와 결합
        if image_analysis:
            visual_context = f"\n\n이미지를 보니 {image_analysis.get('scene_description', '')}이네요."
            answer = base_answer + visual_context
        else:
            answer = base_answer

        return {
            "type": "known_answer",
            "answer": answer,
            "confidence": knowledge.get("confidence", 0.0),
            "source": knowledge.get("source", ""),
            "entity_id": knowledge.get("entity_id", "")
        }

    async def learn_from_user_response(
            self,
            session_id: str,
            user_response: str,
            original_context: Dict
    ) -> Dict[str, Any]:
        """사용자 응답으로부터 학습"""

        try:
            # 새로운 지식 엔티티 생성
            entity_id = await self._create_knowledge_entity(
                original_context["original_input"],
                user_response,
                original_context
            )

            # 임베딩 저장
            await self._store_knowledge_embeddings(
                entity_id,
                original_context["original_input"],
                user_response,
                original_context.get("image_analysis")
            )

            # 출처 기록
            self._record_provenance(
                entity_id,
                "user_teaching",
                f"사용자가 직접 설명: {user_response}"
            )

            return {
                "type": "learning_success",
                "message": "새로운 내용을 배웠어요! 감사합니다.",
                "entity_id": entity_id,
                "confidence": 1.0  # 사용자가 직접 가르쳐준 내용은 높은 확신도
            }

        except Exception as e:
            return {
                "type": "learning_error",
                "message": f"학습 중 오류가 발생했어요: {str(e)}",
                "confidence": 0.0
            }

    async def _create_knowledge_entity(
            self,
            concept: str,
            explanation: str,
            context: Dict
    ) -> str:
        """새로운 지식 엔티티 생성"""

        entity_id = hashlib.md5(f"{concept}_{explanation}".encode()).hexdigest()

        entity_data = {
            "concept": concept,
            "explanation": explanation,
            "created_at": datetime.now().isoformat(),
            "confidence": 1.0,
            "source": "user_teaching",
            "context": context,
            "tags": self._extract_tags(concept, explanation),
            "synonyms": self._extract_synonyms(concept, explanation)
        }

        self.entities["concepts"][entity_id] = entity_data
        self._save_entities()

        return entity_id

    async def _store_knowledge_embeddings(
            self,
            entity_id: str,
            concept: str,
            explanation: str,
            image_analysis: Optional[Dict]
    ):
        """지식 임베딩 저장"""

        # 텍스트 임베딩
        text_content = f"{concept}: {explanation}"
        text_embedding = await self.embedding_manager.get_text_embedding(text_content)

        await self.embedding_manager.store_embedding(
            entity_id,
            text_embedding,
            {
                "type": "knowledge",
                "concept": concept,
                "explanation": explanation
            }
        )

        # 이미지 관련 정보가 있다면 저장
        if image_analysis:
            self.entities["examples"][entity_id] = {
                "image_analysis": image_analysis,
                "associated_concept": concept
            }
            self._save_entities()

    def _extract_tags(self, concept: str, explanation: str) -> List[str]:
        """개념과 설명에서 태그 추출"""
        # 간단한 키워드 추출 (추후 더 정교한 NLP 적용 가능)
        import re

        text = f"{concept} {explanation}".lower()
        words = re.findall(r'\w+', text)

        # 의미있는 단어만 필터링 (길이 2 이상)
        meaningful_words = [w for w in words if len(w) >= 2]

        # 빈도 기반 태그 선택 (상위 5개)
        from collections import Counter
        common_words = Counter(meaningful_words).most_common(5)

        return [word for word, count in common_words]

    def _extract_synonyms(self, concept: str, explanation: str) -> List[str]:
        """동의어 추출"""
        # 간단한 동의어 패턴 매칭
        import re

        patterns = [
            r'또는\s+([가-힣]+)',
            r'즉\s+([가-힣]+)',
            r'다른\s+말로\s+([가-힣]+)'
        ]

        synonyms = []
        for pattern in patterns:
            matches = re.findall(pattern, explanation)
            synonyms.extend(matches)

        return synonyms

    def _record_provenance(self, entity_id: str, source_type: str, details: str):
        """출처 및 이력 기록"""
        provenance_entry = {
            "entity_id": entity_id,
            "timestamp": datetime.now().isoformat(),
            "source_type": source_type,
            "details": details,
            "action": "create"
        }

        self.provenance.append(provenance_entry)
        self._save_provenance()

    async def handle_correction_request(
            self,
            entity_id: str,
            correction: str,
            reason: str = ""
    ) -> Dict[str, Any]:
        """정정 요청 처리"""

        try:
            if entity_id not in self.entities["concepts"]:
                return {
                    "type": "correction_error",
                    "message": "해당 정보를 찾을 수 없어요."
                }

            # 기존 정보 백업
            original_data = self.entities["concepts"][entity_id].copy()

            # 정정 내용 적용
            self.entities["concepts"][entity_id]["explanation"] = correction
            self.entities["concepts"][entity_id]["last_updated"] = datetime.now().isoformat()
            self.entities["concepts"][entity_id]["correction_count"] = \
                self.entities["concepts"][entity_id].get("correction_count", 0) + 1

            # 정정 이력 기록
            self._record_correction(entity_id, original_data, correction, reason)

            # 임베딩 업데이트
            await self._update_embeddings_after_correction(entity_id, correction)

            self._save_entities()

            return {
                "type": "correction_success",
                "message": "정보를 수정했어요. 가르쳐주셔서 감사합니다!",
                "entity_id": entity_id
            }

        except Exception as e:
            return {
                "type": "correction_error",
                "message": f"정정 처리 중 오류가 발생했어요: {str(e)}"
            }

    def _record_correction(
            self,
            entity_id: str,
            original_data: Dict,
            correction: str,
            reason: str
    ):
        """정정 이력 기록"""
        correction_entry = {
            "entity_id": entity_id,
            "timestamp": datetime.now().isoformat(),
            "source_type": "user_correction",
            "action": "update",
            "original_explanation": original_data.get("explanation", ""),
            "new_explanation": correction,
            "reason": reason,
            "details": f"사용자 정정: {reason}" if reason else "사용자 정정"
        }

        self.provenance.append(correction_entry)
        self._save_provenance()

    async def _update_embeddings_after_correction(self, entity_id: str, new_explanation: str):
        """정정 후 임베딩 업데이트"""
        concept = self.entities["concepts"][entity_id]["concept"]
        new_content = f"{concept}: {new_explanation}"

        # 새로운 임베딩 생성
        new_embedding = await self.embedding_manager.get_text_embedding(new_content)

        # 기존 임베딩 업데이트
        await self.embedding_manager.update_embedding(entity_id, new_embedding)

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """학습 통계 조회"""
        total_entities = len(self.entities["concepts"])
        total_corrections = len([p for p in self.provenance if p.get("action") == "update"])

        # 최근 학습 내용
        recent_learning = sorted(
            self.entities["concepts"].values(),
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )[:5]

        return {
            "total_learned_concepts": total_entities,
            "total_corrections": total_corrections,
            "recent_learning": [
                {
                    "concept": item["concept"],
                    "created_at": item.get("created_at", ""),
                    "source": item.get("source", "")
                }
                for item in recent_learning
            ],
            "learning_sources": self._get_source_statistics()
        }

    def _get_source_statistics(self) -> Dict[str, int]:
        """학습 출처 통계"""
        sources = {}
        for entry in self.provenance:
            source_type = entry.get("source_type", "unknown")
            sources[source_type] = sources.get(source_type, 0) + 1
        return sources

    async def search_learned_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """학습된 지식 검색"""
        query_embedding = await self.embedding_manager.get_text_embedding(query)
        matches = await self.embedding_manager.search_similar(query_embedding, top_k=limit)

        results = []
        for match in matches:
            entity_id = match.get("entity_id", "")
            if entity_id in self.entities["concepts"]:
                entity_data = self.entities["concepts"][entity_id]
                results.append({
                    "concept": entity_data["concept"],
                    "explanation": entity_data["explanation"],
                    "similarity": match.get("similarity", 0.0),
                    "created_at": entity_data.get("created_at", ""),
                    "entity_id": entity_id
                })

        return results