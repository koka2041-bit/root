"""
문맥 생성 및 메모리 업데이트 관리
대화 기록과 학습된 정보를 바탕으로 맥락적 메모리 구축
"""
#메모리 빌더 (utils/memory_builder.py)

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import logging
from collections import defaultdict, Counter
import pickle
import hashlib

logger = logging.getLogger(__name__)


class ContextBuilder:
    """문맥 생성 클래스"""

    def __init__(self, data_dir: str = "data"):
        """
        문맥 빌더 초기화

        Args:
            data_dir (str): 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.memory_dir = os.path.join(data_dir, "memory")
        self.dialogues_file = os.path.join(data_dir, "dialogues.json")
        self.tag_map_file = os.path.join(data_dir, "tag_map.json")

        # 메모리 디렉토리 생성
        os.makedirs(self.memory_dir, exist_ok=True)

        self.context_cache = {}
        self.max_context_length = 2000  # 최대 문맥 길이

    def build_context(self, current_query: str, user_id: str = "default",
                      context_type: str = "conversational") -> Dict[str, Any]:
        """
        현재 쿼리에 대한 문맥 구성

        Args:
            current_query (str): 현재 사용자 질문
            user_id (str): 사용자 ID
            context_type (str): 문맥 타입 ('conversational', 'factual', 'visual')

        Returns:
            Dict[str, Any]: 구성된 문맥 정보
        """
        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(current_query, user_id, context_type)
            if cache_key in self.context_cache:
                cached_context = self.context_cache[cache_key]
                if self._is_cache_valid(cached_context):
                    return cached_context

            # 새로운 문맥 구성
            context = {
                "query": current_query,
                "user_id": user_id,
                "context_type": context_type,
                "timestamp": datetime.now().isoformat(),
                "relevant_dialogues": [],
                "related_memories": [],
                "extracted_entities": [],
                "contextual_tags": [],
                "confidence": 0.0
            }

            # 관련 대화 기록 검색
            context["relevant_dialogues"] = self._find_relevant_dialogues(
                current_query, user_id, limit=5
            )

            # 관련 메모리 검색
            context["related_memories"] = self._find_related_memories(
                current_query, context_type, limit=10
            )

            # 엔티티 추출
            context["extracted_entities"] = self._extract_context_entities(current_query)

            # 문맥적 태그 생성
            context["contextual_tags"] = self._generate_contextual_tags(
                current_query, context["relevant_dialogues"], context["related_memories"]
            )

            # 문맥 신뢰도 계산
            context["confidence"] = self._calculate_context_confidence(context)

            # 캐시 저장
            self.context_cache[cache_key] = context

            return context

        except Exception as e:
            logger.error(f"문맥 구성 실패: {str(e)}")
            return self._create_empty_context(current_query, user_id, context_type)

    def _find_relevant_dialogues(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """관련 대화 기록 검색"""
        try:
            if not os.path.exists(self.dialogues_file):
                return []

            with open(self.dialogues_file, 'r', encoding='utf-8') as f:
                all_dialogues = json.load(f)

            # 사용자별 대화 필터링
            user_dialogues = [d for d in all_dialogues if d.get("user_id") == user_id]

            # 최근 대화부터 검색 (시간 순서 중요)
            user_dialogues.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            # 관련성 점수 계산
            relevant_dialogues = []
            query_words = set(query.lower().split())

            for dialogue in user_dialogues:
                similarity = self._calculate_dialogue_similarity(dialogue, query_words)
                if similarity > 0.1:  # 최소 관련성 임계값
                    dialogue["relevance_score"] = similarity
                    relevant_dialogues.append(dialogue)

            # 관련성 순 정렬
            relevant_dialogues.sort(key=lambda x: x["relevance_score"], reverse=True)

            return relevant_dialogues[:limit]

        except Exception as e:
            logger.error(f"관련 대화 검색 실패: {str(e)}")
            return []

    def _find_related_memories(self, query: str, context_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """관련 메모리 검색"""
        try:
            memories = []

            # 엔티티 메모리 검색
            entities_file = os.path.join(self.memory_dir, "entities.json")
            if os.path.exists(entities_file):
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)

                entity_memories = self._search_entity_memories(query, entities_data, context_type)
                memories.extend(entity_memories)

            # 벡터 임베딩 기반 검색
            embeddings_file = os.path.join(self.memory_dir, "embeddings.vec")
            if os.path.exists(embeddings_file):
                vector_memories = self._search_vector_memories(query, embeddings_file)
                memories.extend(vector_memories)

            # 중복 제거 및 정렬
            unique_memories = self._deduplicate_memories(memories)
            unique_memories.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

            return unique_memories[:limit]

        except Exception as e:
            logger.error(f"관련 메모리 검색 실패: {str(e)}")
            return []

    def _search_entity_memories(self, query: str, entities_data: Dict[str, Any],
                                context_type: str) -> List[Dict[str, Any]]:
        """엔티티 기반 메모리 검색"""
        memories = []
        query_lower = query.lower()

        for entity_id, entity_info in entities_data.items():
            # 라벨 매칭
            labels = entity_info.get("labels", [])
            for label in labels:
                if label.lower() in query_lower:
                    memories.append({
                        "type": "entity",
                        "entity_id": entity_id,
                        "content": entity_info,
                        "matched_label": label,
                        "relevance_score": 0.8,
                        "context_type": context_type
                    })

            # 동의어 매칭
            synonyms = entity_info.get("synonyms", [])
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    memories.append({
                        "type": "entity",
                        "entity_id": entity_id,
                        "content": entity_info,
                        "matched_synonym": synonym,
                        "relevance_score": 0.6,
                        "context_type": context_type
                    })

        return memories

    def _search_vector_memories(self, query: str, embeddings_file: str) -> List[Dict[str, Any]]:
        """벡터 임베딩 기반 메모리 검색"""
        try:
            # 임베딩 유틸리티에서 유사도 검색 수행
            from utils.embedding_utils import EmbeddingManager

            embedding_manager = EmbeddingManager()
            similar_items = embedding_manager.find_similar(query, top_k=10)

            memories = []
            for item in similar_items:
                memories.append({
                    "type": "vector",
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "relevance_score": item.get("similarity", 0.0),
                    "embedding_id": item.get("id", "")
                })

            return memories

        except Exception as e:
            logger.error(f"벡터 메모리 검색 실패: {str(e)}")
            return []

    def _calculate_dialogue_similarity(self, dialogue: Dict[str, Any], query_words: set) -> float:
        """대화와 쿼리 간 유사도 계산"""
        try:
            # 사용자 메시지와 AI 응답 모두 고려
            user_message = dialogue.get("user_message", "").lower()
            ai_response = dialogue.get("ai_response", "").lower()

            combined_text = user_message + " " + ai_response
            dialogue_words = set(combined_text.split())

            # Jaccard 유사도
            intersection = query_words.intersection(dialogue_words)
            union = query_words.union(dialogue_words)

            if len(union) == 0:
                return 0.0

            jaccard_similarity = len(intersection) / len(union)

            # 시간 가중치 (최근 대화일수록 높은 가중치)
            time_weight = self._calculate_time_weight(dialogue.get("timestamp", ""))

            return jaccard_similarity * time_weight

        except Exception as e:
            logger.error(f"대화 유사도 계산 실패: {str(e)}")
            return 0.0

    def _calculate_time_weight(self, timestamp_str: str) -> float:
        """시간 가중치 계산 (최근일수록 높음)"""
        try:
            dialogue_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now()

            # 시간 차이 계산 (일 단위)
            time_diff = (current_time - dialogue_time.replace(tzinfo=None)).days

            # 지수적 감소 (7일 기준 반감기)
            weight = np.exp(-time_diff / 7.0)
            return max(0.1, weight)  # 최소 가중치 0.1

        except Exception as e:
            logger.error(f"시간 가중치 계산 실패: {str(e)}")
            return 0.5  # 기본값

    def _extract_context_entities(self, query: str) -> List[Dict[str, Any]]:
        """쿼리에서 문맥적 엔티티 추출"""
        entities = []

        # 간단한 NER (Named Entity Recognition)
        # 실제로는 더 정교한 NLP 모델 사용

        # 날짜 패턴
        date_patterns = [
            r'\d{4}년 \d{1,2}월 \d{1,2}일',
            r'\d{1,2}월 \d{1,2}일',
            r'오늘|어제|내일|이번주|다음주|지난주'
        ]

        for pattern in date_patterns:
            import re
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": "DATE",
                    "confidence": 0.9
                })

        # 숫자 패턴
        number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
        number_matches = re.findall(number_pattern, query)
        for match in number_matches:
            entities.append({
                "text": match,
                "type": "NUMBER",
                "confidence": 0.8
            })

        # 위치 패턴 (간단한 키워드 기반)
        location_keywords = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종']
        for keyword in location_keywords:
            if keyword in query:
                entities.append({
                    "text": keyword,
                    "type": "LOCATION",
                    "confidence": 0.7
                })

        return entities

    def _generate_contextual_tags(self, query: str, dialogues: List[Dict[str, Any]],
                                  memories: List[Dict[str, Any]]) -> List[str]:
        """문맥적 태그 생성"""
        tags = set()

        # 쿼리 기반 태그
        query_tags = self._extract_query_tags(query)
        tags.update(query_tags)

        # 대화 기반 태그
        for dialogue in dialogues:
            dialogue_tags = dialogue.get("tags", [])
            tags.update(dialogue_tags)

        # 메모리 기반 태그
        for memory in memories:
            memory_tags = memory.get("metadata", {}).get("tags", [])
            if isinstance(memory_tags, list):
                tags.update(memory_tags)

        return list(tags)

    def _extract_query_tags(self, query: str) -> List[str]:
        """쿼리에서 태그 추출"""
        tags = []

        # 감정 키워드
        emotion_keywords = {
            "기쁨": ["기쁜", "행복", "즐거운", "신나는"],
            "슬픔": ["슬픈", "우울", "아픈", "힘든"],
            "화남": ["화나는", "짜증", "분노", "열받는"],
            "놀람": ["놀라운", "깜짝", "신기한", "대단한"]
        }

        for emotion, keywords in emotion_keywords.items():
            if any(keyword in query for keyword in keywords):
                tags.append(f"감정_{emotion}")

        # 주제 키워드
        topic_keywords = {
            "음식": ["음식", "요리", "맛", "먹", "식당"],
            "여행": ["여행", "관광", "휴가", "여행지"],
            "학습": ["공부", "학습", "교육", "배우"],
            "건강": ["건강", "운동", "의료", "병원"]
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in query for keyword in keywords):
                tags.append(f"주제_{topic}")

        return tags

    def _calculate_context_confidence(self, context: Dict[str, Any]) -> float:
        """문맥 신뢰도 계산"""
        confidence = 0.0

        # 관련 대화 수에 따른 가중치
        dialogue_count = len(context["relevant_dialogues"])
        confidence += min(0.3, dialogue_count * 0.1)

        # 관련 메모리 수에 따른 가중치
        memory_count = len(context["related_memories"])
        confidence += min(0.3, memory_count * 0.03)

        # 추출된 엔티티 수에 따른 가중치
        entity_count = len(context["extracted_entities"])
        confidence += min(0.2, entity_count * 0.05)

        # 태그 수에 따른 가중치
        tag_count = len(context["contextual_tags"])
        confidence += min(0.2, tag_count * 0.02)

        return min(1.0, confidence)

    def _deduplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """메모리 중복 제거"""
        seen_contents = set()
        unique_memories = []

        for memory in memories:
            content_hash = hashlib.md5(str(memory.get("content", "")).encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_memories.append(memory)

        return unique_memories

    def _generate_cache_key(self, query: str, user_id: str, context_type: str) -> str:
        """캐시 키 생성"""
        key_string = f"{query}_{user_id}_{context_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cached_context: Dict[str, Any], validity_hours: int = 1) -> bool:
        """캐시 유효성 검사"""
        try:
            cache_time = datetime.fromisoformat(cached_context["timestamp"])
            current_time = datetime.now()

            time_diff = current_time - cache_time
            return time_diff.total_seconds() < validity_hours * 3600

        except Exception:
            return False

    def _create_empty_context(self, query: str, user_id: str, context_type: str) -> Dict[str, Any]:
        """빈 문맥 생성 (오류 시 폴백)"""
        return {
            "query": query,
            "user_id": user_id,
            "context_type": context_type,
            "timestamp": datetime.now().isoformat(),
            "relevant_dialogues": [],
            "related_memories": [],
            "extracted_entities": [],
            "contextual_tags": [],
            "confidence": 0.0,
            "error": "문맥 구성 중 오류 발생"
        }


class MemoryUpdater:
    """메모리 업데이트 관리 클래스"""

    def __init__(self, data_dir: str = "data"):
        """
        메모리 업데이터 초기화

        Args:
            data_dir (str): 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.memory_dir = os.path.join(data_dir, "memory")
        self.entities_file = os.path.join(self.memory_dir, "entities.json")
        self.provenance_file = os.path.join(self.memory_dir, "provenance.json")

        os.makedirs(self.memory_dir, exist_ok=True)

    def update_memory(self, memory_type: str, content: Dict[str, Any],
                      source: str = "user", metadata: Dict[str, Any] = None) -> bool:
        """
        메모리 업데이트

        Args:
            memory_type (str): 메모리 타입 ('entity', 'fact', 'experience')
            content (Dict[str, Any]): 메모리 내용
            source (str): 정보 출처
            metadata (Dict[str, Any]): 추가 메타데이터

        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if memory_type == "entity":
                return self._update_entity_memory(content, source, metadata)
            elif memory_type == "fact":
                return self._update_fact_memory(content, source, metadata)
            elif memory_type == "experience":
                return self._update_experience_memory(content, source, metadata)
            else:
                logger.error(f"지원하지 않는 메모리 타입: {memory_type}")
                return False

        except Exception as e:
            logger.error(f"메모리 업데이트 실패: {str(e)}")
            return False

    def _update_entity_memory(self, content: Dict[str, Any], source: str,
                              metadata: Dict[str, Any]) -> bool:
        """엔티티 메모리 업데이트"""
        try:
            # 기존 엔티티 데이터 로드
            entities_data = {}
            if os.path.exists(self.entities_file):
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)

            entity_id = content.get("id") or self._generate_entity_id(content)

            # 새 엔티티 정보 구성
            new_entity = {
                "id": entity_id,
                "labels": content.get("labels", []),
                "synonyms": content.get("synonyms", []),
                "description": content.get("description", ""),
                "properties": content.get("properties", {}),
                "examples": content.get("examples", []),
                "images": content.get("images", []),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "source": source,
                "confidence": content.get("confidence", 0.8),
                "metadata": metadata or {}
            }

            # 기존 엔티티와 병합 또는 새로 추가
            if entity_id in entities_data:
                existing_entity = entities_data[entity_id]
                merged_entity = self._merge_entities(existing_entity, new_entity)
                entities_data[entity_id] = merged_entity

                # 수정 이력 기록
                self._record_provenance("update", "entity", entity_id, source, metadata)
            else:
                entities_data[entity_id] = new_entity

                # 생성 이력 기록
                self._record_provenance("create", "entity", entity_id, source, metadata)

            # 파일 저장
            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)

            # 벡터 임베딩 업데이트
            self._update_embeddings("entity", entity_id, new_entity)

            return True

        except Exception as e:
            logger.error(f"엔티티 메모리 업데이트 실패: {str(e)}")
            return False

    def _update_fact_memory(self, content: Dict[str, Any], source: str,
                            metadata: Dict[str, Any]) -> bool:
        """사실 메모리 업데이트"""
        try:
            fact_id = self._generate_fact_id(content)

            # 사실 정보 구성
            fact_info = {
                "id": fact_id,
                "statement": content.get("statement", ""),
                "subject": content.get("subject", ""),
                "predicate": content.get("predicate", ""),
                "object": content.get("object", ""),
                "confidence": content.get("confidence", 0.7),
                "evidence": content.get("evidence", []),
                "created_at": datetime.now().isoformat(),
                "source": source,
                "metadata": metadata or {}
            }

            # 엔티티 데이터에 사실 추가
            entities_data = {}
            if os.path.exists(self.entities_file):
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)

            # 사실 저장 (별도 섹션)
            if "facts" not in entities_data:
                entities_data["facts"] = {}

            entities_data["facts"][fact_id] = fact_info

            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)

            # 이력 기록
            self._record_provenance("create", "fact", fact_id, source, metadata)

            # 임베딩 업데이트
            self._update_embeddings("fact", fact_id, fact_info)

            return True

        except Exception as e:
            logger.error(f"사실 메모리 업데이트 실패: {str(e)}")
            return False

    def _update_experience_memory(self, content: Dict[str, Any], source: str,
                                  metadata: Dict[str, Any]) -> bool:
        """경험 메모리 업데이트"""
        try:
            experience_id = self._generate_experience_id(content)

            # 경험 정보 구성
            experience_info = {
                "id": experience_id,
                "description": content.get("description", ""),
                "context": content.get("context", ""),
                "participants": content.get("participants", []),
                "location": content.get("location", ""),
                "timestamp": content.get("timestamp", datetime.now().isoformat()),
                "emotions": content.get("emotions", []),
                "outcomes": content.get("outcomes", []),
                "lessons_learned": content.get("lessons_learned", []),
                "created_at": datetime.now().isoformat(),
                "source": source,
                "metadata": metadata or {}
            }

            # 엔티티 데이터에 경험 추가
            entities_data = {}
            if os.path.exists(self.entities_file):
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)

            if "experiences" not in entities_data:
                entities_data["experiences"] = {}

            entities_data["experiences"][experience_id] = experience_info

            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)

            # 이력 기록
            self._record_provenance("create", "experience", experience_id, source, metadata)

            # 임베딩 업데이트
            self._update_embeddings("experience", experience_id, experience_info)

            return True

        except Exception as e:
            logger.error(f"경험 메모리 업데이트 실패: {str(e)}")
            return False

    def _merge_entities(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """기존 엔티티와 새 정보 병합"""
        merged = existing.copy()

        # 리스트 필드 병합 (중복 제거)
        list_fields = ["labels", "synonyms", "examples", "images"]
        for field in list_fields:
            existing_items = set(existing.get(field, []))
            new_items = set(new.get(field, []))
            merged[field] = list(existing_items.union(new_items))

        # 딕셔너리 필드 병합
        if "properties" in new:
            merged["properties"] = {**existing.get("properties", {}), **new["properties"]}

        if "metadata" in new:
            merged["metadata"] = {**existing.get("metadata", {}), **new["metadata"]}

        # 단일 값 필드 업데이트 (새 값이 있으면 덮어쓰기)
        single_fields = ["description", "confidence"]
        for field in single_fields:
            if field in new and new[field]:
                merged[field] = new[field]

        # 업데이트 시간 갱신
        merged["updated_at"] = datetime.now().isoformat()

        return merged

    def _record_provenance(self, action: str, memory_type: str, memory_id: str,
                           source: str, metadata: Dict[str, Any]) -> None:
        """메모리 수정 이력 기록"""
        try:
            # 기존 이력 로드
            provenance_data = []
            if os.path.exists(self.provenance_file):
                with open(self.provenance_file, 'r', encoding='utf-8') as f:
                    provenance_data = json.load(f)

            # 새 이력 항목 추가
            provenance_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,  # create, update, delete
                "memory_type": memory_type,
                "memory_id": memory_id,
                "source": source,
                "metadata": metadata or {},
                "user_id": metadata.get("user_id", "system") if metadata else "system"
            }

            provenance_data.append(provenance_entry)

            # 이력 파일 저장 (최근 1000개만 유지)
            if len(provenance_data) > 1000:
                provenance_data = provenance_data[-1000:]

            with open(self.provenance_file, 'w', encoding='utf-8') as f:
                json.dump(provenance_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"이력 기록 실패: {str(e)}")

    def _update_embeddings(self, memory_type: str, memory_id: str,
                           memory_content: Dict[str, Any]) -> None:
        """임베딩 업데이트"""
        try:
            from utils.embedding_utils import EmbeddingManager

            embedding_manager = EmbeddingManager()

            # 임베딩할 텍스트 추출
            text_content = self._extract_embedding_text(memory_content, memory_type)

            # 메타데이터 구성
            metadata = {
                "memory_type": memory_type,
                "memory_id": memory_id,
                "created_at": memory_content.get("created_at"),
                "source": memory_content.get("source"),
                "confidence": memory_content.get("confidence", 0.5)
            }

            # 임베딩 저장
            embedding_manager.store_embedding(
                content=text_content,
                metadata=metadata,
                item_id=f"{memory_type}_{memory_id}"
            )

        except Exception as e:
            logger.error(f"임베딩 업데이트 실패: {str(e)}")

    def _extract_embedding_text(self, content: Dict[str, Any], memory_type: str) -> str:
        """메모리 내용에서 임베딩용 텍스트 추출"""
        text_parts = []

        if memory_type == "entity":
            text_parts.extend(content.get("labels", []))
            text_parts.extend(content.get("synonyms", []))
            text_parts.append(content.get("description", ""))

        elif memory_type == "fact":
            text_parts.append(content.get("statement", ""))
            text_parts.append(
                f"{content.get('subject', '')} {content.get('predicate', '')} {content.get('object', '')}")

        elif memory_type == "experience":
            text_parts.append(content.get("description", ""))
            text_parts.append(content.get("context", ""))
            text_parts.extend(content.get("lessons_learned", []))

        # 빈 문자열 제거 후 결합
        text_parts = [part for part in text_parts if part and part.strip()]
        return " ".join(text_parts)

    def _generate_entity_id(self, content: Dict[str, Any]) -> str:
        """엔티티 ID 생성"""
        labels = content.get("labels", [])
        primary_label = labels[0] if labels else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"entity_{primary_label}_{timestamp}"

    def _generate_fact_id(self, content: Dict[str, Any]) -> str:
        """사실 ID 생성"""
        statement = content.get("statement", "")
        hash_value = hashlib.md5(statement.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"fact_{hash_value}_{timestamp}"

    def _generate_experience_id(self, content: Dict[str, Any]) -> str:
        """경험 ID 생성"""
        description = content.get("description", "")
        hash_value = hashlib.md5(description.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"experience_{hash_value}_{timestamp}"

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        try:
            stats = {
                "entities": 0,
                "facts": 0,
                "experiences": 0,
                "total_updates": 0,
                "last_update": None
            }

            # 엔티티 데이터 통계
            if os.path.exists(self.entities_file):
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)

                stats["entities"] = len([k for k in entities_data.keys()
                                         if not k.startswith(('facts', 'experiences'))])
                stats["facts"] = len(entities_data.get("facts", {}))
                stats["experiences"] = len(entities_data.get("experiences", {}))

            # 이력 데이터 통계
            if os.path.exists(self.provenance_file):
                with open(self.provenance_file, 'r', encoding='utf-8') as f:
                    provenance_data = json.load(f)

                stats["total_updates"] = len(provenance_data)
                if provenance_data:
                    stats["last_update"] = provenance_data[-1]["timestamp"]

            return stats

        except Exception as e:
            logger.error(f"메모리 통계 조회 실패: {str(e)}")
            return {"error": str(e)}

    def correct_memory(self, memory_type: str, memory_id: str,
                       corrections: Dict[str, Any], source: str = "user") -> bool:
        """메모리 정정"""
        try:
            if not os.path.exists(self.entities_file):
                return False

            with open(self.entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)

            memory_found = False

            # 메모리 타입에 따른 정정 처리
            if memory_type == "entity" and memory_id in entities_data:
                entities_data[memory_id].update(corrections)
                entities_data[memory_id]["updated_at"] = datetime.now().isoformat()
                memory_found = True

            elif memory_type == "fact" and "facts" in entities_data and memory_id in entities_data["facts"]:
                entities_data["facts"][memory_id].update(corrections)
                entities_data["facts"][memory_id]["updated_at"] = datetime.now().isoformat()
                memory_found = True

            elif memory_type == "experience" and "experiences" in entities_data and memory_id in entities_data[
                "experiences"]:
                entities_data["experiences"][memory_id].update(corrections)
                entities_data["experiences"][memory_id]["updated_at"] = datetime.now().isoformat()
                memory_found = True

            if memory_found:
                # 파일 저장
                with open(self.entities_file, 'w', encoding='utf-8') as f:
                    json.dump(entities_data, f, ensure_ascii=False, indent=2)

                # 정정 이력 기록
                self._record_provenance("correct", memory_type, memory_id, source,
                                        {"corrections": corrections})

                return True
            else:
                logger.warning(f"정정할 메모리를 찾을 수 없음: {memory_type}_{memory_id}")
                return False

        except Exception as e:
            logger.error(f"메모리 정정 실패: {str(e)}")
            return False


# 편의 함수들
def build_context_for_query(query: str, user_id: str = "default",
                            context_type: str = "conversational") -> Dict[str, Any]:
    """쿼리에 대한 문맥 구성 편의 함수"""
    builder = ContextBuilder()
    return builder.build_context(query, user_id, context_type)


def update_entity_memory(entity_data: Dict[str, Any], source: str = "user") -> bool:
    """엔티티 메모리 업데이트 편의 함수"""
    updater = MemoryUpdater()
    return updater.update_memory("entity", entity_data, source)


def update_fact_memory(fact_data: Dict[str, Any], source: str = "user") -> bool:
    """사실 메모리 업데이트 편의 함수"""
    updater = MemoryUpdater()
    return updater.update_memory("fact", fact_data, source)


def get_memory_statistics() -> Dict[str, Any]:
    """메모리 통계 조회 편의 함수"""
    updater = MemoryUpdater()
    return updater.get_memory_stats()