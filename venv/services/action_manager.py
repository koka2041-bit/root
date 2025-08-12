# services/action_manager.py
# 승인 기반 행동 관리 시스템 - 모든 도구 실행은 "제안 → 승인 → 실행" 절차

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from debug_logger import get_debugger

debugger = get_debugger()


class ActionType(Enum):
    """행동 유형"""
    WEB_SEARCH = "web_search"
    FILE_OPERATION = "file_operation"
    MEMORY_UPDATE = "memory_update"
    IMAGE_ANALYSIS = "image_analysis"
    LEARNING_UPDATE = "learning_update"
    SYSTEM_COMMAND = "system_command"
    DATA_EXPORT = "data_export"


class ActionStatus(Enum):
    """행동 상태"""
    PENDING = "pending"  # 승인 대기
    APPROVED = "approved"  # 승인됨
    REJECTED = "rejected"  # 거부됨
    EXECUTING = "executing"  # 실행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    EXPIRED = "expired"  # 만료됨


@dataclass
class ActionRequest:
    """행동 요청"""
    id: str
    action_type: ActionType
    title: str
    description: str
    parameters: Dict[str, Any]
    risks: List[str]
    benefits: List[str]
    estimated_time: str
    created_at: str
    expires_at: str
    status: ActionStatus = ActionStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ActionManager:
    """
    승인 기반 행동 관리자
    - 모든 중요한 행동에 대해 사용자 승인 요청
    - 행동 이력 관리 및 롤백 지원
    - 리스크 평가 및 안전 조치
    """

    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """행동 핸들러 등록"""
        self.action_handlers[action_type] = handler
        debugger.debug(f"행동 핸들러 등록: {action_type.value}", "ACTION_MANAGER")

    async def request_action_approval(
            self,
            action_type: ActionType,
            title: str,
            description: str,
            parameters: Dict[str, Any],
            risks: List[str] = None,
            benefits: List[str] = None,
            estimated_time: str = "즉시"
    ) -> str:
        """
        행동 승인 요청

        Args:
            action_type: 행동 유형
            title: 행동 제목
            description: 행동 설명
            parameters: 실행 매개변수
            risks: 위험 요소 목록
            benefits: 이점 목록
            estimated_time: 예상 소요 시간

        Returns:
            행동 요청 ID
        """
        # 대기 중인 행동 수 제한 확인
        if len(self.pending_actions) >= self.max_pending_actions:
            self._cleanup_old_actions()

        # 고유 ID 생성
        action_id = str(uuid.uuid4())

        # 만료 시간 설정
        expires_at = (datetime.now() + timedelta(hours=self.default_expiry_hours)).isoformat()

        # 행동 요청 생성
        action_request = ActionRequest(
            id=action_id,
            action_type=action_type,
            title=title,
            description=description,
            parameters=parameters,
            risks=risks or [],
            benefits=benefits or [],
            estimated_time=estimated_time,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
            status=ActionStatus.PENDING
        )

        # 대기 목록에 추가
        self.pending_actions[action_id] = action_request
        self._save_pending_actions()

        debugger.info(f"행동 승인 요청 생성: {title} (ID: {action_id})", "ACTION_MANAGER")

        return action_id

    def get_approval_message(self, action_id: str) -> Dict[str, Any]:
        """승인 요청 메시지 생성"""

        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]

        # 위험도 평가
        risk_level = self._assess_risk_level(action)

        return {
            "type": "approval_request",
            "action_id": action_id,
            "title": action.title,
            "description": action.description,
            "action_type": action.action_type.value,
            "estimated_time": action.estimated_time,
            "risk_level": risk_level,
            "risks": action.risks,
            "benefits": action.benefits,
            "expires_at": action.expires_at,
            "approval_options": [
                {"value": "approve", "label": "승인", "emoji": "✅"},
                {"value": "reject", "label": "거부", "emoji": "❌"},
                {"value": "details", "label": "자세히", "emoji": "📋"}
            ]
        }

    def _assess_risk_level(self, action: ActionRequest) -> str:
        """위험도 평가"""

        # 행동 유형별 기본 위험도
        type_risks = {
            ActionType.WEB_SEARCH: "low",
            ActionType.IMAGE_ANALYSIS: "low",
            ActionType.MEMORY_UPDATE: "medium",
            ActionType.LEARNING_UPDATE: "medium",
            ActionType.FILE_OPERATION: "high",
            ActionType.SYSTEM_COMMAND: "high",
            ActionType.DATA_EXPORT: "medium"
        }

        base_risk = type_risks.get(action.action_type, "medium")

        # 위험 요소 개수로 조정
        risk_count = len(action.risks)

        if risk_count == 0:
            return base_risk
        elif risk_count <= 2:
            return base_risk
        elif risk_count <= 4:
            # 위험도 한 단계 상승
            if base_risk == "low":
                return "medium"
            elif base_risk == "medium":
                return "high"
            else:
                return "critical"
        else:
            return "critical"

    async def approve_action(self, action_id: str, user_note: str = "") -> Dict[str, Any]:
        """행동 승인 및 실행"""

        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]

        # 만료 확인
        if datetime.now() > datetime.fromisoformat(action.expires_at):
            action.status = ActionStatus.EXPIRED
            self._save_pending_actions()
            return {"error": "이 요청은 만료되었습니다."}

        try:
            # 상태 업데이트
            action.status = ActionStatus.APPROVED
            self._save_pending_actions()

            debugger.info(f"행동 승인됨: {action.title} (ID: {action_id})", "ACTION_MANAGER")

            # 실행
            result = await self._execute_action(action, user_note)

            # 이력에 추가
            self._add_to_history(action, result, user_note)

            # 대기 목록에서 제거
            del self.pending_actions[action_id]
            self._save_pending_actions()

            return {
                "success": True,
                "message": f"'{action.title}' 실행이 완료되었습니다.",
                "result": result,
                "action_id": action_id
            }

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            self._save_pending_actions()

            debugger.error(f"행동 실행 실패: {action.title} - {e}", "ACTION_MANAGER")

            return {
                "success": False,
                "message": f"실행 중 오류가 발생했습니다: {str(e)}",
                "action_id": action_id
            }

    def reject_action(self, action_id: str, reason: str = "") -> Dict[str, Any]:
        """행동 거부"""

        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]
        action.status = ActionStatus.REJECTED

        # 이력에 추가
        self._add_to_history(action, {"rejected": True, "reason": reason})

        # 대기 목록에서 제거
        del self.pending_actions[action_id]
        self._save_pending_actions()

        debugger.info(f"행동 거부됨: {action.title} (ID: {action_id})", "ACTION_MANAGER")

        return {
            "success": True,
            "message": f"'{action.title}' 요청이 거부되었습니다.",
            "action_id": action_id
        }

    async def _execute_action(self, action: ActionRequest, user_note: str = "") -> Dict[str, Any]:
        """행동 실행"""

        action.status = ActionStatus.EXECUTING
        self._save_pending_actions()

        debugger.info(f"행동 실행 시작: {action.title}", "ACTION_MANAGER")

        # 핸들러 확인
        if action.action_type not in self.action_handlers:
            raise Exception(f"'{action.action_type.value}' 유형의 핸들러가 등록되지 않았습니다.")

        handler = self.action_handlers[action.action_type]

        # 실행
        start_time = datetime.now()

        try:
            result = await handler(action.parameters, user_note)

            action.status = ActionStatus.COMPLETED
            execution_time = (datetime.now() - start_time).total_seconds()

            debugger.success(f"행동 실행 완료: {action.title} ({execution_time:.2f}초)", "ACTION_MANAGER")

            return {
                "success": True,
                "data": result,
                "execution_time": execution_time,
                "completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)

            debugger.error(f"행동 실행 실패: {action.title} - {e}", "ACTION_MANAGER")
            raise e

    def _add_to_history(self, action: ActionRequest, result: Dict[str, Any], user_note: str = ""):
        """이력에 추가"""

        history_entry = {
            "action_id": action.id,
            "action_type": action.action_type.value,
            "title": action.title,
            "description": action.description,
            "parameters": action.parameters,
            "status": action.status.value,
            "result": result,
            "user_note": user_note,
            "created_at": action.created_at,
            "completed_at": datetime.now().isoformat(),
            "risks": action.risks,
            "benefits": action.benefits
        }

        self.action_history.append(history_entry)
        self._save_action_history()

        # 이력 크기 제한 (최대 1000개)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]  # 최근 500개만 유지
            self._save_action_history()

    def _cleanup_expired_actions(self):
        """만료된 행동 정리"""

        current_time = datetime.now()
        expired_actions = []

        for action_id, action in self.pending_actions.items():
            if current_time > datetime.fromisoformat(action.expires_at):
                action.status = ActionStatus.EXPIRED
                expired_actions.append(action_id)

        # 만료된 행동들을 이력으로 이동
        for action_id in expired_actions:
            action = self.pending_actions[action_id]
            self._add_to_history(action, {"expired": True})
            del self.pending_actions[action_id]

        if expired_actions:
            debugger.info(f"만료된 행동 {len(expired_actions)}개 정리", "ACTION_MANAGER")
            self._save_pending_actions()

    def _cleanup_old_actions(self):
        """오래된 대기 행동 정리"""

        if len(self.pending_actions) <= self.max_pending_actions:
            return

        # 생성 시간 순으로 정렬
        sorted_actions = sorted(
            self.pending_actions.items(),
            key=lambda x: x[1].created_at
        )

        # 오래된 것부터 제거
        remove_count = len(self.pending_actions) - self.max_pending_actions + 10

        for i in range(remove_count):
            action_id, action = sorted_actions[i]
            action.status = ActionStatus.EXPIRED
            self._add_to_history(action, {"auto_expired": True})
            del self.pending_actions[action_id]

        debugger.info(f"오래된 행동 {remove_count}개 정리", "ACTION_MANAGER")
        self._save_pending_actions()

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """대기 중인 행동 목록 조회"""

        self._cleanup_expired_actions()

        actions = []
        for action in self.pending_actions.values():
            actions.append({
                "id": action.id,
                "title": action.title,
                "action_type": action.action_type.value,
                "created_at": action.created_at,
                "expires_at": action.expires_at,
                "status": action.status.value,
                "risk_level": self._assess_risk_level(action)
            })

        # 생성 시간 역순 정렬
        actions.sort(key=lambda x: x["created_at"], reverse=True)

        return actions

    def get_action_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """행동 이력 조회"""

        # 최근 이력 반환
        recent_history = sorted(
            self.action_history,
            key=lambda x: x.get("completed_at", x.get("created_at", "")),
            reverse=True
        )

        return recent_history[:limit]

    def get_action_statistics(self) -> Dict[str, Any]:
        """행동 통계"""

        total_actions = len(self.action_history)

        # 상태별 통계
        status_counts = {}
        type_counts = {}

        for action in self.action_history:
            status = action.get("status", "unknown")
            action_type = action.get("action_type", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[action_type] = type_counts.get(action_type, 0) + 1

        # 성공률 계산
        completed = status_counts.get("completed", 0)
        failed = status_counts.get("failed", 0)
        success_rate = (completed / (completed + failed)) * 100 if (completed + failed) > 0 else 0

        return {
            "total_actions": total_actions,
            "pending_actions": len(self.pending_actions),
            "success_rate": round(success_rate, 2),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "most_common_action": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }

    async def quick_approve_safe_actions(self) -> Dict[str, Any]:
        """안전한 행동들 자동 승인 (선택적)"""

        safe_action_types = [ActionType.WEB_SEARCH, ActionType.IMAGE_ANALYSIS]
        auto_approved = []

        for action_id, action in list(self.pending_actions.items()):
            if (action.action_type in safe_action_types and
                    len(action.risks) == 0 and
                    self._assess_risk_level(action) == "low"):

                try:
                    result = await self.approve_action(action_id, "자동 승인 (안전한 행동)")
                    if result.get("success"):
                        auto_approved.append(action.title)
                except Exception as e:
                    debugger.error(f"자동 승인 실패: {action.title} - {e}", "ACTION_MANAGER")

        return {
            "auto_approved_count": len(auto_approved),
            "auto_approved_actions": auto_approved
        }

    def cancel_action(self, action_id: str) -> Dict[str, Any]:
        """행동 요청 취소"""

        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]

        if action.status != ActionStatus.PENDING:
            return {"error": f"이미 {action.status.value} 상태인 요청은 취소할 수 없습니다."}

        # 이력에 추가
        self._add_to_history(action, {"cancelled": True})

        # 대기 목록에서 제거
        del self.pending_actions[action_id]
        self._save_pending_actions()

        return {
            "success": True,
            "message": f"'{action.title}' 요청이 취소되었습니다."
        }

    def get_rollback_options(self, action_id: str) -> Dict[str, Any]:
        """롤백 옵션 조회"""

        # 이력에서 해당 행동 찾기
        target_action = None
        for action in self.action_history:
            if action.get("action_id") == action_id:
                target_action = action
                break

        if not target_action:
            return {"error": "해당 행동을 찾을 수 없습니다."}

        if target_action.get("status") != "completed":
            return {"error": "완료된 행동만 롤백할 수 있습니다."}

        # 행동 유형별 롤백 가능성 확인
        action_type = target_action.get("action_type")
        rollback_info = self._get_rollback_info(action_type, target_action)

        return {
            "action_id": action_id,
            "action_title": target_action.get("title"),
            "rollback_possible": rollback_info["possible"],
            "rollback_description": rollback_info["description"],
            "rollback_risks": rollback_info["risks"],
            "rollback_steps": rollback_info["steps"]
        }

    def _get_rollback_info(self, action_type: str, action_data: Dict) -> Dict[str, Any]:
        """롤백 정보 생성"""

        rollback_configs = {
            "memory_update": {
                "possible": True,
                "description": "메모리 변경사항을 이전 상태로 되돌립니다.",
                "risks": ["기존 학습 내용이 손실될 수 있습니다."],
                "steps": ["백업에서 이전 메모리 상태 복원", "변경 이력 업데이트"]
            },
            "learning_update": {
                "possible": True,
                "description": "학습된 내용을 제거하고 이전 상태로 되돌립니다.",
                "risks": ["학습된 지식이 완전히 삭제됩니다."],
                "steps": ["해당 엔티티 삭제", "임베딩 제거", "출처 이력 업데이트"]
            },
            "file_operation": {
                "possible": False,
                "description": "파일 작업은 롤백할 수 없습니다.",
                "risks": ["파일 시스템 변경은 되돌릴 수 없습니다."],
                "steps": []
            },
            "web_search": {
                "possible": False,
                "description": "웹 검색은 롤백이 필요하지 않습니다.",
                "risks": [],
                "steps": []
            }
        }

        return rollback_configs.get(action_type, {
            "possible": False,
            "description": "이 행동 유형은 롤백을 지원하지 않습니다.",
            "risks": [],
            "steps": []
        })

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = data_dir
        self.actions_file = os.path.join(data_dir, "pending_actions.json")
        self.history_file = os.path.join(data_dir, "action_history.json")

        # 필수 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)

        # 행동 함수 매핑
        self.action_handlers: Dict[ActionType, Callable] = {}

        # 승인 대기 행동들
        self.pending_actions: Dict[str, ActionRequest] = self._load_pending_actions()

        # 행동 이력
        self.action_history: List[Dict[str, Any]] = self._load_action_history()

        # 기본 설정
        self.default_expiry_hours = 24  # 24시간 후 만료
        self.max_pending_actions = 50  # 최대 대기 행동 수

    def _load_pending_actions(self) -> Dict[str, ActionRequest]:
        """대기 중인 행동 로드"""
        if os.path.exists(self.actions_file):
            try:
                with open(self.actions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                actions = {}
                for action_id, action_data in data.items():
                    # Enum 복원
                    action_data['action_type'] = ActionType(action_data['action_type'])
                    action_data['status'] = ActionStatus(action_data['status'])
                    actions[action_id] = ActionRequest(**action_data)

                return actions
            except Exception as e:
                debugger.error(f"대기 행동 로드 실패: {e}", "ACTION_MANAGER")
                return {}
        return {}

    def _load_action_history(self) -> List[Dict[str, Any]]:
        """행동 이력 로드"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                debugger.error(f"행동 이력 로드 실패: {e}", "ACTION_MANAGER")
                return []
        return []

    def _save_pending_actions(self):
        """대기 중인 행동 저장"""
        try:
            # 만료된 행동 정리
            self._cleanup_expired_actions()

            # Enum을 문자열로 변환하여 저장
            data = {}
            for action_id, action in self.pending_actions.items():
                action_dict = asdict(action)
                action_dict['action_type'] = action.action_type.value
                action_dict['status'] = action.status.value
                data[action_id] = action_dict

            with open(self.actions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            debugger.error(f"대기 행동 저장 실패: {e}", "ACTION_MANAGER")

    def _save_action_history(self):
        """행동 이력 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            debugger.error(f"행동 이력 저장 실패: {e}", "ACTION_MANAGER")

    def register_handler(self, action_type: ActionType, handler: Callable):
        """행동 핸들러 등록"""
        self.action_handlers[action_type] = handler
        debugger.info(f"핸들러 등록: {action_type.value}", "ACTION_MANAGER")

    def request_action(
            self,
            title: str,
            description: str,
            action_type: ActionType,
            parameters: Dict[str, Any],
            risks: Optional[List[str]] = None,
            benefits: Optional[List[str]] = None,
            estimated_time: str = "빠르게"
    ) -> str:
        """승인이 필요한 행동 요청 생성"""
        from uuid import uuid4
        from datetime import datetime, timedelta

        # 용량 제어
        self._cleanup_expired_actions()
        if len(self.pending_actions) >= self.max_pending_actions:
            self._cleanup_old_actions()

        action_id = str(uuid4())
        now = datetime.now()
        expires_at = (now + timedelta(hours=self.default_expiry_hours)).isoformat()

        action_request = ActionRequest(
            id=action_id,
            action_type=action_type,
            title=title,
            description=description,
            parameters=parameters or {},
            status=ActionStatus.PENDING,
            created_at=now.isoformat(),
            expires_at=expires_at,
            risks=risks or [],
            benefits=benefits or [],
            estimated_time=estimated_time
        )

        # 대기 목록에 추가
        self.pending_actions[action_id] = action_request
        self._save_pending_actions()

        debugger.info(f"행동 승인 요청 생성: {title} (ID: {action_id})", "ACTION_MANAGER")
        return action_id

    def get_approval_message(self, action_id: str) -> Dict[str, Any]:
        """승인 요청 메시지 생성"""
        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]
        risk_level = self._assess_risk_level(action)

        return {
            "type": "approval_request",
            "action_id": action_id,
            "title": action.title,
            "description": action.description,
            "action_type": action.action_type.value,
            "estimated_time": action.estimated_time,
            "risk_level": risk_level,
            "risks": action.risks,
            "benefits": action.benefits,
            "expires_at": action.expires_at,
            "approval_options": [
                {"value": "approve", "label": "승인", "emoji": "✅"},
                {"value": "reject", "label": "거부", "emoji": "❌"},
                {"value": "details", "label": "자세히", "emoji": "📋"}
            ]
        }

    def _assess_risk_level(self, action: ActionRequest) -> str:
        """위험도 평가"""
        type_risks = {
            ActionType.WEB_SEARCH: "low",
            ActionType.IMAGE_ANALYSIS: "low",
            ActionType.MEMORY_UPDATE: "medium",
            ActionType.LEARNING_UPDATE: "medium",
            ActionType.FILE_OPERATION: "high",
            ActionType.SYSTEM_COMMAND: "high",
            ActionType.DATA_EXPORT: "medium"
        }
        base_risk = type_risks.get(action.action_type, "medium")
        risk_count = len(action.risks)

        if risk_count == 0:
            return base_risk
        elif risk_count <= 2:
            return base_risk
        elif risk_count <= 4:
            if base_risk == "low":
                return "medium"
            elif base_risk == "medium":
                return "high"
            else:
                return "critical"
        else:
            return "critical"

    async def approve_action(self, action_id: str, user_note: str = "") -> Dict[str, Any]:
        """행동 승인 및 실행"""
        from datetime import datetime

        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]

        # 만료 확인
        if datetime.now() > datetime.fromisoformat(action.expires_at):
            action.status = ActionStatus.EXPIRED
            self._save_pending_actions()
            return {"error": "이 요청은 만료되었습니다."}

        try:
            # 상태 업데이트
            action.status = ActionStatus.APPROVED
            self._save_pending_actions()
            debugger.info(f"행동 승인됨: {action.title} (ID: {action_id})", "ACTION_MANAGER")

            # 실행
            result = await self._execute_action(action, user_note)

            # 이력에 추가
            self._add_to_history(action, result, user_note)

            # 대기 목록에서 제거
            del self.pending_actions[action_id]
            self._save_pending_actions()

            return {
                "success": True,
                "message": f"'{action.title}' 실행이 완료되었습니다.",
                "result": result,
                "action_id": action_id
            }

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            self._save_pending_actions()

            debugger.error(f"행동 실행 실패: {action.title} - {e}", "ACTION_MANAGER")
            return {
                "success": False,
                "message": f"실행 중 오류가 발생했습니다: {str(e)}",
                "action_id": action_id
            }

    def reject_action(self, action_id: str, reason: str = "") -> Dict[str, Any]:
        """행동 거부"""
        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]
        action.status = ActionStatus.REJECTED

        # 이력에 추가
        self._add_to_history(action, {"rejected": True, "reason": reason})

        # 대기 목록에서 제거
        del self.pending_actions[action_id]
        self._save_pending_actions()

        debugger.info(f"행동 거부됨: {action.title} (ID: {action_id})", "ACTION_MANAGER")
        return {
            "success": True,
            "message": f"'{action.title}' 요청이 거부되었습니다.",
            "action_id": action_id
        }

    async def _execute_action(self, action: ActionRequest, user_note: str = "") -> Dict[str, Any]:
        """행동 실행"""
        from datetime import datetime

        action.status = ActionStatus.EXECUTING
        self._save_pending_actions()
        debugger.info(f"행동 실행 시작: {action.title}", "ACTION_MANAGER")

        if action.action_type not in self.action_handlers:
            raise Exception(f"'{action.action_type.value}' 유형의 핸들러가 등록되지 않았습니다.")

        handler = self.action_handlers[action.action_type]
        start_time = datetime.now()

        try:
            result = await handler(action.parameters, user_note)

            action.status = ActionStatus.COMPLETED
            execution_time = (datetime.now() - start_time).total_seconds()
            debugger.success(f"행동 실행 완료: {action.title} ({execution_time:.2f}초)", "ACTION_MANAGER")

            return {
                "success": True,
                "data": result,
                "execution_time": execution_time,
                "completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            self._save_pending_actions()

            debugger.error(f"행동 실행 실패: {action.title} - {e}", "ACTION_MANAGER")
            return {
                "success": False,
                "message": f"실행 중 오류가 발생했습니다: {str(e)}",
                "action_id": action.id
            }

    def _add_to_history(self, action: ActionRequest, result: Dict[str, Any], user_note: str = ""):
        """이력에 추가"""
        from datetime import datetime

        history_entry = {
            "action_id": action.id,
            "action_type": action.action_type.value,
            "title": action.title,
            "description": action.description,
            "parameters": action.parameters,
            "status": action.status.value,
            "result": result,
            "user_note": user_note,
            "created_at": action.created_at,
            "completed_at": datetime.now().isoformat(),
            "risks": action.risks,
            "benefits": action.benefits
        }

        self.action_history.append(history_entry)
        self._save_action_history()

        # 이력 크기 제한 (최대 1000개)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]
            self._save_action_history()

    def _cleanup_expired_actions(self):
        """만료된 행동 정리"""
        from datetime import datetime

        current_time = datetime.now()
        expired_actions = []

        for action_id, action in self.pending_actions.items():
            if current_time > datetime.fromisoformat(action.expires_at):
                action.status = ActionStatus.EXPIRED
                expired_actions.append(action_id)

        for action_id in expired_actions:
            action = self.pending_actions[action_id]
            self._add_to_history(action, {"expired": True})
            del self.pending_actions[action_id]

        if expired_actions:
            debugger.info(f"만료된 행동 {len(expired_actions)}개 정리", "ACTION_MANAGER")
            self._save_pending_actions()

    def _cleanup_old_actions(self):
        """오래된 대기 행동 정리"""
        if len(self.pending_actions) <= self.max_pending_actions:
            return

        sorted_actions = sorted(
            self.pending_actions.items(),
            key=lambda x: x[1].created_at
        )

        remove_count = len(self.pending_actions) - self.max_pending_actions + 10
        for i in range(remove_count):
            action_id, action = sorted_actions[i]
            action.status = ActionStatus.EXPIRED
            self._add_to_history(action, {"auto_expired": True})
            del self.pending_actions[action_id]

        debugger.info(f"오래된 행동 {remove_count}개 정리", "ACTION_MANAGER")
        self._save_pending_actions()

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """대기 중인 행동 목록 조회"""
        self._cleanup_expired_actions()

        actions = []
        for action in self.pending_actions.values():
            actions.append({
                "id": action.id,
                "title": action.title,
                "action_type": action.action_type.value,
                "created_at": action.created_at,
                "expires_at": action.expires_at,
                "status": action.status.value,
                "risk_level": self._assess_risk_level(action)
            })

        actions.sort(key=lambda x: x["created_at"], reverse=True)
        return actions

    def get_action_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """행동 이력 조회"""
        recent_history = sorted(
            self.action_history,
            key=lambda x: x.get("completed_at", x.get("created_at", "")),
            reverse=True
        )
        return recent_history[:limit]

    def get_action_statistics(self) -> Dict[str, Any]:
        """행동 통계"""
        total_actions = len(self.action_history)

        status_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}

        for action in self.action_history:
            status = action.get("status", "unknown")
            action_type = action.get("action_type", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[action_type] = type_counts.get(action_type, 0) + 1

        completed = status_counts.get("completed", 0)
        failed = status_counts.get("failed", 0)
        success_rate = (completed / (completed + failed)) * 100 if (completed + failed) > 0 else 0

        return {
            "total_actions": total_actions,
            "pending_actions": len(self.pending_actions),
            "success_rate": round(success_rate, 2),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "most_common_action": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }

    async def quick_approve_safe_actions(self) -> Dict[str, Any]:
        """안전한 행동들 자동 승인 (선택적)"""
        safe_action_types = [ActionType.WEB_SEARCH, ActionType.IMAGE_ANALYSIS]
        auto_approved = []

        for action_id, action in list(self.pending_actions.items()):
            if (action.action_type in safe_action_types and
                    len(action.risks) == 0 and
                    self._assess_risk_level(action) == "low"):

                try:
                    result = await self.approve_action(action_id, "자동 승인 (안전한 행동)")
                    if result.get("success"):
                        auto_approved.append(action.title)
                except Exception as e:
                    debugger.error(f"자동 승인 실패: {action.title} - {e}", "ACTION_MANAGER")

        return {
            "auto_approved_count": len(auto_approved),
            "auto_approved_actions": auto_approved
        }

    def cancel_action(self, action_id: str) -> Dict[str, Any]:
        """행동 요청 취소"""
        if action_id not in self.pending_actions:
            return {"error": "해당 행동 요청을 찾을 수 없습니다."}

        action = self.pending_actions[action_id]

        if action.status != ActionStatus.PENDING:
            return {"error": f"이미 {action.status.value} 상태인 요청은 취소할 수 없습니다."}

        self._add_to_history(action, {"cancelled": True})
        del self.pending_actions[action_id]
        self._save_pending_actions()

        return {
            "success": True,
            "message": f"'{action.title}' 요청이 취소되었습니다."
        }

    def get_rollback_options(self, action_id: str) -> Dict[str, Any]:
        """롤백 옵션 조회"""
        target_action = None
        for action in self.action_history:
            if action.get("action_id") == action_id:
                target_action = action
                break

        if not target_action:
            return {"error": "해당 행동을 찾을 수 없습니다."}

        if target_action.get("status") != "completed":
            return {"error": "완료된 행동만 롤백할 수 있습니다."}

        action_type = target_action.get("action_type")
        rollback_info = self._get_rollback_info(action_type, target_action)

        return {
            "action_id": action_id,
            "action_title": target_action.get("title"),
            "rollback_possible": rollback_info["possible"],
            "rollback_description": rollback_info["description"],
            "rollback_risks": rollback_info["risks"],
            "rollback_steps": rollback_info["steps"]
        }

    def _get_rollback_info(self, action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """롤백 정보 생성"""
        rollback_configs = {
            "memory_update": {
                "possible": True,
                "description": "메모리 변경사항을 이전 상태로 되돌립니다.",
                "risks": ["기존 학습 내용이 손실될 수 있습니다."],
                "steps": ["백업에서 이전 메모리 상태 복원", "변경 이력 업데이트"]
            },
            "learning_update": {
                "possible": True,
                "description": "학습된 내용을 제거하고 이전 상태로 되돌립니다.",
                "risks": ["학습된 지식이 완전히 삭제됩니다."],
                "steps": ["해당 엔티티 삭제", "임베딩 제거", "출처 이력 업데이트"]
            },
            "file_operation": {
                "possible": False,
                "description": "파일 작업은 롤백할 수 없습니다.",
                "risks": ["파일 시스템 변경은 되돌릴 수 없습니다."],
                "steps": []
            },
            "web_search": {
                "possible": False,
                "description": "웹 검색은 롤백이 필요하지 않습니다.",
                "risks": [],
                "steps": []
            }
        }

        return rollback_configs.get(action_type, {
            "possible": False,
            "description": "이 행동 유형은 롤백을 지원하지 않습니다.",
            "risks": [],
            "steps": []
        })