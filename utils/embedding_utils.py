# utils/embedding_utils.py
# 벡터 임베딩 관리 - 텍스트/이미지 임베딩 생성, 저장, 검색

import numpy as np
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

from debug_logger import get_debugger

debugger = get_debugger()


class EmbeddingManager:
    """
    임베딩 관리자
    - 텍스트/이미지 임베딩 생성
    - 벡터 데이터베이스 관리
    - 유사도 검색
    """

    def __init__(self, data_dir: str = "data/memory"):
        """
        Args:
            data_dir: 임베딩 데이터 저장 디렉토리
        """
        self.data_dir = data_dir
        self.embeddings_file = os.path.join(data_dir, "embeddings.vec")
        self.metadata_file = os.path.join(data_dir, "embedding_metadata.json")

        # 필수 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)

        # 임베딩 모델 초기화
        self.text_model = None
        self.image_model = None
        self._init_models()

        # 임베딩 데이터
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # 데이터 로드
        self._load_embeddings()
        self._load_metadata()

    def _init_models(self):
        """임베딩 모델 초기화"""
        try:
            # 텍스트 임베딩 모델 (다국어 지원)
            debugger.debug("텍스트 임베딩 모델 로딩 중...", "EMBEDDING")
            self.text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            debugger.success("텍스트 임베딩 모델 로딩 완료", "EMBEDDING")

        except Exception as e:
            debugger.error(f"임베딩 모델 로딩 실패: {e}", "EMBEDDING")
            self.text_model = None

        # 이미지 임베딩은 MiniCPM-V를 사용하므로 별도 초기화
        debugger.info("이미지 임베딩은 MiniCPM-V 모델 사용", "EMBEDDING")

    def _load_embeddings(self):
        """저장된 임베딩 로드"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                debugger.success(f"임베딩 {len(self.embeddings)}개 로드 완료", "EMBEDDING")
            except Exception as e:
                debugger.error(f"임베딩 로드 실패: {e}", "EMBEDDING")
                self.embeddings = {}
        else:
            self.embeddings = {}

    def _load_metadata(self):
        """임베딩 메타데이터 로드"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                debugger.success(f"메타데이터 {len(self.metadata)}개 로드 완료", "EMBEDDING")
            except Exception as e:
                debugger.error(f"메타데이터 로드 실패: {e}", "EMBEDDING")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_embeddings(self):
        """임베딩 저장"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            debugger.debug(f"임베딩 {len(self.embeddings)}개 저장 완료", "EMBEDDING")
        except Exception as e:
            debugger.error(f"임베딩 저장 실패: {e}", "EMBEDDING")

    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            debugger.debug(f"메타데이터 {len(self.metadata)}개 저장 완료", "EMBEDDING")
        except Exception as e:
            debugger.error(f"메타데이터 저장 실패: {e}", "EMBEDDING")

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """
        텍스트 임베딩 생성

        Args:
            text: 입력 텍스트

        Returns:
            임베딩 벡터
        """
        if not self.text_model:
            debugger.error("텍스트 임베딩 모델이 없습니다", "EMBEDDING")
            # 기본 해시 기반 벡터 반환 (fallback)
            return self._hash_to_vector(text)

        try:
            debugger.debug(f"텍스트 임베딩 생성: {text[:50]}...", "EMBEDDING")

            # SentenceTransformer로 임베딩 생성
            embedding = self.text_model.encode(text, convert_to_numpy=True)

            debugger.success(f"텍스트 임베딩 생성 완료: {embedding.shape}", "EMBEDDING")
            return embedding

        except Exception as e:
            debugger.error(f"텍스트 임베딩 생성 실패: {e}", "EMBEDDING")
            return self._hash_to_vector(text)

    async def get_image_embedding(self, image: Image.Image, model=None, tokenizer=None) -> np.ndarray:
        """
        이미지 임베딩 생성 (MiniCPM-V 사용)

        Args:
            image: PIL 이미지
            model: MiniCPM-V 모델 (선택사항)
            tokenizer: 토크나이저 (선택사항)

        Returns:
            이미지 임베딩 벡터
        """
        try:
            if model and tokenizer:
                debugger.debug(f"MiniCPM-V 이미지 임베딩 생성: {image.size}", "EMBEDDING")

                # MiniCPM-V로 이미지 피처 추출
                with torch.no_grad():
                    # 이미지를 모델 입력 형식으로 변환
                    inputs = tokenizer("이미지를 설명해주세요.", return_tensors="pt")

                    # 이미지 임베딩 추출 (내부 피처 사용)
                    # 실제 구현에서는 모델의 비전 인코더 출력을 사용
                    embedding = self._extract_vision_features(model, image)

                debugger.success(f"이미지 임베딩 생성 완료: {embedding.shape}", "EMBEDDING")
                return embedding

            else:
                # 기본 이미지 임베딩 (픽셀 히스토그램 기반)
                debugger.warning("MiniCPM-V 모델 없이 기본 이미지 임베딩 사용", "EMBEDDING")
                return self._basic_image_embedding(image)

        except Exception as e:
            debugger.error(f"이미지 임베딩 생성 실패: {e}", "EMBEDDING")
            return self._basic_image_embedding(image)

    def _extract_vision_features(self, model, image: Image.Image) -> np.ndarray:
        """MiniCPM-V에서 비전 피처 추출 (간소화)"""
        try:
            # 이미지를 텐서로 변환
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(image).unsqueeze(0)

            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            # 간단한 피처 추출 (실제로는 모델의 비전 인코더 사용)
            with torch.no_grad():
                # 평균 풀링으로 피처 압축
                features = torch.mean(image_tensor.view(1, -1), dim=1)
                return features.cpu().numpy().flatten()

        except Exception as e:
            debugger.warning(f"비전 피처 추출 실패, 기본 방법 사용: {e}", "EMBEDDING")
            return self._basic_image_embedding(image)

    def _basic_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        기본 이미지 임베딩
        - 모델이 없을 때 사용할 간단하고 견고한 특징 추출
        - 컬러 히스토그램 + 그레이스케일 히스토그램 + 채널 모멘트 + 엣지 분포
        Returns:
            고정 길이의 임베딩 벡터(np.ndarray)
        """
        try:
            # 1) 전처리: 크기 축소 + RGB 변환 + [0,1] 스케일링
            img = image.convert("RGB")
            img_small = img.resize((64, 64))
            arr = np.asarray(img_small, dtype=np.float32) / 255.0  # (64, 64, 3)

            # 2) 컬러 채널 히스토그램(각 32 bins)
            bins = 32
            hist_r = np.histogram(arr[:, :, 0], bins=bins, range=(0.0, 1.0))[0]
            hist_g = np.histogram(arr[:, :, 1], bins=bins, range=(0.0, 1.0))[0]
            hist_b = np.histogram(arr[:, :, 2], bins=bins, range=(0.0, 1.0))[0]

            # 3) 그레이스케일 히스토그램(32 bins)
            gray = (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2])
            hist_gray = np.histogram(gray, bins=bins, range=(0.0, 1.0))[0]

            # 4) 채널 모멘트(평균, 표준편차, 왜도)
            def _moments(x: np.ndarray) -> np.ndarray:
                mu = float(x.mean())
                sigma = float(x.std()) + 1e-8
                skew = float(((x - mu) ** 3).mean() / (sigma ** 3 + 1e-8))
                return np.array([mu, sigma, skew], dtype=np.float32)

            moments_r = _moments(arr[:, :, 0])
            moments_g = _moments(arr[:, :, 1])
            moments_b = _moments(arr[:, :, 2])

            # 5) 엣지 크기 분포(간단한 그래디언트 기반, 16 bins)
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            edge = np.sqrt(gx ** 2 + gy ** 2)
            edge_max = float(edge.max()) + 1e-8
            hist_edge = np.histogram(edge, bins=16, range=(0.0, edge_max))[0]

            # 6) 피처 결합 + L2 정규화
            feat = np.concatenate([
                hist_r, hist_g, hist_b,  # 32*3 = 96
                hist_gray,  # 32     = 32
                moments_r, moments_g, moments_b,  # 3*3  = 9
                hist_edge  # 16
            ]).astype(np.float32)  # 총 153 차원

            norm = np.linalg.norm(feat) + 1e-8
            feat = feat / norm

            debugger.debug(f"기본 이미지 임베딩 생성 완료(차원=153)", "EMBEDDING")
            return feat

        except Exception as e:
            debugger.warning(f"기본 이미지 임베딩 실패, 영벡터 반환: {e}", "EMBEDDING")
            # 실패 시 고정 길이의 영벡터 반환(동일 차원 유지)
            return np.zeros(153, dtype=np.float32)

    def _hash_to_vector(self, text: str, dim: int = 384) -> np.ndarray:
        """
        텍스트 해시 기반 임베딩(fallback)
        - SentenceTransformer를 사용할 수 없을 때 안정적으로 동작
        - 토큰 단위 해시 + 바이그램 해시 누적 후 L2 정규화
        Args:
            text: 원문
            dim: 출력 벡터 차원(텍스트 모델의 기본 384에 맞춤)
        """
        try:
            tokens = [t for t in text.lower().split() if t]
            vec = np.zeros(dim, dtype=np.float32)

            # 토큰 단일 해시
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h & 1) == 0 else -1.0
                mag = ((h >> 8) & 0xFFFF) / 65535.0  # 0~1
                vec[idx] += sign * (0.5 + 0.5 * mag)

            # 바이그램 해시(연속 토큰)
            if len(tokens) >= 2:
                for a, b in zip(tokens, tokens[1:]):
                    bigram = f"{a}_{b}"
                    h = int(hashlib.sha1(bigram.encode("utf-8")).hexdigest(), 16)
                    idx = h % dim
                    sign = 1.0 if (h & 1) == 0 else -1.0
                    mag = ((h >> 7) & 0xFFFF) / 65535.0
                    vec[idx] += sign * (0.25 + 0.75 * mag)

            # L2 정규화
            n = np.linalg.norm(vec) + 1e-8
            vec = vec / n

            debugger.debug(f"해시 임베딩 생성 완료(차원={dim})", "EMBEDDING")
            return vec

        except Exception as e:
            debugger.error(f"해시 임베딩 생성 실패: {e}", "EMBEDDING")
            # 최악의 경우라도 고정 차원 영벡터 반환
            return np.zeros(dim, dtype=np.float32)