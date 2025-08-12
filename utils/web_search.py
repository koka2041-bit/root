"""
승인 기반 웹 검색 유틸리티
사용자 승인을 받아 웹 검색을 수행하고 결과를 필터링
"""
#웹 검색 유틸리티 (utils/web_search.py)

import requests
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from urllib.parse import quote, urlparse
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
from typing import List, Dict, Any
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class WebSearchManager:
    """웹 검색 관리 클래스"""

    def __init__(self, api_keys_file: str = "api_keys.py"):
        """
        웹 검색 관리자 초기화

        Args:
            api_keys_file (str): API 키 파일 경로
        """
        self.search_history = []
        self.api_keys = self._load_api_keys(api_keys_file)
        self.search_engines = {
            "google": self._search_google,
            "bing": self._search_bing,
            "duckduckgo": self._search_duckduckgo
        }
        self.default_engine = "duckduckgo"  # API 키가 필요 없는 기본 엔진

    def _load_api_keys(self, api_keys_file: str) -> Dict[str, str]:
        """API 키 로딩"""
        try:
            if os.path.exists(api_keys_file):
                with open(api_keys_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # api_keys.py에서 키 추출
                api_keys = {}
                for line in content.split('\n'):
                    if 'GOOGLE_SEARCH_API_KEY' in line:
                        api_keys['google'] = line.split('=')[1].strip().strip('"\'')
                    elif 'BING_SEARCH_API_KEY' in line:
                        api_keys['bing'] = line.split('=')[1].strip().strip('"\'')

                return api_keys
            else:
                logger.warning("API 키 파일을 찾을 수 없습니다.")
                return {}

        except Exception as e:
            logger.error(f"API 키 로딩 실패: {str(e)}")
            return {}

    def suggest_search(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        검색 제안 생성

        Args:
            query (str): 검색 쿼리
            context (str): 검색 문맥

        Returns:
            Dict[str, Any]: 검색 제안 정보
        """
        try:
            # 검색 필요성 평가
            search_necessity = self._evaluate_search_necessity(query, context)

            # 검색 쿼리 최적화
            optimized_query = self._optimize_search_query(query)

            # 예상 검색 결과 타입 분석
            expected_types = self._analyze_expected_result_types(query)

            suggestion = {
                "original_query": query,
                "optimized_query": optimized_query,
                "context": context,
                "necessity_score": search_necessity,
                "expected_result_types": expected_types,
                "estimated_relevance": self._estimate_relevance(query),
                "search_engines": list(self.search_engines.keys()),
                "recommended_engine": self._recommend_search_engine(query),
                "safety_check": self._check_query_safety(query),
                "timestamp": datetime.now().isoformat()
            }

            return suggestion

        except Exception as e:
            logger.error(f"검색 제안 생성 실패: {str(e)}")
            return {"error": str(e)}

    async def perform_search(self, query: str, engine: str = None,
                             max_results: int = 10) -> Dict[str, Any]:
        """
        웹 검색 수행

        Args:
            query (str): 검색 쿼리
            engine (str): 검색 엔진 선택
            max_results (int): 최대 결과 수

        Returns:
            Dict[str, Any]: 검색 결과
        """
        try:
            if engine is None:
                engine = self.default_engine

            if engine not in self.search_engines:
                raise ValueError(f"지원하지 않는 검색 엔진: {engine}")

            # 검색 수행
            search_func = self.search_engines[engine]
            raw_results = await search_func(query, max_results)

            # 결과 필터링 및 정제
            filtered_results = self._filter_search_results(raw_results, query)

            # 검색 기록 저장
            search_record = {
                "query": query,
                "engine": engine,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(filtered_results.get("results", [])),
                "success": True
            }
            self.search_history.append(search_record)

            return {
                "query": query,
                "engine": engine,
                "results": filtered_results,
                "search_metadata": search_record
            }

        except Exception as e:
            logger.error(f"웹 검색 실패: {str(e)}")
            # 실패 기록 저장
            error_record = {
                "query": query,
                "engine": engine,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
            self.search_history.append(error_record)
            return {"error": str(e), "query": query}

    async def _search_google(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Google 검색"""
        if 'google' not in self.api_keys:
            raise ValueError("Google Search API 키가 설정되지 않았습니다.")

        # Google Custom Search API 사용
        api_key = self.api_keys['google']
        cx = "your_custom_search_engine_id"  # 실제 CSE ID로 교체 필요

        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(max_results, 10)
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_results(data)
                else:
                    raise Exception(f"Google 검색 API 오류: {response.status}")

    async def _search_bing(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Bing 검색"""
        if 'bing' not in self.api_keys:
            raise ValueError("Bing Search API 키가 설정되지 않았습니다.")

        api_key = self.api_keys['bing']
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "count": min(max_results, 50)}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_bing_results(data)
                else:
                    raise Exception(f"Bing 검색 API 오류: {response.status}")

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """DuckDuckGo 검색 (API 키 불필요)"""
        try:
            # DuckDuckGo Instant Answer API 사용
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_duckduckgo_results(data, query, max_results)
                    else:
                        raise Exception(f"DuckDuckGo 검색 오류: {response.status}")

        except Exception as e:
            logger.error(f"DuckDuckGo 검색 실패: {str(e)}")
            # 폴백: 간단한 웹 스크래핑
            return await self._fallback_search(query, max_results)

    async def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """폴백 검색 (웹 스크래핑)"""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_html_results(html, max_results)
                    else:
                        return []

        except Exception as e:
            logger.error(f"폴백 검색 실패: {str(e)}")
            return []

    def _parse_google_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Google 검색 결과 파싱"""
        results = []

        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "display_url": item.get("displayLink", ""),
                "source": "google"
            })

        return results

    def _parse_bing_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Bing 검색 결과 파싱"""
        results = []

        web_pages = data.get("webPages", {}).get("value", [])
        for item in web_pages:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "display_url": item.get("displayUrl", ""),
                "source": "bing"
            })

        return results

    def _parse_duckduckgo_results(self, data: Dict[str, Any], query: str, max_results: int) -> List[Dict[str, Any]]:
        """DuckDuckGo 검색 결과 파싱"""
        results = []

        # Instant Answer가 있는 경우
        if data.get("Abstract"):
            results.append({
                "title": f"DuckDuckGo 즉석 답변: {query}",
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("Abstract", ""),
                "display_url": "duckduckgo.com",
                "source": "duckduckgo_instant"
            })

        # Related Topics
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                    "display_url": "duckduckgo.com",
                    "source": "duckduckgo_related"
                })

            if len(results) >= max_results:
                break

        return results[:max_results]

    def _parse_html_results(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        """HTML 검색 결과 파싱"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []

            # DuckDuckGo HTML 결과 파싱
            result_divs = soup.find_all('div', class_='result')

            for div in result_divs[:max_results]:
                title_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('a', class_='result__snippet')

                if title_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "url": title_elem.get('href', ''),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                        "display_url": urlparse(title_elem.get('href', '')).netloc,
                        "source": "duckduckgo_html"
                    })

            return results

        except Exception as e:
            logger.error(f"HTML 파싱 실패: {str(e)}")
            return []

    def _filter_search_results(self, raw_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """검색 결과 필터링 및 정제"""
        try:
            filtered_results: List[Dict[str, Any]] = []

            # 신뢰 가능한 도메인(가산점)
            trusted_domains = [
                ".edu", ".gov", ".org", "wikipedia.org", "britannica.com",
                "reuters.com", "bbc.com", "cnn.com", "nytimes.com",
                "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov"
            ]

            for result in raw_results:
                # 1) 기본 필터링
                title = (result.get("title") or "").strip()
                url = (result.get("url") or "").strip()
                if not title or not url:
                    continue

                # 2) 안전성 검사
                if not self._is_safe_url(url):
                    continue

                # 3) 기본 관련도 점수(0~1 클램프)
                base = float(self._calculate_relevance_score(result, query))
                score = max(0.0, min(1.0, base))

                # 4) 도메인/프로토콜 기반 가산점
                domain = urlparse(url).netloc.lower()

                for trusted in trusted_domains:
                    if trusted in domain:
                        score += 0.3
                        break

                if url.startswith("https://"):
                    score += 0.1

                # 5) 스니펫 길이 가산점
                snippet = (result.get("snippet") or result.get("summary") or "").strip()
                if 50 <= len(snippet) <= 300:
                    score += 0.1

                # 6) 최종 스코어 클램프 및 누적
                score = max(0.0, min(1.0, score))
                result["relevance_score"] = score
                result["domain"] = domain
                filtered_results.append(result)

            # 7) 점수순 정렬 후 반환
            filtered_results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
            return {
                "results": filtered_results,
                "count": len(filtered_results)
            }

        except Exception as e:
            return {
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def _is_safe_url(self, url: str) -> bool:
        """기본 URL 안전성 검사"""
        if not url or len(url) > 2048:
            return False
        # 스킴 제한
        if not (url.startswith("http://") or url.startswith("https://")):
            return False
        # 위험 스킴/패턴 차단
        lowered = url.lower()
        if lowered.startswith(("javascript:", "data:", "file:", "vbscript:")):
            return False
        # 간단한 블랙리스트(필요 시 확장)
        bad_patterns = ["malware", "phishing", "ad.doubleclick.net"]
        return not any(p in lowered for p in bad_patterns)

    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """아주 단순한 질의-결과 관련도(0~1)"""
        q = (query or "").lower()
        title = (result.get("title") or "").lower()
        snippet = (result.get("snippet") or result.get("summary") or "").lower()

        if not q:
            return 0.2  # 질의 없으면 낮은 기본값

        # 토큰 기반 부분 일치 점수
        tokens = [t for t in re.findall(r"[가-힣a-z0-9]+", q) if t]
        if not tokens:
            return 0.2

        hit = 0
        for t in tokens:
            if t in title:
                hit += 2  # 제목은 가중치 2
            if t in snippet:
                hit += 1
        # 토큰 수 대비 정규화, 상한 1.0
        score = min(1.0, hit / max(1, len(tokens) * 2))
        # 제목 전체 포함 가산
        if any(title.startswith(t) for t in tokens):
            score = min(1.0, score + 0.1)
        return float(score)

    def _calculate_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """결과 품질 메트릭 계산"""
        if not results:
            return {"average_relevance": 0.0, "average_credibility": 0.0}

        relevance_scores = [r.get("relevance_score", 0.0) for r in results]
        credibility_scores = [r.get("credibility_score", 0.0) for r in results]

        return {
            "average_relevance": sum(relevance_scores) / len(relevance_scores),
            "average_credibility": sum(credibility_scores) / len(credibility_scores),
            "total_results": len(results),
            "high_quality_results": len([r for r in results if r.get("overall_score", 0.0) > 0.7])
        }

    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """검색 기록 반환"""
        return self.search_history[-limit:] if self.search_history else []

    def clear_search_history(self) -> bool:
        """검색 기록 삭제"""
        try:
            self.search_history.clear()
            return True
        except Exception as e:
            logger.error(f"검색 기록 삭제 실패: {str(e)}")
            return False


class SearchResultProcessor:
    """검색 결과 후처리 클래스"""

    @staticmethod
    def summarize_results(results: List[Dict[str, Any]], max_summary_length: int = 500) -> Dict[str, Any]:
        """
        검색 결과 요약

        Args:
            results (List[Dict]): 검색 결과 목록
            max_summary_length (int): 최대 요약 길이

        Returns:
            Dict[str, Any]: 요약된 검색 결과
        """
        try:
            if not results:
                return {"summary": "검색 결과가 없습니다.", "key_points": []}

            # 주요 정보 추출
            key_points = []
            all_snippets = []

            for result in results[:5]:  # 상위 5개 결과만 사용
                snippet = result.get("snippet", "")
                if snippet and len(snippet) > 20:
                    all_snippets.append(snippet)

                    # 핵심 문장 추출 (간단한 방법)
                    sentences = snippet.split('.')
                    for sentence in sentences:
                        if len(sentence.strip()) > 10:
                            key_points.append(sentence.strip())
                            break

            # 요약 생성 (단순 연결 방식)
            combined_text = " ".join(all_snippets)
            if len(combined_text) > max_summary_length:
                summary = combined_text[:max_summary_length] + "..."
            else:
                summary = combined_text

            return {
                "summary": summary,
                "key_points": key_points[:10],  # 상위 10개 핵심 포인트
                "source_count": len(results),
                "confidence": SearchResultProcessor._calculate_summary_confidence(results)
            }

        except Exception as e:
            logger.error(f"검색 결과 요약 실패: {str(e)}")
            return {"summary": "요약 생성에 실패했습니다.", "key_points": []}

    @staticmethod
    def _calculate_summary_confidence(results: List[Dict[str, Any]]) -> float:
        """요약의 신뢰도 계산"""
        if not results:
            return 0.0

        # 평균 신뢰도 점수 계산
        credibility_scores = [r.get("credibility_score", 0.5) for r in results]
        avg_credibility = sum(credibility_scores) / len(credibility_scores)

        # 결과 수에 따른 보정
        result_count_factor = min(1.0, len(results) / 5)

        return avg_credibility * result_count_factor

    @staticmethod
    def extract_facts(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과에서 사실 정보 추출"""
        facts = []

        for result in results:
            snippet = result.get("snippet", "")
            title = result.get("title", "")

            # 숫자나 날짜가 포함된 문장을 사실로 간주
            sentences = snippet.split('.')
            for sentence in sentences:
                if re.search(r'\d+', sentence) and len(sentence.strip()) > 15:
                    facts.append({
                        "fact": sentence.strip(),
                        "source": result.get("display_url", ""),
                        "url": result.get("url", ""),
                        "confidence": result.get("credibility_score", 0.5)
                    })

        # 신뢰도 순으로 정렬
        facts.sort(key=lambda x: x["confidence"], reverse=True)
        return facts[:10]

    @staticmethod
    def categorize_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """검색 결과 카테고리별 분류"""
        categories = {
            "news": [],
            "educational": [],
            "commercial": [],
            "reference": [],
            "other": []
        }

        for result in results:
            url = result.get("url", "").lower()
            domain = urlparse(url).netloc.lower()
            title = result.get("title", "").lower()

            # 카테고리 분류 로직
            if any(word in domain for word in ["news", "뉴스", "media"]) or "뉴스" in title:
                categories["news"].append(result)
            elif any(word in domain for word in [".edu", "wikipedia", "학습", "교육"]):
                categories["educational"].append(result)
            elif any(word in domain for word in ["shop", "mall", "buy", "쇼핑"]):
                categories["commercial"].append(result)
            elif any(word in domain for word in ["reference", "dictionary", "사전"]):
                categories["reference"].append(result)
            else:
                categories["other"].append(result)

        return categories


# 편의 함수들
async def search_web(query: str, engine: str = "duckduckgo", max_results: int = 10) -> Dict[str, Any]:
    """웹 검색 편의 함수"""
    search_manager = WebSearchManager()
    return await search_manager.perform_search(query, engine, max_results)


def suggest_web_search(query: str, context: str = "") -> Dict[str, Any]:
    """웹 검색 제안 편의 함수"""
    search_manager = WebSearchManager()
    return search_manager.suggest_search(query, context)


def summarize_search_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """검색 결과 요약 편의 함수"""
    return SearchResultProcessor.summarize_results(results)
    뢰도
    점수
    계산
    credibility_score = self._calculate_credibility_score(result)
    result["credibility_score"] = credibility_score

    # 종합 점수
    result["overall_score"] = (relevance_score + credibility_score) / 2

    filtered_results.append(result)

    # 종합 점수로 정렬


filtered_results.sort(key=lambda x: x["overall_score"], reverse=True)

return {
    "results": filtered_results,
    "total_results": len(filtered_results),
    "filtering_applied": True,
    "quality_metrics": self._calculate_quality_metrics(filtered_results)
}

except Exception as e:
logger.error(f"결과 필터링 실패: {str(e)}")
return {"results": raw_results, "filtering_applied": False}


def _evaluate_search_necessity(self, query: str, context: str) -> float:
    """검색 필요성 평가 (0~1 점수)"""
    necessity_score = 0.5  # 기본값

    # 시간에 민감한 정보 키워드
    time_sensitive_keywords = ["최신", "현재", "오늘", "지금", "실시간", "뉴스"]
    for keyword in time_sensitive_keywords:
        if keyword in query:
            necessity_score += 0.2
            break

    # 사실 확인이 필요한 키워드
    fact_check_keywords = ["통계", "데이터", "연구", "조사", "발표"]
    for keyword in fact_check_keywords:
        if keyword in query:
            necessity_score += 0.15
            break

    # 구체적 정보 요청
    specific_keywords = ["어떻게", "언제", "어디서", "누가", "얼마나"]
    for keyword in specific_keywords:
        if keyword in query:
            necessity_score += 0.1
            break

    return min(1.0, necessity_score)


def _optimize_search_query(self, query: str) -> str:
    """검색 쿼리 최적화"""
    # 불용어 제거
    stop_words = ["은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도"]
    words = query.split()

    optimized_words = [word for word in words if word not in stop_words]

    # 키워드 추출 및 정리
    optimized_query = " ".join(optimized_words)

    # 특수 문자 정리
    optimized_query = re.sub(r'[^\w\s가-힣]', ' ', optimized_query)
    optimized_query = re.sub(r'\s+', ' ', optimized_query).strip()

    return optimized_query


def _analyze_expected_result_types(self, query: str) -> List[str]:
    """예상 검색 결과 타입 분석"""
    result_types = []

    if any(word in query for word in ["뉴스", "소식", "발표", "보도"]):
        result_types.append("news")

    if any(word in query for word in ["사진", "이미지", "그림", "영상"]):
        result_types.append("media")

    if any(word in query for word in ["방법", "방식", "하는법", "튜토리얼"]):
        result_types.append("tutorial")

    if any(word in query for word in ["정의", "의미", "뜻", "개념"]):
        result_types.append("definition")

    if any(word in query for word in ["통계", "데이터", "수치", "그래프"]):
        result_types.append("statistics")

    return result_types if result_types else ["general"]


def _estimate_relevance(self, query: str) -> float:
    """검색 결과의 예상 관련성 추정"""
    # 쿼리의 구체성에 따른 관련성 추정
    word_count = len(query.split())

    if word_count >= 5:
        return 0.9  # 구체적인 쿼리
    elif word_count >= 3:
        return 0.7  # 보통 구체성
    else:
        return 0.5  # 일반적인 쿼리


def _recommend_search_engine(self, query: str) -> str:
    """쿼리에 따른 검색 엔진 추천"""
    if "뉴스" in query or "최신" in query:
        return "bing"  # 최신 정보에 강함
    elif "학술" in query or "논문" in query:
        return "google"  # 학술 정보에 강함
    else:
        return "duckduckgo"  # 개인정보 보호


def _check_query_safety(self, query: str) -> Dict[str, Any]:
    """쿼리 안전성 검사"""
    safety_check = {
        "is_safe": True,
        "warnings": [],
        "blocked_terms": []
    }

    # 위험한 키워드 검사
    dangerous_keywords = ["해킹", "불법", "폭력", "테러"]
    for keyword in dangerous_keywords:
        if keyword in query:
            safety_check["is_safe"] = False
            safety_check["warnings"].append(f"위험 키워드 감지: {keyword}")
            safety_check["blocked_terms"].append(keyword)

    return safety_check


def _is_safe_url(self, url: str) -> bool:
    """URL 안전성 검사"""
    try:
        parsed_url = urlparse(url)

        # 의심스러운 도메인 차단
        blocked_domains = ["malware.com", "phishing.com"]  # 실제 차단 목록으로 교체
        if parsed_url.netloc.lower() in blocked_domains:
            return False

        # HTTPS 선호 (HTTP도 허용)
        return parsed_url.scheme in ['http', 'https']

    except Exception:
        return False


def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
    """검색 결과의 관련성 점수 계산"""
    score = 0.0
    query_words = set(query.lower().split())

    # 제목에서 쿼리 단어 매칭
    title_words = set(result.get("title", "").lower().split())
    title_matches = len(query_words.intersection(title_words))
    score += (title_matches / len(query_words)) * 0.6

    # 스니펫에서 쿼리 단어 매칭
    snippet_words = set(result.get("snippet", "").lower().split())
    snippet_matches = len(query_words.intersection(snippet_words))
    score += (snippet_matches / len(query_words)) * 0.4

    return min(1.0, score)


def _calculate_credibility_score(self, result: Dict[str, Any]) -> float:
    """검색 결과의 신뢰도 점수 계산"""
    score = 0.5  # 기본 점수

    url = result.get("url", "")
    domain = urlparse(url).netloc.lower()

    # 신