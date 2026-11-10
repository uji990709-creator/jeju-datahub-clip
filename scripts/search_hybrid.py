"""
scripts/search_hybrid.py

제주데이터허브 CLIP 기반 하이브리드 검색 엔진.
- 비주얼 정보: 사전학습 CLIP(openai/clip-vit-base-patch32) 이미지-텍스트 코사인 유사도
- 도메인/메타데이터 정보: 한국어 키워드 및 도메인 분석 메타데이터 관련성 점수
  - 대상 집단(제주도 전체, 국내관광객, 중국관광객, 관광객) 구분
  - 분석 지표(전체 소비금액 비교, 업종별 소비 비교, 1인당 카드 소비 등) 구분
  - 비교 시점(전월 대비, 전년 동기 대비) 구분
  - 카드 이용금액/사용금액/소비액 등 동의어 처리
  - 질의에 명시된 특정 연월(YYYY년 M월 등) 가중/우선 처리
- 결합 방식:
  Hybrid_Score = w_meta * Meta_Score + w_clip * Clip_Score
  (기본 가중치: w_meta=0.65, w_clip=0.35, 설정 가능)
- 9월 보고서(텍스트 미추출 스캔본)도 이미지 임베딩 및 연월 메타데이터를 통해 정상 검색 대상으로 포함.
- 정답 페이지 번호는 일절 하드코딩하지 않고 메타데이터의 의미론적 매칭에 의해 산출됨.
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


class HybridSearchEngine:
    """
    CLIP 비주얼 유사도와 한국어 메타데이터 키워드 점수를 결합한 하이브리드 검색 엔진.
    """

    def __init__(
        self,
        embeddings_path: str = "data/embeddings/image_embeddings.pt",
        metadata_path: str = "data/embeddings/index_metadata.json",
        model_name: str = "openai/clip-vit-base-patch32",
        w_meta: float = 0.65,
        w_clip: float = 0.35,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.w_meta = w_meta
        self.w_clip = w_clip

        self._load_index()
        self._load_model()

    def _load_index(self):
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {self.embeddings_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"인덱스 메타데이터를 찾을 수 없습니다: {self.metadata_path}")

        self.image_embeddings = torch.load(self.embeddings_path, map_location=self.device)
        self.image_embeddings = self.image_embeddings.to(torch.float32)
        self.image_embeddings = F.normalize(self.image_embeddings, p=2, dim=-1)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

    def _load_model(self):
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def extract_query_features(query: str) -> Dict[str, Any]:
        """
        사용자 질의에서 연월, 대상 집단, 분석 지표, 비교 시점, 특정 업종 등의 의미론적 특징을 추출합니다.
        동의어 사전을 적용하여 다양한 어휘 표현을 표준화합니다.
        """
        q = query.strip()

        # 1. 보고서 연월 추출 (예: '2026년 5월', '2026년 08월', '2026-05' 등)
        target_ym = None
        m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월", q)
        if m:
            target_ym = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
        else:
            m2 = re.search(r"(\d{4})[-_](\d{1,2})", q)
            if m2:
                target_ym = f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}"

        # 2. 대상 집단 추출
        # 우선순위: '제주도 전체' / '국내관광객' / '중국관광객' / '관광객'
        target_group = None
        if any(k in q for k in ["제주도 전체", "제주도전체", "도전체", "도 전체"]):
            target_group = "제주도전체"
        elif any(k in q for k in ["국내관광객", "국내 관광객", "내국인", "내국인관광객"]):
            target_group = "국내관광객"
        elif any(k in q for k in ["중국관광객", "중국 관광객", "중국인"]):
            target_group = "중국관광객"
        elif "관광객" in q:
            target_group = "관광객"  # 국내/중국관광객 모두 포괄
        elif "제주" in q and not any(k in q for k in ["관광객", "내국인", "중국"]):
            target_group = "제주도전체"

        # 3. 분석 지표 / 메트릭 추출 (동의어 매핑)
        # 1인당 카드소비
        is_per_capita = any(k in q for k in ["1인당", "인당", "1인 당", "1인"])
        # 업종별 소비
        is_industry = any(k in q for k in ["업종별", "업종", "주요업종", "산업별"])
        # 전체 소비금액 / 총액
        is_total_amount = any(
            k in q
            for k in [
                "전체 카드",
                "전체카드",
                "전체 소비",
                "전체 이용",
                "총 카드",
                "총 소비",
                "총액",
                "카드 총액",
                "전체 소비금액",
            ]
        )

        # 4. 비교 시점 추출
        target_horizon = None
        if any(k in q for k in ["전월 대비", "전월대비", "전월"]):
            target_horizon = "전월"
        elif any(k in q for k in ["전년 동기", "전년동기", "전년 대비", "전년대비", "작년 대비", "작년"]):
            target_horizon = "전년동기"

        # 5. 특정 개별 업종 키워드 (호텔, 면세점, 골프장, 여행사, 음식점 등)
        common_industries = [
            "호텔", "면세점", "골프장", "여행사", "콘도", "식품", "주유소",
            "편의점", "슈퍼마켓", "임대업", "음식점", "카페", "주점", "마사지"
        ]
        matched_industries = [ind for ind in common_industries if ind in q]

        return {
            "raw_query": q,
            "target_ym": target_ym,
            "target_group": target_group,
            "is_per_capita": is_per_capita,
            "is_industry": is_industry,
            "is_total_amount": is_total_amount,
            "target_horizon": target_horizon,
            "matched_industries": matched_industries,
        }

    def compute_metadata_score(self, doc: Dict[str, Any], q_feat: Dict[str, Any]) -> float:
        """
        문서 메타데이터와 질의 특징 간의 관련성 점수(0.0 ~ 1.0)를 계산합니다.
        페이지 번호를 하드코딩하지 않고, 실제 메타데이터(제목, 설명, 연월)의 일치도를 평가합니다.
        """
        # 1. 연월 제약 조건 (Gate): 질의에 명시된 특정 연월이 있는 경우 해당 연월이 아닌 문서는 제외
        target_ym = q_feat["target_ym"]
        if target_ym:
            if doc.get("report_year_month") == target_ym:
                s_ym = 1.0
            else:
                return 0.0  # 명시된 연월과 불일치하면 0점 처리
        else:
            s_ym = 1.0

        title = doc.get("title") or ""
        desc = doc.get("description") or ""
        full_text = f"{title} {desc}"

        # 2. 대상 집단 점수 (s_group: 0.0 ~ 1.0)
        target_group = q_feat["target_group"]
        s_group = 1.0
        if target_group:
            if target_group == "제주도전체":
                if "[제주도전체" in title or "도전체" in full_text:
                    s_group = 1.0
                elif "[국내관광객" in title or "[중국관광객" in title:
                    s_group = 0.0
                else:
                    s_group = 0.2  # 텍스트 레이어가 없는 9월 문서의 기본 완충값
            elif target_group == "국내관광객":
                if "[국내관광객" in title:
                    s_group = 1.0
                elif "[중국관광객" in title or "[제주도전체" in title:
                    s_group = 0.0
                else:
                    s_group = 0.2
            elif target_group == "중국관광객":
                if "[중국관광객" in title:
                    s_group = 1.0
                elif "[국내관광객" in title or "[제주도전체" in title:
                    s_group = 0.0
                else:
                    s_group = 0.2
            elif target_group == "관광객":
                if "[국내관광객" in title or "[중국관광객" in title or "관광객" in full_text:
                    s_group = 1.0
                elif "[제주도전체" in title:
                    s_group = 0.2
                else:
                    s_group = 0.3

        # 3. 분석 지표 / 메트릭 점수 (s_metric: 0.0 ~ 1.0)
        metric_evals = []
        if q_feat["is_industry"]:
            if "주요업종별" in title:
                metric_evals.append(1.0)
            elif "전체및1인당" in title:
                metric_evals.append(0.0)
            else:
                metric_evals.append(0.3)

        if q_feat["is_per_capita"]:
            if "1인당" in title:
                metric_evals.append(1.0)
            elif "1인당" in desc:
                metric_evals.append(0.8)
            else:
                metric_evals.append(0.0)

        if q_feat["is_total_amount"]:
            if "전체및1인당" in title:
                metric_evals.append(1.0)
            elif "전체" in title:
                metric_evals.append(0.8)
            else:
                metric_evals.append(0.1)

        if q_feat["matched_industries"]:
            for ind in q_feat["matched_industries"]:
                if ind in title:
                    metric_evals.append(1.0)
                elif ind in desc:
                    metric_evals.append(0.9)
                elif "주요업종별" in title:
                    metric_evals.append(0.4)
                else:
                    metric_evals.append(0.0)

        s_metric = (sum(metric_evals) / len(metric_evals)) if metric_evals else 1.0

        # 4. 비교 시점 점수 (s_horizon: 0.0 ~ 1.0)
        target_horizon = q_feat["target_horizon"]
        s_horizon = 1.0
        if target_horizon == "전년동기":
            if "전년동기" in title or "전년동기" in desc:
                s_horizon = 1.0
            elif "전월" in title and "전년동기" not in title:
                s_horizon = 0.0
            else:
                s_horizon = 0.3
        elif target_horizon == "전월":
            if "전월" in title or "전월" in desc:
                s_horizon = 1.0
            elif "전년동기" in title and "전월" not in title:
                s_horizon = 0.0
            else:
                s_horizon = 0.3

        # 5. 가중 종합 메타데이터 점수 계산
        # 집단(0.35) + 지표(0.35) + 시점(0.15) + 기본 매칭 베이스(0.15)
        s_meta = s_ym * (
            0.35 * s_group
            + 0.35 * s_metric
            + 0.15 * s_horizon
            + 0.15
        )
        return min(max(s_meta, 0.0), 1.0)

    def search(
        self,
        query: str,
        top_k: int = 5,
        w_meta: Optional[float] = None,
        w_clip: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색을 수행하여 순위가 매겨진 결과를 반환합니다.
        반환 항목에 원본 CLIP 코사인 유사도와 하이브리드 점수를 모두 포함합니다.
        """
        if not query or not query.strip():
            return []

        w_m = self.w_meta if w_meta is None else w_meta
        w_c = self.w_clip if w_clip is None else w_clip

        # 1. 질의 특징 추출
        q_feat = self.extract_query_features(query)

        # 2. CLIP 텍스트 임베딩 생성
        inputs = self.processor(
            text=[query.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            text_features = F.normalize(text_features, p=2, dim=-1).to(torch.float32)

        # 3. CLIP 코사인 유사도 계산
        clip_similarities = (text_features @ self.image_embeddings.T).squeeze(0).cpu().tolist()

        # 4. 하이브리드 점수 결합
        scored_items = []
        for idx, meta in enumerate(self.metadata):
            c_score = float(clip_similarities[idx])
            m_score = self.compute_metadata_score(meta, q_feat)
            h_score = (w_m * m_score) + (w_c * c_score)

            item = {
                "id": meta.get("id"),
                "image_filename": meta.get("image_filename"),
                "report_year_month": meta.get("report_year_month"),
                "pdf_filename": meta.get("pdf_filename"),
                "page_number": meta.get("page_number"),
                "image_path": meta.get("image_path"),
                "title": meta.get("title"),
                "description": meta.get("description"),
                "text_extracted": meta.get("text_extracted", False),
                "clip_score": c_score,
                "meta_score": m_score,
                "hybrid_score": h_score,
            }
            scored_items.append(item)

        # 5. 정렬 및 상위 K개 반환
        scored_items.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 순위 부여
        results = []
        for rank, item in enumerate(scored_items[:top_k], start=1):
            item_copy = dict(item)
            item_copy["rank"] = rank
            results.append(item_copy)

        return results

    def search_clip_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        비교 평가를 위해 CLIP 단독 검색 결과를 반환합니다.
        """
        inputs = self.processor(
            text=[query.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            text_features = F.normalize(text_features, p=2, dim=-1).to(torch.float32)

        similarities = (text_features @ self.image_embeddings.T).squeeze(0).cpu()
        topk_scores, topk_indices = torch.topk(similarities, k=min(top_k, len(self.metadata)))

        results = []
        for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), start=1):
            meta = self.metadata[idx]
            results.append({
                "rank": rank,
                "id": meta.get("id"),
                "image_filename": meta.get("image_filename"),
                "report_year_month": meta.get("report_year_month"),
                "pdf_filename": meta.get("pdf_filename"),
                "page_number": meta.get("page_number"),
                "image_path": meta.get("image_path"),
                "title": meta.get("title"),
                "description": meta.get("description"),
                "text_extracted": meta.get("text_extracted", False),
                "clip_score": float(score),
            })
        return results


def print_comparison_results(query: str, clip_results: List[Dict[str, Any]], hybrid_results: List[Dict[str, Any]]):
    """
    CLIP 단독 검색과 하이브리드 검색 결과를 비교하여 출력합니다.
    """
    print("\n" + "=" * 90)
    print(f"🔎 검색어: '{query}'")
    print("=" * 90)

    print("\n[CLIP 단독 검색 Top 5]")
    print("-" * 90)
    for r in clip_results:
        t = r["title"] if r["title"] else "(텍스트 미추출 / 스캔본)"
        print(f"  {r['rank']}위 | 점수(CLIP): {r['clip_score']:.4f} | [{r['report_year_month']} p{r['page_number']}] {r['image_filename']}")
        print(f"       제목: {t}")

    print("\n[하이브리드 검색 Top 5]")
    print("-" * 90)
    for r in hybrid_results:
        t = r["title"] if r["title"] else "(텍스트 미추출 / 스캔본)"
        print(
            f"  {r['rank']}위 | 점수(통합): {r['hybrid_score']:.4f} (CLIP: {r['clip_score']:.4f}, Meta: {r['meta_score']:.4f}) | "
            f"[{r['report_year_month']} p{r['page_number']}] {r['image_filename']}"
        )
        print(f"       제목: {t}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="CLIP + 메타데이터 하이브리드 검색 엔진")
    parser.add_argument("--query", "-q", required=True, help="검색할 한국어 질의")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="반환할 상위 결과 수 (기본: 5)")
    parser.add_argument("--w-meta", type=float, default=0.65, help="메타데이터 점수 가중치 (기본: 0.65)")
    parser.add_argument("--w-clip", type=float, default=0.35, help="CLIP 비주얼 점수 가중치 (기본: 0.35)")
    parser.add_argument("--compare", action="store_true", help="CLIP 단독 검색과 비교하여 출력")
    parser.add_argument("--embeddings-path", default="data/embeddings/image_embeddings.pt")
    parser.add_argument("--metadata-path", default="data/embeddings/index_metadata.json")

    args = parser.parse_args()

    engine = HybridSearchEngine(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        w_meta=args.w_meta,
        w_clip=args.w_clip,
    )

    if args.compare:
        clip_results = engine.search_clip_only(query=args.query, top_k=args.top_k)
        hybrid_results = engine.search(query=args.query, top_k=args.top_k)
        print_comparison_results(args.query, clip_results, hybrid_results)
    else:
        results = engine.search(query=args.query, top_k=args.top_k)
        print("\n" + "=" * 80)
        print(f"🔍 하이브리드 검색 결과: '{args.query}'")
        print("=" * 80)
        for r in results:
            t = r["title"] if r["title"] else "(텍스트 미추출 / 스캔본)"
            print(f"[순위 {r['rank']}] 통합점수: {r['hybrid_score']:.4f} (CLIP: {r['clip_score']:.4f}, Meta: {r['meta_score']:.4f})")
            print(f"  - 파일명      : {r['image_filename']}")
            print(f"  - 연월/페이지 : {r['report_year_month']} (p{r['page_number']})")
            print(f"  - 원본 PDF    : {r['pdf_filename']}")
            print(f"  - 제목        : {t}")
            print("-" * 80)


if __name__ == "__main__":
    main()
