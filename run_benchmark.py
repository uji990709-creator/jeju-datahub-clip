"""
run_benchmark.py
6개 질의에 대해 CLIP 단독 검색과 하이브리드 검색의 성능 및 Hit@5를 정밀 비교 평가합니다.
"""

import sys
from pathlib import Path
import json

sys.path.append("scripts")
from search_hybrid import HybridSearchEngine

# 평가 기준(Ground Truth) 사전 정의
# 각 질의별로 적합(Relevant)한 문서 판정 기준
EVAL_CRITERIA = {
    "제주도 전체 카드이용금액 비교": {
        "description": "제주도 전체의 총 카드이용금액을 다루는 페이지 (엄격: p2, 완화: p2, p3, p4)",
        "is_relevant_strict": lambda doc: doc["page_number"] == 2,
        "is_relevant_relaxed": lambda doc: doc["page_number"] in [2, 3, 4],
    },
    "국내관광객 업종별 소비 비교": {
        "description": "국내관광객 대상 주요 업종별 카드이용금액 비교 페이지 (p6, p7)",
        "is_relevant_strict": lambda doc: doc["page_number"] in [6, 7],
        "is_relevant_relaxed": lambda doc: doc["page_number"] in [6, 7],
    },
    "중국관광객 전년 동기 카드 소비": {
        "description": "중국관광객의 전년 동기 대비 카드 소비 비교 페이지 (엄격: p10, 완화: p8, p10)",
        "is_relevant_strict": lambda doc: doc["page_number"] == 10,
        "is_relevant_relaxed": lambda doc: doc["page_number"] in [8, 10],
    },
    "2026년 5월 호텔업 소비 변화": {
        "description": "2026년 5월 보고서 중 호텔업 소비 변화를 명시적으로 분석한 페이지 (엄격: 2026-05 p3, p6, p10 / 완화: 2026-05 업종페이지)",
        "is_relevant_strict": lambda doc: doc["report_year_month"] == "2026-05" and doc["page_number"] in [3, 6, 10],
        "is_relevant_relaxed": lambda doc: doc["report_year_month"] == "2026-05" and doc["page_number"] in [3, 4, 6, 7, 9, 10],
    },
    "제주 관광객 1인당 카드 소비": {
        "description": "관광객(국내/중국)의 1인당 카드 소비 분석 페이지 (엄격: p5, p8 / 완화: p2, p5, p8)",
        "is_relevant_strict": lambda doc: doc["page_number"] in [5, 8],
        "is_relevant_relaxed": lambda doc: doc["page_number"] in [2, 5, 8],
    },
    "2026년 8월 업종별 소비 증감": {
        "description": "2026년 8월 보고서의 주요 업종별 소비 증감율 비교 페이지 (2026-08 p3, p4, p6, p7, p9, p10)",
        "is_relevant_strict": lambda doc: doc["report_year_month"] == "2026-08" and doc["page_number"] in [3, 4, 6, 7, 9, 10],
        "is_relevant_relaxed": lambda doc: doc["report_year_month"] == "2026-08" and doc["page_number"] in [3, 4, 6, 7, 9, 10],
    },
}

def main():
    engine = HybridSearchEngine(
        embeddings_path="data/embeddings/image_embeddings.pt",
        metadata_path="data/embeddings/index_metadata.json",
        w_meta=0.65,
        w_clip=0.35,
    )

    queries = list(EVAL_CRITERIA.keys())
    results_summary = []

    for q in queries:
        crit = EVAL_CRITERIA[q]
        clip_res = engine.search_clip_only(query=q, top_k=5)
        hybrid_res = engine.search(query=q, top_k=5)

        # Hit@5 계산 (Top 5 결과 중 1개 이상의 관련 문서가 포함되었는지 여부)
        clip_hit_strict = any(crit["is_relevant_strict"](r) for r in clip_res)
        clip_hit_relaxed = any(crit["is_relevant_relaxed"](r) for r in clip_res)
        
        hybrid_hit_strict = any(crit["is_relevant_strict"](r) for r in hybrid_res)
        hybrid_hit_relaxed = any(crit["is_relevant_relaxed"](r) for r in hybrid_res)

        # Top-5 내 적합 문서 비율 (Precision@5)
        clip_p5_strict = sum(1 for r in clip_res if crit["is_relevant_strict"](r)) / 5.0
        hybrid_p5_strict = sum(1 for r in hybrid_res if crit["is_relevant_strict"](r)) / 5.0

        q_report = {
            "query": q,
            "description": crit["description"],
            "clip_results": clip_res,
            "hybrid_results": hybrid_res,
            "clip_hit_strict": clip_hit_strict,
            "clip_hit_relaxed": clip_hit_relaxed,
            "hybrid_hit_strict": hybrid_hit_strict,
            "hybrid_hit_relaxed": hybrid_hit_relaxed,
            "clip_p5_strict": clip_p5_strict,
            "hybrid_p5_strict": hybrid_p5_strict,
        }
        results_summary.append(q_report)

    # JSON 및 텍스트 요약 출력
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("🏆 6개 검색어 종합 Hit@5 및 Precision@5 비교 요약")
    print("=" * 90)
    print(f"{'검색어':<30} | {'CLIP Hit@5':<12} | {'하이브리드 Hit@5':<15} | {'CLIP P@5':<10} | {'하이브리드 P@5':<12}")
    print("-" * 90)
    for s in results_summary:
        c_hit = "O (1.0)" if s["clip_hit_strict"] else "X (0.0)"
        h_hit = "O (1.0)" if s["hybrid_hit_strict"] else "X (0.0)"
        print(f"{s['query']:<30} | {c_hit:<12} | {h_hit:<15} | {s['clip_p5_strict']*100:.0f}%{' ':<6} | {s['hybrid_p5_strict']*100:.0f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
