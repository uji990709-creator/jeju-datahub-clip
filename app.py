"""
app.py

제주 소비동향 AI 차트 검색 웹 애플리케이션 (Streamlit)
- 부제: 2026년 1~9월 제주도 주요 소비동향 분석 보고서 기반
- 주요 개선:
  1. 가독성 높은 카드 제목 및 글자 크기 최적화
  2. 다크/라이트 모드 자동 지원 고대비 스타일링
  3. 모바일/PC 반응형 레이아웃 강화 (글씨-이미지 겹침 방지)
  4. 주요 분석 내용 2~3줄 기본 요약 + 상세 내용 펼치기(Expander)
  5. 연월, 페이지, PDF명, CLIP 유사도, 하이브리드 점수의 명확한 시각적 분리
  6. 원본 PDF 다운로드 및 모든 기존 검색/필터 기능 100% 보존
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# scripts 경로 추가
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "scripts"))

from search_hybrid import HybridSearchEngine

# 페이지 설정
st.set_page_config(
    page_title="제주 소비동향 AI 차트 검색",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 다크/라이트 모드 호환 고대비 반응형 CSS
st.markdown(
    """
    <style>
    /* 제목 및 부제 영역 */
    .main-title {
        font-size: 2.0rem;
        font-weight: 800;
        color: inherit;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.0rem;
        opacity: 0.85;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }

    /* 카드 헤더 및 제목 */
    .card-title-box {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 0.65rem;
    }
    .rank-tag {
        background-color: #2563EB;
        color: #FFFFFF !important;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.88rem;
        font-weight: 700;
        white-space: nowrap;
        line-height: 1.2;
        margin-top: 2px;
    }
    .card-title-text {
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.45;
        color: inherit;
        word-break: keep-all;
        overflow-wrap: break-word;
    }

    /* 점수 배지 (다크/라이트 모드 대응 반투명 고대비 스타일) */
    .score-badge-clip {
        background-color: rgba(99, 102, 241, 0.16);
        color: #6366F1;
        font-weight: 700;
        font-size: 0.86rem;
        padding: 5px 10px;
        border-radius: 6px;
        border: 1px solid rgba(99, 102, 241, 0.45);
        display: inline-flex;
        align-items: center;
        margin-right: 6px;
        margin-bottom: 6px;
        white-space: nowrap;
    }
    .score-badge-hybrid {
        background-color: rgba(16, 185, 129, 0.16);
        color: #10B981;
        font-weight: 700;
        font-size: 0.86rem;
        padding: 5px 10px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.45);
        display: inline-flex;
        align-items: center;
        margin-right: 6px;
        margin-bottom: 6px;
        white-space: nowrap;
    }
    @media (prefers-color-scheme: dark) {
        .score-badge-clip { color: #A5B4FC; }
        .score-badge-hybrid { color: #6EE7B7; }
    }

    /* 메타데이터 정보 태그 */
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .tag-pill {
        background-color: rgba(156, 163, 175, 0.14);
        color: inherit;
        border: 1px solid rgba(156, 163, 175, 0.32);
        font-size: 0.84rem;
        padding: 4px 9px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        line-height: 1.3;
    }

    /* 분석 내용 요약 박스 */
    .analysis-preview-box {
        background-color: rgba(156, 163, 175, 0.08);
        border-left: 3.5px solid #2563EB;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0 12px 0;
        line-height: 1.65;
        font-size: 0.92rem;
        word-break: keep-all;
        overflow-wrap: break-word;
    }

    /* 모바일 반응형 뷰 (너비 768px 이하) */
    @media (max-width: 768px) {
        .main-title { font-size: 1.6rem !important; }
        .sub-title { font-size: 0.92rem !important; margin-bottom: 1.0rem !important; }
        .card-title-text { font-size: 1.05rem !important; }
        .score-badge-clip, .score-badge-hybrid, .tag-pill {
            font-size: 0.8rem !important;
            padding: 3px 8px !important;
        }
        .analysis-preview-box {
            font-size: 0.88rem !important;
            padding: 8px 10px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="AI 검색 엔진(CLIP & 메타데이터 인덱스)을 로드하고 있습니다...")
def load_search_engine():
    """
    사전 구축된 CLIP 모델 및 임베딩 인덱스를 로드합니다.
    Streamlit 캐시를 사용하여 최초 1회만 로딩됩니다.
    """
    engine = HybridSearchEngine(
        embeddings_path="data/embeddings/image_embeddings.pt",
        metadata_path="data/embeddings/index_metadata.json",
        model_name="openai/clip-vit-base-patch32",
        w_meta=0.65,
        w_clip=0.35,
    )
    return engine


def format_clean_text(text: str) -> str:
    """
    PDF에서 추출된 텍스트의 줄바꿈과 여백을 가독성 있게 정리합니다.
    원문 텍스트의 어휘나 숫자는 일절 변경하지 않습니다.
    """
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def render_query_explanation(q_feat: Dict[str, Any]):
    """
    검색어에서 추출된 도메인 특징 및 동의어 매칭 이유를 시각화합니다.
    """
    reasons = []

    # 1. 연월
    if q_feat.get("target_ym"):
        reasons.append(f"📅 **연월 필터**: '{q_feat['target_ym']}' 보고서 우선 매칭")
    else:
        reasons.append("📅 **연월 필터**: 특정 연월 미지정 (전체 1~9월 대상)")

    # 2. 대상 집단
    group_map = {
        "제주도전체": "제주도 전체 (도민·관광객·법인 포괄)",
        "국내관광객": "국내관광객 (내국인)",
        "중국관광객": "중국관광객",
        "관광객": "관광객 전체 (국내 및 중국관광객 포괄)",
    }
    if q_feat.get("target_group"):
        reasons.append(f"👥 **대상 집단**: {group_map.get(q_feat['target_group'], q_feat['target_group'])}")

    # 3. 지표
    metrics = []
    if q_feat.get("is_total_amount"):
        metrics.append("전체 카드 소비총액/이용금액")
    if q_feat.get("is_per_capita"):
        metrics.append("1인당 1일 평균 카드소비")
    if q_feat.get("is_industry"):
        metrics.append("주요 업종별 소비 비교")
    if q_feat.get("matched_industries"):
        metrics.append(f"개별 업종 키워드({', '.join(q_feat['matched_industries'])})")
    if metrics:
        reasons.append(f"📈 **분석 지표/동의어**: {', '.join(metrics)}")

    # 4. 비교 시점
    if q_feat.get("target_horizon") == "전월":
        reasons.append("⏱️ **비교 시점**: 전월 대비(MoM) 증감")
    elif q_feat.get("target_horizon") == "전년동기":
        reasons.append("⏱️ **비교 시점**: 전년 동기 대비(YoY) 증감")

    with st.expander("💡 검색어 분석 및 매칭 근거 확인", expanded=False):
        for r in reasons:
            st.markdown(f"- {r}")
        st.caption("※ '소비', '사용금액', '이용금액' 등의 표현은 표준 지표명과 동의어로 자동 매핑됩니다.")


def main():
    # 1. 헤더 영역
    st.markdown('<div class="main-title">📊 제주 소비동향 AI 차트 검색</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">2026년 1~9월 제주도 주요 소비동향 분석 보고서(BC카드 빅데이터) 기반 텍스트-차트 검색 시스템</div>',
        unsafe_allow_html=True,
    )

    # 검색 엔진 로드
    engine = load_search_engine()

    # 2. 사이드바 제어 패널
    with st.sidebar:
        st.header("⚙️ 검색 옵션 설정")

        search_mode = st.radio(
            "검색 모드 선택",
            options=["하이브리드 검색 (권장)", "CLIP 단독 검색"],
            help="하이브리드 모드는 CLIP 비주얼 유사도와 한국어 도메인 메타데이터를 결합하여 정확도를 높입니다.",
        )

        top_k = st.slider(
            "최대 결과 개수",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="화면에 표시할 상위 검색 결과 이미지 수를 설정합니다.",
        )

        st.subheader("🎯 점수 필터 및 가중치")

        clip_threshold = st.slider(
            "CLIP 코사인 유사도 최소 임계값",
            min_value=-0.20,
            max_value=0.50,
            value=0.15,
            step=0.01,
            format="%.2f",
            help="CLIP 원본 코사인 유사도가 이 값 미만인 이미지는 결과에서 제외됩니다. 실제 유사도 분포는 약 0.20 ~ 0.35 수준입니다.",
        )

        if search_mode == "하이브리드 검색 (권장)":
            sort_by = st.selectbox(
                "결과 정렬 기준",
                options=["하이브리드 점수순 (기본)", "CLIP 코사인 유사도순"],
            )
            with st.expander("하이브리드 가중치 미세 조정", expanded=False):
                w_meta = st.slider("메타데이터 가중치 (w_meta)", 0.0, 1.0, 0.65, 0.05)
                w_clip = st.slider("CLIP 가중치 (w_clip)", 0.0, 1.0, 0.35, 0.05)
        else:
            sort_by = "CLIP 코사인 유사도순"
            w_meta, w_clip = 0.0, 1.0

        st.divider()
        st.markdown(
            """
            **📌 점수 해석 안내**
            - **CLIP 유사도**: 이미지와 질의 텍스트 간의 벡터 각도 유사도입니다. (이론 범위: -1.0 ~ +1.0)
            - **하이브리드 점수**: 키워드 적합도와 CLIP 유사도를 가중 합산한 0.0 ~ 1.0 종합 지표입니다.
            - ⚠️ **점수는 정답 확률(%)이나 모델 정확도를 의미하는 백분율이 아닙니다.**
            """
        )

    # 3. 검색 입력 영역
    st.markdown("##### 🔍 검색어 입력")

    # 추천 질의 버튼
    st.caption("추천 검색어를 클릭하여 바로 테스트할 수 있습니다:")
    col_q1, col_q2, col_q3 = st.columns(3)
    col_q4, col_q5, col_q6 = st.columns(3)

    query_preset = None
    if col_q1.button("제주도 전체 카드이용금액 비교", use_container_width=True):
        query_preset = "제주도 전체 카드이용금액 비교"
    if col_q2.button("국내관광객 업종별 소비 비교", use_container_width=True):
        query_preset = "국내관광객 업종별 소비 비교"
    if col_q3.button("중국관광객 전년 동기 카드 소비", use_container_width=True):
        query_preset = "중국관광객 전년 동기 카드 소비"
    if col_q4.button("2026년 5월 호텔업 소비 변화", use_container_width=True):
        query_preset = "2026년 5월 호텔업 소비 변화"
    if col_q5.button("제주 관광객 1인당 카드 소비", use_container_width=True):
        query_preset = "제주 관광객 1인당 카드 소비"
    if col_q6.button("2026년 8월 업종별 소비 증감", use_container_width=True):
        query_preset = "2026년 8월 업종별 소비 증감"

    # 세션 상태 관리
    if "search_query" not in st.session_state:
        st.session_state.search_query = "제주도 전체 카드이용금액 비교"

    if query_preset:
        st.session_state.search_query = query_preset

    search_input = st.text_input(
        "검색어 입력창",
        value=st.session_state.search_query,
        placeholder="예: 국내관광객 업종별 소비 비교, 2026년 5월 호텔업...",
        label_visibility="collapsed",
    )

    search_clicked = st.button("🔎 차트 검색 실행", type="primary", use_container_width=True)

    # 검색 실행 로직
    if search_clicked or search_input:
        query_str = search_input.strip()
        if not query_str:
            st.warning("검색어를 입력해 주세요.")
            return

        # 4. 검색 수행 (81개 이미지 대상 인덱스 질의)
        with st.spinner("이미지 및 메타데이터 검색 중..."):
            if search_mode == "하이브리드 검색 (권장)":
                raw_results = engine.search(
                    query=query_str,
                    top_k=81,  # 임계값 필터링을 위해 전체 후보군 산출 후 필터
                    w_meta=w_meta,
                    w_clip=w_clip,
                )
            else:
                raw_results = engine.search_clip_only(
                    query=query_str,
                    top_k=81,
                )
                for item in raw_results:
                    item["hybrid_score"] = None
                    item["meta_score"] = None

        # 5. 임계값 필터링 (CLIP 코사인 유사도 기준)
        filtered_results = [r for r in raw_results if r["clip_score"] >= clip_threshold]

        # 6. 정렬
        if sort_by == "CLIP 코사인 유사도순":
            filtered_results.sort(key=lambda x: x["clip_score"], reverse=True)
        else:
            filtered_results.sort(key=lambda x: x.get("hybrid_score", 0.0) or 0.0, reverse=True)

        final_results = filtered_results[:top_k]

        # 7. 검색어 매칭 근거 표시
        q_feat = engine.extract_query_features(query_str)
        render_query_explanation(q_feat)

        # 8. 검색 결과 표시
        st.subheader(f"📌 검색 결과 ({len(final_results)}건 표시 / 임계값 통과 {len(filtered_results)}건)")

        if not final_results:
            st.warning(
                f"⚠️ 검색 조건(CLIP 유사도 임계값: {clip_threshold:.2f})을 만족하는 결과가 없습니다.\n\n"
                "좌측 사이드바에서 임계값을 낮추거나 검색어를 변경해 보세요."
            )
            return

        # 결과 카드 렌더링
        for rank, res in enumerate(final_results, start=1):
            with st.container(border=True):
                col_img, col_info = st.columns([1.1, 1.4], gap="medium")

                # 이미지 표시 영역
                with col_img:
                    img_path = Path(res["image_path"])
                    if img_path.exists():
                        st.image(str(img_path), caption=f"차트 미리보기 ({res['image_filename']})", use_container_width=True)
                    else:
                        st.error(f"이미지 파일을 찾을 수 없습니다: {res['image_path']}")

                # 정보 및 메타데이터 영역
                with col_info:
                    # 1) 헤더: 순위 태그 + 컴팩트 제목 (글자 크기 적정화)
                    title_text = res["title"] if res["title"] else "(텍스트 레이어 미존재 / 9월 스캔본)"
                    st.markdown(
                        f"""
                        <div class="card-title-box">
                            <span class="rank-tag">#{rank}</span>
                            <span class="card-title-text">{title_text}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 2) 점수 표시 배지 (CLIP 유사도 vs 하이브리드 점수 명확한 분리)
                    clip_score_val = res["clip_score"]
                    hybrid_score_val = res.get("hybrid_score")

                    badge_html = f'<span class="score-badge-clip">📐 CLIP 코사인 유사도: {clip_score_val:.4f}</span>'
                    if hybrid_score_val is not None:
                        badge_html += f'<span class="score-badge-hybrid">✨ 하이브리드 관련성 점수: {hybrid_score_val:.4f}</span>'
                    st.markdown(f'<div style="margin-bottom: 8px;">{badge_html}</div>', unsafe_allow_html=True)

                    # 3) 상세 메타데이터 태그 (연월, 페이지 번호, PDF 파일명)
                    st.markdown(
                        f"""
                        <div class="tag-container">
                            <span class="tag-pill">📅 보고서 연월: <b>{res['report_year_month']}</b></span>
                            <span class="tag-pill">📄 해당 페이지: <b>{res['page_number']}p</b> (12p 중)</span>
                            <span class="tag-pill">📁 원본 문서: <b>{res['pdf_filename']}</b></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 4) 주요 분석 내용 요약 (기본 2~3줄 표시 + 상세 내용은 펼치기)
                    if res.get("description"):
                        clean_desc = format_clean_text(res["description"])
                        desc_lines = clean_desc.splitlines()

                        # 기본 2~3줄 미리보기 (최대 3줄)
                        preview_count = min(3, len(desc_lines))
                        preview_text = desc_lines[:preview_count]
                        remaining_text = desc_lines[preview_count:]

                        st.markdown("**📝 주요 분석 내용 요약:**")
                        preview_html = "<div class='analysis-preview-box'>" + "<br>".join([f"• {line.lstrip('•- ')}" for line in preview_text]) + "</div>"
                        st.markdown(preview_html, unsafe_allow_html=True)

                        # 4줄 이상 있거나 추가 주석이 있는 경우 펼쳐보기 제공
                        if remaining_text:
                            with st.expander(f"🔍 전체 분석 내용 및 주석 더보기 (+{len(remaining_text)}개 항목)", expanded=False):
                                for line in remaining_text:
                                    st.markdown(f"- {line.lstrip('•- ')}")
                    else:
                        st.info("ℹ️ 해당 페이지는 원본이 스캔 이미지이므로 본문 텍스트 요약이 제공되지 않습니다.")

                    # 5) 원본 PDF 파일 다운로드 버튼 연동
                    pdf_file_path = BASE_DIR / "data" / "reports" / res["pdf_filename"]
                    if pdf_file_path.exists():
                        with open(pdf_file_path, "rb") as f_pdf:
                            pdf_bytes = f_pdf.read()
                        st.download_button(
                            label=f"📥 원본 PDF 다운로드 ({res['report_year_month']} 보고서)",
                            data=pdf_bytes,
                            file_name=res["pdf_filename"],
                            mime="application/pdf",
                            key=f"dl_btn_{rank}_{res['id']}",
                            help="분석서 원본 PDF 전체 문서를 다운로드하여 확인할 수 있습니다.",
                        )


if __name__ == "__main__":
    main()
