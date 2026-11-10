"""
scripts/preprocess_pdfs.py

제주데이터허브 CLIP 이미지 검색 시스템을 위한 PDF 데이터 전처리 파이프라인.
- 대상: 2026년 1월 ~ 9월 제주도 주요 소비동향 보고서 PDF (총 9개)
- 처리 범위: 각 PDF의 2~10페이지
- 해상도: 150 DPI 이상 PNG 이미지 렌더링
- 메타데이터: JSON (보고서 연월, PDF 파일명, 페이지 번호, 제목, 설명 텍스트, 이미지 경로, 차트 분리 정보 등)
- 텍스트 미추출 페이지(스캔본 등)는 오류 없이 '미추출' 플래그로 기록
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pymupdf


def extract_year_month(filename: str) -> Optional[str]:
    """
    PDF 파일명에서 보고서 연월(YYYY-MM)을 추출합니다.
    예:
      '2026-1_제주-AI데이터행정팀...' -> '2026-01'
      '2026년 8월_제주-AI데이터팀...' -> '2026-08'
    """
    m = re.search(r"(\d{4})[-_](\d{1,2})", filename)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월", filename)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    return None


def parse_page_text(page: pymupdf.Page) -> Dict[str, Any]:
    """
    PDF 페이지에서 제목과 본문 설명 텍스트를 추출합니다.
    텍스트 레이어가 없는 페이지(스캔 등)의 경우 내용을 임의로 꾸며내지 않고
    text_extracted=False로 처리합니다.
    """
    text = page.get_text()
    if not text or not text.strip():
        return {
            "text_extracted": False,
            "extraction_status": "no_text_extracted",
            "title": None,
            "description": None,
            "raw_text": "",
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. 제목 추출 ([...] 형태의 대괄호 타이틀 탐색)
    title = None
    for line in lines:
        if line.startswith("[") and "]" in line:
            title = line
            break

    # 2. 설명 텍스트 추출 (불릿 기호 '•' 또는 '- ' 등으로 시작하는 문장들)
    bullets = []
    current_bullet: List[str] = []

    for line in lines:
        if line.startswith("•") or line.startswith("- "):
            if current_bullet:
                bullets.append(" ".join(current_bullet))
                current_bullet = []
            current_bullet.append(line)
        elif current_bullet:
            # 주석 및 퍼센트 단독 행, 메타 문구는 불릿 종료 신호로 간주
            if (
                line.startswith("*")
                or line.startswith("※")
                or line.endswith("%")
                or "BC카드" in line
                or "도출한수치와" in line
            ):
                bullets.append(" ".join(current_bullet))
                current_bullet = []
            else:
                current_bullet.append(line)

    if current_bullet:
        bullets.append(" ".join(current_bullet))

    description = "\n".join(bullets) if bullets else None

    return {
        "text_extracted": True,
        "extraction_status": "success",
        "title": title,
        "description": description,
        "raw_text": text.strip(),
    }


def get_chart_split_info(page_num: int, width: int, height: int) -> List[Dict[str, Any]]:
    """
    페이지 구조에 따라 추후 차트별 분리가 가능하도록 바운딩 박스 정보를 구성합니다.
    - 2, 5, 8페이지: 좌우 2개의 비교 차트 (전체 이용금액 / 1인당 이용금액)
    - 3, 4, 6, 7, 9, 10페이지: 중앙 1개의 대형 주요 업종별 비교 차트
    좌표는 [x0, y0, x1, y1] 형식 (0.0~1.0 상대 좌표 및 픽셀 좌표 제공)
    """
    charts = []
    if page_num in (2, 5, 8):
        # 좌측 차트: 전체 카드 이용금액
        left_box_norm = [0.03, 0.12, 0.49, 0.78]
        charts.append({
            "chart_index": 1,
            "chart_type": "total_card_usage_comparison",
            "chart_name": "전체 카드이용금액 비교 차트",
            "normalized_bbox": left_box_norm,
            "pixel_bbox": [
                int(round(left_box_norm[0] * width)),
                int(round(left_box_norm[1] * height)),
                int(round(left_box_norm[2] * width)),
                int(round(left_box_norm[3] * height)),
            ],
        })
        # 우측 차트: 1인당 1일 평균 카드 이용금액
        right_box_norm = [0.51, 0.12, 0.97, 0.78]
        charts.append({
            "chart_index": 2,
            "chart_type": "per_capita_card_usage_comparison",
            "chart_name": "1인당 1일 평균 카드이용금액 비교 차트",
            "normalized_bbox": right_box_norm,
            "pixel_bbox": [
                int(round(right_box_norm[0] * width)),
                int(round(right_box_norm[1] * height)),
                int(round(right_box_norm[2] * width)),
                int(round(right_box_norm[3] * height)),
            ],
        })
    else:
        # 주요 업종별 카드 이용금액 차트
        box_norm = [0.03, 0.10, 0.97, 0.82]
        charts.append({
            "chart_index": 1,
            "chart_type": "industry_breakdown_chart",
            "chart_name": "주요 업종별 카드이용금액 비교 차트",
            "normalized_bbox": box_norm,
            "pixel_bbox": [
                int(round(box_norm[0] * width)),
                int(round(box_norm[1] * height)),
                int(round(box_norm[2] * width)),
                int(round(box_norm[3] * height)),
            ],
        })
    return charts


def extract_subcharts_from_image(
    image_path: str,
    chart_info_list: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    추후 차트별 분리가 필요할 때 호출할 수 있는 유틸리티 함수.
    생성된 전체 페이지 PNG에서 각 차트 영역을 잘라내어 별도 PNG로 저장합니다.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    target_dir = Path(output_dir) if output_dir else img_p.parent / "subcharts"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    doc = pymupdf.open(str(img_p))
    page = doc[0]

    # PNG 원본 픽셀 해상도 유지를 위한 Matrix 배율 계산
    # PyMuPDF에서 이미지 문서를 열 때 page.rect는 72 DPI 기준 가상 크기로 설정됨
    zoom = 150.0 / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    for item in chart_info_list:
        norm_bbox = item["normalized_bbox"]
        clip_rect = pymupdf.Rect(
            page.rect.width * norm_bbox[0],
            page.rect.height * norm_bbox[1],
            page.rect.width * norm_bbox[2],
            page.rect.height * norm_bbox[3],
        )
        pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
        out_name = f"{img_p.stem}_chart_{item['chart_index']}.png"
        out_path = target_dir / out_name
        pix.save(str(out_path))
        saved_paths.append(str(out_path).replace("\\", "/"))

    doc.close()
    return saved_paths


def process_reports(
    reports_dir: str = "data/reports",
    output_images_dir: str = "data/images",
    output_metadata_path: str = "data/metadata.json",
    target_pages: tuple = (2, 10),
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    전체 보고서 PDF 전처리 메인 함수.
    """
    reports_path = Path(reports_dir)
    images_path = Path(output_images_dir)
    images_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(reports_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {reports_dir}")

    total_pdfs = len(pdf_files)
    total_pngs = 0
    metadata_list: List[Dict[str, Any]] = []
    failed_pages: List[Dict[str, Any]] = []
    summary_by_pdf: List[Dict[str, Any]] = []

    print(f"=== 제주데이터허브 PDF 전처리 시작 (총 {total_pdfs}개 PDF) ===")

    for pdf_file in pdf_files:
        ym = extract_year_month(pdf_file.name)
        doc = pymupdf.open(str(pdf_file))
        page_count = len(doc)

        pdf_summary = {
            "pdf_filename": pdf_file.name,
            "year_month": ym,
            "total_pages": page_count,
            "processed_pages": [],
            "text_extractable_pages": 0,
            "failed_pages_count": 0,
        }

        print(f"\n[PDF 처리] {pdf_file.name} (총 {page_count}p, 연월: {ym})")

        # 대상 페이지: 2 ~ 10페이지 (1-based index)
        start_p, end_p = target_pages
        for page_num in range(start_p, end_p + 1):
            if page_num > page_count:
                print(f"  - 경고: {page_num}페이지가 문서 범위를 초과합니다.")
                continue

            page_idx = page_num - 1
            page = doc[page_idx]

            # 1. 텍스트 파싱
            text_info = parse_page_text(page)

            # 2. 150 DPI PNG 렌더링 및 저장
            pix = page.get_pixmap(dpi=dpi)
            img_filename = f"{ym}_page_{page_num:02d}.png"
            img_dest = images_path / img_filename
            pix.save(str(img_dest))
            total_pngs += 1

            # 3. 차트 분리 정보 계산
            chart_info = get_chart_split_info(page_num, pix.width, pix.height)

            # 4. 상대 경로 저장 (플랫폼 독립적 슬래시)
            rel_img_path = f"{output_images_dir}/{img_filename}".replace("\\", "/")

            entry = {
                "id": f"{ym}_p{page_num:02d}",
                "report_year_month": ym,
                "pdf_filename": pdf_file.name,
                "page_number": page_num,
                "title": text_info["title"],
                "description": text_info["description"],
                "image_path": rel_img_path,
                "image_width": pix.width,
                "image_height": pix.height,
                "dpi": dpi,
                "text_extracted": text_info["text_extracted"],
                "extraction_status": text_info["extraction_status"],
                "chart_split_info": chart_info,
            }

            metadata_list.append(entry)
            pdf_summary["processed_pages"].append(page_num)

            if text_info["text_extracted"]:
                pdf_summary["text_extractable_pages"] += 1
                status_str = f"텍스트 추출 성공 (제목: {text_info['title']})"
            else:
                pdf_summary["failed_pages_count"] += 1
                failed_entry = {
                    "pdf_filename": pdf_file.name,
                    "year_month": ym,
                    "page_number": page_num,
                    "reason": "스캔본 또는 텍스트 레이어 미존재 (no_text_extracted)",
                }
                failed_pages.append(failed_entry)
                status_str = "텍스트 추출 불가 (별도 미추출 플래그 기록)"

            print(f"  - Page {page_num:02d} -> {img_filename} [{status_str}]")

        summary_by_pdf.append(pdf_summary)
        doc.close()

    # JSON 메타데이터 파일 저장 (UTF-8, indent=2)
    meta_path = Path(output_metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"\n=== 전처리 완료 ===")
    print(f"- 처리된 PDF 개수: {total_pdfs}개")
    print(f"- 생성된 PNG 이미지 수: {total_pngs}개")
    print(f"- 메타데이터 항목 수: {len(metadata_list)}개")
    print(f"- 텍스트 미추출/실패 페이지 수: {len(failed_pages)}개")
    print(f"- 메타데이터 저장 위치: {meta_path.resolve()}")

    return {
        "total_pdfs": total_pdfs,
        "total_pngs": total_pngs,
        "metadata_count": len(metadata_list),
        "failed_pages": failed_pages,
        "summary_by_pdf": summary_by_pdf,
    }


def main():
    parser = argparse.ArgumentParser(description="제주 소비동향 PDF 데이터 전처리 스크립트")
    parser.add_argument("--reports-dir", default="data/reports", help="PDF 보고서 폴더 경로")
    parser.add_argument("--output-images-dir", default="data/images", help="PNG 이미지 저장 폴더")
    parser.add_argument("--output-metadata", default="data/metadata.json", help="메타데이터 JSON 저장 경로")
    parser.add_argument("--dpi", type=int, default=150, help="이미지 렌더링 DPI (기본: 150)")

    args = parser.parse_args()

    process_reports(
        reports_dir=args.reports_dir,
        output_images_dir=args.output_images_dir,
        output_metadata_path=args.output_metadata,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
