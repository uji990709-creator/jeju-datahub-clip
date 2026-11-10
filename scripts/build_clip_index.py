"""
scripts/build_clip_index.py

제주데이터허브 CLIP 이미지 검색 시스템: 이미지 임베딩 인덱스 구축 스크립트.
- 사전학습 모델: openai/clip-vit-base-patch32
- 입력: data/metadata.json 및 data/images/*.png (총 81개)
- 처리 방식: torch.inference_mode() 및 배치 단위 처리로 메모리 최적화
- 정규화: L2 정규화 (Cosine 유사도 계산을 위함)
- 출력: data/embeddings/image_embeddings.pt 및 data/embeddings/index_metadata.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def build_clip_index(
    metadata_path: str = "data/metadata.json",
    output_dir: str = "data/embeddings",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 8,
    device: str = None,
) -> Dict[str, Any]:
    """
    data/images에 존재하는 81개 이미지에 대해 CLIP 임베딩을 생성하고 L2 정규화하여 저장합니다.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== CLIP 이미지 임베딩 인덱스 구축 시작 ===")
    print(f"- 사용 디바이스: {device}")
    print(f"- 모델 식별자: {model_name}")
    print(f"- 배치 크기: {batch_size}")

    # 1. 메타데이터 로드
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"메타데이터 파일이 존재하지 않습니다: {metadata_path}")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata_list: List[Dict[str, Any]] = json.load(f)

    total_images = len(metadata_list)
    print(f"- 로드된 메타데이터 항목 수: {total_images}개")

    # 2. 모델 및 프로세서 로드
    print(f"- CLIP 모델 및 프로세서 로드 중...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 3. 배치 단위로 이미지 임베딩 생성
    all_embeddings_list: List[torch.Tensor] = []
    index_records: List[Dict[str, Any]] = []

    print(f"- 이미지 임베딩 생성 진행 (총 {total_images}개)...")

    for i in range(0, total_images, batch_size):
        batch_meta = metadata_list[i : i + batch_size]
        batch_images = []
        batch_valid_indices = []

        for offset, item in enumerate(batch_meta):
            img_path = Path(item["image_path"])
            if not img_path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

            img = Image.open(img_path).convert("RGB")
            batch_images.append(img)
            batch_valid_indices.append(i + offset)

        # Preprocessing & Model Inference
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = model.get_image_features(**inputs)
            if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            # L2 정규화
            image_features = F.normalize(image_features, p=2, dim=-1)

        # CPU 텐서로 이동하여 리스트에 추가
        all_embeddings_list.append(image_features.cpu())

        # 인덱스 레코드 매핑 생성
        for offset, item in enumerate(batch_meta):
            idx = i + offset
            record = {
                "embedding_index": idx,
                "id": item.get("id"),
                "report_year_month": item.get("report_year_month"),
                "pdf_filename": item.get("pdf_filename"),
                "page_number": item.get("page_number"),
                "image_path": item.get("image_path"),
                "image_filename": Path(item.get("image_path")).name,
                "title": item.get("title"),
                "description": item.get("description"),
                "text_extracted": item.get("text_extracted", False),
                "extraction_status": item.get("extraction_status", "unknown"),
            }
            index_records.append(record)

        print(f"  진행률: [{min(i + batch_size, total_images)} / {total_images}] 완료")

    # 4. 전체 임베딩 결합
    full_embeddings = torch.cat(all_embeddings_list, dim=0)

    # 5. 무결성 검증 (NaN / Inf / 차원 / 정규화 확인)
    print("\n=== 무결성 검증 ===")
    print(f"- 생성된 임베딩 텐서 형상: {tuple(full_embeddings.shape)}")
    has_nan = torch.isnan(full_embeddings).any().item()
    has_inf = torch.isinf(full_embeddings).any().item()
    print(f"- NaN 값 존재 여부: {has_nan}")
    print(f"- Inf(무한대) 값 존재 여부: {has_inf}")

    if has_nan or has_inf:
        raise ValueError("임베딩 텐서에 NaN 또는 Inf 비정상 값이 포함되어 있습니다.")

    # L2 노름 확인 (1.0에 근사해야 함)
    norms = torch.norm(full_embeddings, p=2, dim=-1)
    norm_min = norms.min().item()
    norm_max = norms.max().item()
    print(f"- L2 Norm 최소/최대값: {norm_min:.6f} / {norm_max:.6f}")

    # 6. 저장
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "image_embeddings.pt"
    meta_out_path = out_dir / "index_metadata.json"

    torch.save(full_embeddings, emb_path)
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    print(f"\n=== 인덱스 저장 완료 ===")
    print(f"- 임베딩 텐서 파일: {emb_path.resolve()}")
    print(f"- 인덱스 메타데이터 파일: {meta_out_path.resolve()}")
    print(f"- 총 인덱싱된 이미지 수: {len(index_records)}개")

    return {
        "embedding_shape": list(full_embeddings.shape),
        "total_indexed": len(index_records),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "embeddings_path": str(emb_path).replace("\\", "/"),
        "index_metadata_path": str(meta_out_path).replace("\\", "/"),
    }


def main():
    parser = argparse.ArgumentParser(description="CLIP 이미지 임베딩 인덱스 구축")
    parser.add_argument("--metadata-path", default="data/metadata.json", help="입력 메타데이터 경로")
    parser.add_argument("--output-dir", default="data/embeddings", help="임베딩 출력 폴더")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="사전학습 CLIP 모델 식별자")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    parser.add_argument("--device", default=None, help="실행 디바이스 (cuda 또는 cpu)")

    args = parser.parse_args()

    build_clip_index(
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
