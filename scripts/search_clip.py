"""
scripts/search_clip.py

제주데이터허브 CLIP 이미지 검색 시스템: 텍스트-이미지 검색 스크립트.
- 사전학습 모델: openai/clip-vit-base-patch32
- 검색 방식: 사용자가 한국어 검색어를 입력하면 텍스트 임베딩을 생성하여
  사전 구축된 L2 정규화 이미지 임베딩과의 코사인 유사도를 계산하고 상위 K개 이미지를 반환.
- 출력 정보:
  1) 이미지 파일명
  2) 보고서 연월
  3) 원본 PDF 파일명
  4) 페이지 번호
  5) 코사인 유사도 점수
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


class ClipSearchEngine:
    """
    CLIP 기반 텍스트-이미지 유사도 검색 엔진.
    """

    def __init__(
        self,
        embeddings_path: str = "data/embeddings/image_embeddings.pt",
        metadata_path: str = "data/embeddings/index_metadata.json",
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name

        self._load_index()
        self._load_model()

    def _load_index(self):
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {self.embeddings_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"인덱스 메타데이터를 찾을 수 없습니다: {self.metadata_path}")

        print(f"[검색 엔진] 인덱스 로드 중...")
        self.image_embeddings = torch.load(self.embeddings_path, map_location=self.device)
        # float32 보장 및 L2 정규화 확인
        self.image_embeddings = self.image_embeddings.to(torch.float32)
        self.image_embeddings = F.normalize(self.image_embeddings, p=2, dim=-1)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[검색 엔진] 총 {len(self.metadata)}개 이미지 임베딩 로드 완료 (형상: {tuple(self.image_embeddings.shape)})")

    def _load_model(self):
        print(f"[검색 엔진] 모델 로드 중 ({self.model_name}, device: {self.device})...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[검색 엔진] 모델 로드 완료.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        입력된 텍스트 질의(한국어 등)로 이미지 임베딩과의 코사인 유사도를 계산하고
        상위 top_k개 검색 결과를 반환합니다.
        """
        if not query or not query.strip():
            return []

        # 1. 텍스트 토크나이징 및 전처리
        inputs = self.processor(
            text=[query.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. 텍스트 임베딩 생성 (inference_mode)
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            # L2 정규화
            text_features = F.normalize(text_features, p=2, dim=-1).to(torch.float32)

        # 3. 코사인 유사도 계산 (모두 L2 정규화되어 있으므로 단순 내적)
        # text_features: (1, 512), image_embeddings: (N, 512)
        # similarities: (N,)
        similarities = (text_features @ self.image_embeddings.T).squeeze(0)

        # 4. 상위 K개 추출
        actual_k = min(top_k, len(self.metadata))
        topk_scores, topk_indices = torch.topk(similarities, k=actual_k)

        results = []
        for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), start=1):
            meta = self.metadata[idx]
            result_item = {
                "rank": rank,
                "score": float(score),
                "image_filename": meta.get("image_filename"),
                "report_year_month": meta.get("report_year_month"),
                "pdf_filename": meta.get("pdf_filename"),
                "page_number": meta.get("page_number"),
                "image_path": meta.get("image_path"),
                "title": meta.get("title"),
                "description": meta.get("description"),
                "text_extracted": meta.get("text_extracted", False),
            }
            results.append(result_item)

        return results


def print_search_results(query: str, results: List[Dict[str, Any]]):
    """
    검색 결과를 포맷팅하여 콘솔에 출력합니다.
    """
    print("\n" + "=" * 80)
    print(f"🔍 검색어: '{query}' (상위 {len(results)}개 결과)")
    print("=" * 80)

    for item in results:
        title_str = item["title"] if item["title"] else "(텍스트 미추출 / 스캔본)"
        desc_preview = ""
        if item["description"]:
            first_line = item["description"].splitlines()[0]
            desc_preview = f" | 설명: {first_line[:50]}..."

        print(f"[순위 {item['rank']}] 유사도 점수: {item['score']:.4f}")
        print(f"  - 이미지 파일명 : {item['image_filename']}")
        print(f"  - 보고서 연월   : {item['report_year_month']}")
        print(f"  - 원본 PDF 파일 : {item['pdf_filename']}")
        print(f"  - 페이지 번호   : {item['page_number']}페이지")
        print(f"  - 제목/내용     : {title_str}{desc_preview}")
        print(f"  - 파일 경로     : {item['image_path']}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="CLIP 기반 텍스트-이미지 유사도 검색")
    parser.add_argument("--query", "-q", required=True, help="검색할 텍스트 질의 (한국어 등)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="반환할 상위 결과 개수 (기본: 5)")
    parser.add_argument("--embeddings-path", default="data/embeddings/image_embeddings.pt", help="임베딩 파일 경로")
    parser.add_argument("--metadata-path", default="data/embeddings/index_metadata.json", help="인덱스 메타데이터 경로")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="CLIP 모델 식별자")
    parser.add_argument("--device", default=None, help="실행 디바이스 (cuda 또는 cpu)")

    args = parser.parse_args()

    engine = ClipSearchEngine(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        device=args.device,
    )

    results = engine.search(query=args.query, top_k=args.top_k)
    print_search_results(args.query, results)


if __name__ == "__main__":
    main()
