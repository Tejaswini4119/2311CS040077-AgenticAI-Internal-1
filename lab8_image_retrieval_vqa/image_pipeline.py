"""
Lab 4: Image Retrieval / Visual QA System (multimodal pipeline)
==================================================================
A two-stage multimodal pipeline:

  1. IMAGE RETRIEVAL: given a text query, find the most visually/semantically
     matching image(s) in an image library, using a lightweight, fully local
     "feature extractor" (color histogram + simple shape/edge statistics)
     turned into a searchable vector index - no GPU or model download
     required. This is the same *pattern* a CLIP-embedding retrieval system
     uses (image -> vector -> nearest-neighbor search); only the feature
     extractor differs, and can be swapped in `ImageFeatureExtractor` for a
     real embedding model (CLIP, SigLIP, etc.) when internet/GPU is available.

  2. VISUAL QUESTION ANSWERING: given an image + a natural-language question,
     answer it. Uses the shared LLMClient's `complete_vision()`, which calls
     a real multimodal LLM (Claude/GPT-4o) if an API key is configured, and
     otherwise falls back to a deterministic feature-based mock captioner
     (dominant color, brightness, size) so the pipeline is still runnable
     and demonstrable offline.

Run:
    python image_pipeline.py --index sample_images
    python image_pipeline.py --retrieve "a red shape"
    python image_pipeline.py --vqa sample_images/blue_circle.png --question "What color is this shape?"
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.llm_client import LLMClient
from shared.utils import print_header


# ---------------------------------------------------------------------- #
# Stage 1: local, model-free image feature extractor
# (drop-in swap point for a real embedding model like CLIP)
# ---------------------------------------------------------------------- #
class ImageFeatureExtractor:
    """
    Produces a fixed-length numeric feature vector for an image using:
      - an RGB color histogram (captures dominant colors / palette)
      - coarse spatial color-mean grid (captures rough shape/position, e.g.
        "color concentrated in the center" vs "spread to the edges")
    This is a classical CV feature vector, not a learned embedding - but it
    plays the exact same architectural role as a CLIP embedding in the
    pipeline: turning an image into a vector that can be compared with
    cosine similarity.
    """

    def __init__(self, bins: int = 8, grid: int = 4):
        self.bins = bins
        self.grid = grid

    def extract(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB").resize((128, 128))
        arr = np.asarray(img).astype(np.float32) / 255.0

        # 1. Color histogram per channel
        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=self.bins, range=(0, 1))
            hist_features.append(hist / hist.sum())
        hist_vec = np.concatenate(hist_features)

        # 2. Coarse spatial grid of mean colors (captures rough layout)
        h, w, _ = arr.shape
        gh, gw = h // self.grid, w // self.grid
        grid_features = []
        for i in range(self.grid):
            for j in range(self.grid):
                patch = arr[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw, :]
                grid_features.append(patch.mean(axis=(0, 1)))
        grid_vec = np.concatenate(grid_features)

        vec = np.concatenate([hist_vec, grid_vec])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def caption_from_text_query(self, query: str) -> np.ndarray:
        """
        For a fair architectural demo without a real joint text-image
        embedding model, we approximate 'text->image feature space' by
        mapping simple color-name keywords in the query onto the same
        color-histogram feature space used for images. This lets the text
        query and image features live in a comparable space for nearest-
        neighbor search. A real system would replace this with CLIP's text
        encoder.
        """
        color_map = {
            "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0), "green": (0.0, 1.0, 0.0),
            "yellow": (1.0, 1.0, 0.0), "purple": (0.6, 0.0, 0.8), "orange": (1.0, 0.5, 0.0),
        }
        matched_rgb = None
        for name, rgb in color_map.items():
            if name in query.lower():
                matched_rgb = rgb
                break
        if matched_rgb is None:
            matched_rgb = (0.5, 0.5, 0.5)  # neutral if no color keyword found

        # Build a synthetic "solid color image" feature vector for the query
        synthetic = np.ones((128, 128, 3), dtype=np.float32) * np.array(matched_rgb)
        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(synthetic[:, :, c], bins=self.bins, range=(0, 1))
            hist_features.append(hist / hist.sum())
        hist_vec = np.concatenate(hist_features)
        grid_vec = np.tile(np.array(matched_rgb), self.grid * self.grid)
        vec = np.concatenate([hist_vec, grid_vec])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------- #
# Stage 1b: retrieval index
# ---------------------------------------------------------------------- #
@dataclass
class ImageRetriever:
    extractor: ImageFeatureExtractor = field(default_factory=ImageFeatureExtractor)
    index: Dict[str, np.ndarray] = field(default_factory=dict)

    def build_index(self, image_dir: str) -> None:
        paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")))
        print(f"[ImageRetriever] Indexing {len(paths)} images from {image_dir}")
        for p in paths:
            self.index[p] = self.extractor.extract(p)

    def search(self, text_query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.extractor.caption_from_text_query(text_query)
        scored = []
        for path, vec in self.index.items():
            sim = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8))
            scored.append((path, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------- #
# Stage 2: Visual QA agent
# ---------------------------------------------------------------------- #
@dataclass
class VisualQAAgent:
    llm: LLMClient = field(default_factory=LLMClient)

    def answer(self, image_path: str, question: str) -> str:
        return self.llm.complete_vision(
            prompt=question,
            image_path=image_path,
            system="You are a precise visual question-answering assistant. Answer only based on what is visible in the image.",
        )


# ---------------------------------------------------------------------- #
# Combined pipeline: retrieve then answer questions about the top result
# ---------------------------------------------------------------------- #
class MultimodalPipeline:
    def __init__(self):
        self.retriever = ImageRetriever()
        self.vqa = VisualQAAgent()

    def run(self, image_dir: str, text_query: str, question: str | None = None, top_k: int = 3) -> Dict:
        self.retriever.build_index(image_dir)
        results = self.retriever.search(text_query, top_k=top_k)

        print_header("RETRIEVAL RESULTS")
        for path, score in results:
            print(f"  {score:.3f}  {os.path.basename(path)}")

        vqa_answer = None
        if question and results:
            top_image = results[0][0]
            print_header(f"VISUAL QA on top result: {os.path.basename(top_image)}")
            vqa_answer = self.vqa.answer(top_image, question)
            print(f"Q: {question}\nA: {vqa_answer}")

        return {"query": text_query, "results": results, "question": question, "answer": vqa_answer}


def main():
    parser = argparse.ArgumentParser(description="Lab 4: Image Retrieval / Visual QA multimodal pipeline")
    parser.add_argument("--image_dir", default="sample_images", help="Directory of images to index")
    parser.add_argument("--query", default="a red shape", help="Text query for retrieval")
    parser.add_argument("--question", default="What color and shape is this?", help="VQA question on top retrieved image")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    pipeline = MultimodalPipeline()
    result = pipeline.run(args.image_dir, args.query, args.question, args.top_k)

    print_header("SUMMARY")
    print(json.dumps({
        "query": result["query"],
        "top_results": [(os.path.basename(p), round(s, 3)) for p, s in result["results"]],
        "vqa_question": result["question"],
        "vqa_answer": result["answer"],
    }, indent=2))


if __name__ == "__main__":
    main()
