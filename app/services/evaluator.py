"""
Pipeline Evaluation & Benchmark Monitoring Engine.
Measures:
1. Extraction throughput and page latency (ms/page, chunks/sec)
2. Content noise ratios and OCR quality scoring
3. Retrieval performance metrics: MRR (Mean Reciprocal Rank), Precision@K, Recall@K
"""

from datetime import datetime, timezone
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from app.ingestion.cleaner import text_cleaner
from app.services.retrieval import retrieval_service


class PipelineEvaluator:
    """Evaluates pipeline processing efficiency and information retrieval accuracy."""

    @staticmethod
    def calculate_mrr(rank_positions: List[Optional[int]]) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR):
        MRR = 1/|Q| * sum(1 / rank_i) for each query i where relevant doc is at rank_i.
        """
        if not rank_positions:
            return 0.0
        reciprocal_ranks = [1.0 / r for r in rank_positions if r is not None and r > 0]
        return round(sum(reciprocal_ranks) / len(rank_positions), 4) if rank_positions else 0.0

    @staticmethod
    def calculate_precision_recall_at_k(
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        k: int = 3,
    ) -> Tuple[float, float]:
        """
        Computes Precision@K and Recall@K.
        """
        if not ground_truth_ids or k <= 0:
            return 0.0, 0.0

        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(ground_truth_ids)

        hits = len(top_k_retrieved.intersection(relevant_set))
        precision_at_k = round(hits / k, 4)
        recall_at_k = round(hits / len(relevant_set), 4)

        return precision_at_k, recall_at_k

    def evaluate_retrieval_suite(
        self,
        benchmark_queries: List[Dict[str, Any]],
        k: int = 3,
    ) -> Dict[str, Any]:
        """
        Runs automated evaluation over a set of query evaluation test cases:
        [{"query": "...", "expected_chunk_ids": ["c1", "c2"]}]
        """
        ranks = []
        precisions = []
        recalls = []
        latencies = []

        for item in benchmark_queries:
            q = item["query"]
            expected = item.get("expected_chunk_ids", [])

            t0 = time.perf_counter()
            results = retrieval_service.hybrid_search(q, top_k=k * 2)
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_ids = [r.chunk_id for r in results]

            # Find rank of first relevant match
            first_rank = None
            for idx, r_id in enumerate(retrieved_ids, start=1):
                if r_id in expected:
                    first_rank = idx
                    break
            ranks.append(first_rank)

            p_k, r_k = self.calculate_precision_recall_at_k(retrieved_ids, expected, k=k)
            precisions.append(p_k)
            recalls.append(r_k)

        mrr = self.calculate_mrr(ranks)
        avg_precision = round(sum(precisions) / max(len(precisions), 1), 4)
        avg_recall = round(sum(recalls) / max(len(recalls), 1), 4)
        avg_latency = round(sum(latencies) / max(len(latencies), 1), 2)

        return {
            "total_queries_evaluated": len(benchmark_queries),
            "k": k,
            "mean_reciprocal_rank_mrr": mrr,
            "average_precision_at_k": avg_precision,
            "average_recall_at_k": avg_recall,
            "average_search_latency_ms": avg_latency,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    def benchmark_extraction_quality(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Measures extraction noise, alphanumeric ratios, and quality scores across text samples.
        """
        ratings = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "EMPTY": 0}
        total_chars = 0
        total_words = 0
        noise_scores = []

        for text in sample_texts:
            q = text_cleaner.calculate_quality(text)
            ratings[q.quality_rating] = ratings.get(q.quality_rating, 0) + 1
            total_chars += q.total_characters
            total_words += q.total_words
            noise_scores.append(q.noise_score)

        avg_noise = round(sum(noise_scores) / max(len(noise_scores), 1), 4)

        return {
            "documents_evaluated": len(sample_texts),
            "total_words": total_words,
            "total_characters": total_chars,
            "average_noise_score": avg_noise,
            "quality_distribution": ratings,
        }


pipeline_evaluator = PipelineEvaluator()
