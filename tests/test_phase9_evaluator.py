"""
Phase 9 Tests: System Evaluation, Benchmark Metrics, MRR, and Precision@K.
"""

from app.services.evaluator import PipelineEvaluator


def test_mrr_calculation():
    evaluator = PipelineEvaluator()

    # Query 1 rank 1 (1/1=1.0), Query 2 rank 2 (1/2=0.5), Query 3 rank 4 (1/4=0.25)
    # Expected MRR = (1.0 + 0.5 + 0.25) / 3 = 1.75 / 3 = 0.5833
    ranks = [1, 2, 4]
    mrr = evaluator.calculate_mrr(ranks)
    assert mrr == 0.5833

    # Query with no match (None)
    ranks_with_miss = [1, None]
    mrr_miss = evaluator.calculate_mrr(ranks_with_miss)
    assert mrr_miss == 0.5


def test_precision_recall_at_k():
    evaluator = PipelineEvaluator()

    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
    ground_truth = ["chunk_2", "chunk_5"]

    # At k=3: retrieved are [chunk_1, chunk_2, chunk_3]. Hits = 1 (chunk_2).
    # Precision@3 = 1 / 3 = 0.3333
    # Recall@3 = 1 / 2 = 0.5000
    p3, r3 = evaluator.calculate_precision_recall_at_k(retrieved, ground_truth, k=3)
    assert p3 == 0.3333
    assert r3 == 0.5


def test_benchmark_extraction_quality():
    evaluator = PipelineEvaluator()
    samples = [
        "Clean paragraph with well formatted engineering words.",
        "Another clean document describing architecture specifications.",
        "$$$ %%% @@@ ### !!!",  # noisy sample
    ]

    report = evaluator.benchmark_extraction_quality(samples)
    assert report["documents_evaluated"] == 3
    assert report["quality_distribution"]["POOR"] == 1
    assert report["quality_distribution"]["EXCELLENT"] >= 1
    assert report["average_noise_score"] > 0
