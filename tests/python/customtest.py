import argparse
import os
import time
from typing import Dict, Tuple

import hnswlib
import numpy as np


def load_or_generate_data(
    num_elements: int, dim: int, data_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or generate train/test vectors."""
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
    else:
        print("Generating random data (NOT RECOMMENDED FOR REAL BENCHMARKS)")
        rng = np.random.default_rng(42)
        data = rng.random((num_elements, dim), dtype=np.float32)
        np.save(data_path, data)

    split = int(0.95 * len(data))
    return data[:split], data[split:]


def benchmark_build(index: hnswlib.Index, data: np.ndarray, threads: int) -> float:
    """Benchmark index construction."""
    index.set_num_threads(threads)
    t0 = time.time()
    index.add_items(data)
    return time.time() - t0


def overlap_at_k(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Average set overlap between two result matrices."""
    per_query = []
    for lhs_row, rhs_row in zip(lhs, rhs):
        per_query.append(len(set(lhs_row.tolist()) & set(rhs_row.tolist())) / lhs.shape[1])
    return float(np.mean(per_query))


def exact_knn_l2(
    train_data: np.ndarray, queries: np.ndarray, k: int, chunk_size: int = 5000
) -> np.ndarray:
    """Compute exact top-k labels for a small query batch using chunked matrix ops."""
    query_norms = np.sum(queries * queries, axis=1, dtype=np.float32)
    best_distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    best_labels = np.full((len(queries), k), -1, dtype=np.int64)

    for start in range(0, len(train_data), chunk_size):
        chunk = train_data[start : start + chunk_size]
        chunk_norms = np.sum(chunk * chunk, axis=1, dtype=np.float32)
        distances = (
            query_norms[:, None]
            + chunk_norms[None, :]
            - 2.0 * queries @ chunk.T
        )
        distances = np.maximum(distances, 0.0)
        labels = np.arange(start, start + len(chunk), dtype=np.int64)
        labels = np.broadcast_to(labels, distances.shape)

        merged_distances = np.concatenate((best_distances, distances), axis=1)
        merged_labels = np.concatenate((best_labels, labels), axis=1)
        topk_idx = np.argpartition(merged_distances, kth=k - 1, axis=1)[:, :k]

        best_distances = np.take_along_axis(merged_distances, topk_idx, axis=1)
        best_labels = np.take_along_axis(merged_labels, topk_idx, axis=1)

        order = np.argsort(best_distances, axis=1)
        best_distances = np.take_along_axis(best_distances, order, axis=1)
        best_labels = np.take_along_axis(best_labels, order, axis=1)

    return best_labels


def recall_at_k(predicted_labels: np.ndarray, exact_labels: np.ndarray) -> float:
    """Average recall@k for two result matrices."""
    per_query = []
    for predicted, exact in zip(predicted_labels, exact_labels):
        per_query.append(len(set(predicted.tolist()) & set(exact.tolist())) / exact.shape[0])
    return float(np.mean(per_query))


def benchmark_search(
    index: hnswlib.Index,
    queries: np.ndarray,
    k: int,
    ef: int,
    threads: int,
    mode: str,
    warmup: int = 1,
    iterations: int = 5,
) -> Dict[str, object]:
    """Benchmark one search mode with multiple iterations."""
    index.set_ef(ef)
    index.set_num_threads(threads)
    index.set_search_mode(mode)

    warmup_queries = queries[: min(100, len(queries))]
    for _ in range(warmup):
        index.knn_query(warmup_queries, k=k)

    times = []
    final_labels = None
    final_distances = None
    for _ in range(iterations):
        t0 = time.time()
        labels, distances = index.knn_query(queries, k=k)
        elapsed = time.time() - t0
        times.append(elapsed)
        final_labels = labels
        final_distances = distances

    mean_time = float(np.mean(times))
    qps = len(queries) / mean_time
    latency_ms = (mean_time / len(queries)) * 1000

    return {
        "mode": mode,
        "mean_time": mean_time,
        "std_time": float(np.std(times)),
        "qps": qps,
        "latency_ms": latency_ms,
        "labels": final_labels,
        "distances": final_distances,
    }


def summarize_mode(name: str, metrics: Dict[str, object]) -> None:
    print(f"\n{name}")
    print(f"  Search mode: {metrics['mode']}")
    print(f"  QPS: {metrics['qps']:.0f}")
    print(f"  Latency: {metrics['latency_ms']:.3f}ms per query")
    print(f"  Time: {metrics['mean_time']:.3f}s +/- {metrics['std_time']:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="HNSW vs VF-HNSW performance benchmark")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimension")
    parser.add_argument(
        "--num_elements", type=int, default=100000, help="Number of vectors"
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=50)
    parser.add_argument("--k", type=int, default=10, help="Top-k results to return")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Timed iterations per mode"
    )
    parser.add_argument(
        "--recall_queries",
        type=int,
        default=100,
        help="Queries sampled for exact recall@k",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to pre-generated embeddings"
    )
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = f"random_data_{args.num_elements}_{args.dim}.npy"

    train_data, test_data = load_or_generate_data(
        args.num_elements, args.dim, args.data_path
    )

    print(f"\n{'=' * 60}")
    print("HNSW Benchmark - baseline vs VF-HNSW")
    print(f"{'=' * 60}")
    print(f"Dataset: {len(train_data)} train, {len(test_data)} test")
    print(f"Dimension: {args.dim}")
    print(f"M={args.M}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    print(f"Threads: {args.threads}, k={args.k}")

    index = hnswlib.Index(space="l2", dim=args.dim)
    index.init_index(
        max_elements=len(train_data), ef_construction=args.ef_construction, M=args.M
    )

    print("\nBuilding index...")
    build_time = benchmark_build(index, train_data, args.threads)
    print(f"Build time: {build_time:.2f}s ({len(train_data) / build_time:.0f} vectors/s)")

    print(f"\nSearching {len(test_data)} queries...")
    baseline = benchmark_search(
        index,
        test_data,
        k=args.k,
        ef=args.ef_search,
        threads=args.threads,
        mode="standard",
        warmup=args.warmup,
        iterations=args.iterations,
    )
    vf = benchmark_search(
        index,
        test_data,
        k=args.k,
        ef=args.ef_search,
        threads=args.threads,
        mode="vf_hnsw",
        warmup=args.warmup,
        iterations=args.iterations,
    )

    summarize_mode("Baseline HNSW", baseline)
    summarize_mode("VF-HNSW", vf)

    overlap = overlap_at_k(baseline["labels"], vf["labels"])
    print(f"\nResult overlap@{args.k}: {overlap:.4f}")
    print(f"VF time ratio vs baseline: {vf['mean_time'] / baseline['mean_time']:.3f}x")

    recall_queries = min(args.recall_queries, len(test_data))
    if recall_queries > 0:
        sample_queries = test_data[:recall_queries]
        exact_labels = exact_knn_l2(train_data, sample_queries, args.k)
        baseline_recall = recall_at_k(baseline["labels"][:recall_queries], exact_labels)
        vf_recall = recall_at_k(vf["labels"][:recall_queries], exact_labels)
        print(f"Baseline recall@{args.k} on {recall_queries} exact queries: {baseline_recall:.4f}")
        print(f"VF-HNSW recall@{args.k} on {recall_queries} exact queries: {vf_recall:.4f}")

    log_file = f"benchmark_dim{args.dim}_n{args.num_elements}_t{args.threads}.csv"
    with open(log_file, "a", encoding="utf-8") as handle:
        for metrics in (baseline, vf):
            handle.write(
                f"{metrics['mode']},{args.M},{args.ef_construction},{args.ef_search},"
                f"{build_time:.3f},{metrics['qps']:.0f},{metrics['latency_ms']:.3f},"
                f"{overlap:.4f}\n"
            )

    print(f"\nResults appended to {log_file}")


if __name__ == "__main__":
    main()
