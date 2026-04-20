import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import hnswlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


def load_hdf5_data(
    data_path: str, num_elements: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load train/test vectors from HDF5 file (e.g., ann-benchmarks format)."""
    print(f"Loading HDF5 data from {data_path}")
    with h5py.File(data_path, "r") as f:
        train_data = f["train"][:]
        test_data = f["test"][:]

        if num_elements is not None and num_elements < len(train_data):
            print(f"Truncating train data from {len(train_data)} to {num_elements}")
            train_data = train_data[:num_elements]

    return train_data.astype(np.float32), test_data.astype(np.float32)


def load_or_generate_data(
    num_elements: int, dim: int, data_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or generate train/test vectors."""
    if data_path.endswith(".hdf5") or data_path.endswith(".h5"):
        return load_hdf5_data(data_path, num_elements)

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
        per_query.append(
            len(set(lhs_row.tolist()) & set(rhs_row.tolist())) / lhs.shape[1]
        )
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
            query_norms[:, None] + chunk_norms[None, :] - 2.0 * queries @ chunk.T
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
        per_query.append(
            len(set(predicted.tolist()) & set(exact.tolist())) / exact.shape[0]
        )
    return float(np.mean(per_query))


def normalize_ef_values(ef_values: Optional[List[int]], fallback_ef: int) -> List[int]:
    """Normalize ef inputs while preserving order."""
    values = ef_values if ef_values is not None else [fallback_ef]
    normalized = []
    seen = set()
    for ef in values:
        if ef <= 0:
            raise ValueError("All ef values must be positive integers")
        if ef not in seen:
            normalized.append(ef)
            seen.add(ef)
    return normalized


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


def generate_recall_qps_chart(
    comparisons: List[Dict], k: int, chart_dir: str = "charts"
):
    """
    Create a standard ANN-benchmark chart: X = Recall, Y = Log(QPS).
    """
    # Filter out entries that don't have recall computed
    comps = [c for c in comparisons if "baseline_recall" in c and "vf_recall" in c]
    if not comps:
        return None

    ef_values = [c["ef"] for c in comps]
    base_recall = [c["baseline_recall"] for c in comps]
    base_qps = [c["baseline"]["qps"] for c in comps]
    
    vf_recall = [c["vf_recall"] for c in comps]
    vf_qps = [c["vf"]["qps"] for c in comps]

    out_path = Path(chart_dir) / f"recall_qps_k{k}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    
    # Plot curves
    plt.plot(base_recall, base_qps, marker="o", label=f"Baseline HNSW (k={k})", linestyle='-', markersize=8)
    plt.plot(vf_recall, vf_qps, marker="s", label=f"VF‑HNSW (k={k})", linestyle='-', markersize=8)
    
    # Annotate points with their ef values
    for i, ef in enumerate(ef_values):
        plt.annotate(f"ef={ef}", (base_recall[i], base_qps[i]), textcoords="offset points", xytext=(-10,10), ha='center', fontsize=9, color='blue')
        plt.annotate(f"ef={ef}", (vf_recall[i], vf_qps[i]), textcoords="offset points", xytext=(10,-15), ha='center', fontsize=9, color='orange')

    plt.title(f"Recall vs. QPS Frontier (Dataset: MNIST-784)")
    plt.xlabel(f"Recall@{k}")
    plt.ylabel("Queries per second (QPS)")
    
    # Standard ANN graph uses Log scale for QPS
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return str(out_path)


def summarize_frontier_table(
    comparisons: List[Dict[str, object]], k: int, recall_queries: int
) -> None:
    print("\nSummary")
    # Changed header from vf/base to time_ratio for clarity
    header = (
        f"{'ef':>6} {'base_qps':>10} {'vf_qps':>10} {'time_ratio':>12} " f"{'overlap':>8}"
    )
    if recall_queries > 0:
        header += f" {'base_r@' + str(k):>10} {'vf_r@' + str(k):>10}"
    print(header)

    for comparison in comparisons:
        row = (
            f"{comparison['ef']:>6} "
            f"{comparison['baseline']['qps']:>10.0f} "
            f"{comparison['vf']['qps']:>10.0f} "
            f"{comparison['vf_time_ratio']:>11.3f}x "
            f"{comparison['overlap']:>8.4f}"
        )
        if recall_queries > 0:
            row += (
                f" {comparison['baseline_recall']:>10.4f}"
                f" {comparison['vf_recall']:>10.4f}"
            )
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="HNSW vs VF-HNSW performance benchmark")
    parser.add_argument("--dim", type=int, default=None, help="Vector dimension")
    parser.add_argument("--num_elements", type=int, default=None, help="Num vectors")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=50)
    parser.add_argument(
        "--ef_values", type=int, nargs="+", default=None, help="Sweep over ef_search"
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k results to return")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Timed iterations")
    parser.add_argument(
        "--recall_queries", type=int, default=200, help="Queries for exact recall baseline"
    )
    parser.add_argument("--data_path", type=str, default="data/mnist-784-euclidean.hdf5")
    args = parser.parse_args()

    # Auto-detect dimension from HDF5 if needed
    if args.data_path.endswith((".hdf5", ".h5")):
        with h5py.File(args.data_path, "r") as f:
            detected_dim = f["train"].shape[1]
            args.dim = detected_dim if args.dim is None else args.dim

    if args.dim is None:
        raise ValueError("--dim is required for non-HDF5 datasets")

    train_data, test_data = load_or_generate_data(
        args.num_elements or 100000, args.dim, args.data_path
    )
    ef_values = normalize_ef_values(args.ef_values, args.ef_search)

    print(f"\n{'=' * 60}")
    print("HNSW Benchmark - Baseline vs VF-HNSW")
    print(f"{'=' * 60}")
    print(f"Data: {args.data_path} ({len(train_data)} train, {len(test_data)} test)")
    print(f"Dim: {args.dim}, M: {args.M}, ef_construction: {args.ef_construction}")
    print(f"Sweep ef_values: {ef_values}, Threads: {args.threads}, K: {args.k}")

    index = hnswlib.Index(space="l2", dim=args.dim)
    index.init_index(max_elements=len(train_data), ef_construction=args.ef_construction, M=args.M)

    print("\nBuilding index...")
    build_time = benchmark_build(index, train_data, args.threads)
    print(f"Build time: {build_time:.2f}s ({len(train_data) / build_time:.0f} vectors/s)")

    recall_queries = min(args.recall_queries, len(test_data))
    exact_labels = None
    if recall_queries > 0:
        print(f"\nComputing exact ground truth on {recall_queries} queries (brute force)...")
        exact_labels = exact_knn_l2(train_data, test_data[:recall_queries], args.k)

    print(f"\nBenchmarking Search ({len(test_data)} queries)...")
    comparisons = []
    
    for ef in ef_values:
        baseline = benchmark_search(
            index, test_data, k=args.k, ef=ef, threads=args.threads,
            mode="standard", warmup=args.warmup, iterations=args.iterations,
        )
        vf = benchmark_search(
            index, test_data, k=args.k, ef=ef, threads=args.threads,
            mode="vf_hnsw", warmup=args.warmup, iterations=args.iterations,
        )

        comparison = {
            "ef": ef,
            "baseline": baseline,
            "vf": vf,
            "overlap": overlap_at_k(baseline["labels"], vf["labels"]),
            "vf_time_ratio": vf["mean_time"] / baseline["mean_time"],
        }
        
        if exact_labels is not None:
            comparison["baseline_recall"] = recall_at_k(baseline["labels"][:recall_queries], exact_labels)
            comparison["vf_recall"] = recall_at_k(vf["labels"][:recall_queries], exact_labels)

        comparisons.append(comparison)

    if len(comparisons) > 0:
        summarize_frontier_table(comparisons, args.k, recall_queries)

    if len(comparisons) > 1 and exact_labels is not None:
        chart_file = generate_recall_qps_chart(comparisons, k=args.k)
        if chart_file:
            print(f"\n[Success] Recall vs QPS chart saved to: {chart_file}")

if __name__ == "__main__":
    main()
    