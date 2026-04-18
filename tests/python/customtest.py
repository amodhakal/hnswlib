import hnswlib
import numpy as np
import time
import argparse
from typing import Tuple
import os

def load_or_generate_data(
    num_elements: int, dim: int, data_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or generate train/test vectors."""
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
    else:
        print(f"Generating random data (NOT RECOMMENDED FOR REAL BENCHMARKS)")
        np.random.seed(42)
        data = np.float32(np.random.random((num_elements, dim)))
        np.save(data_path, data)

    # Split into train/test
    split = int(0.95 * len(data))
    return data[:split], data[split:]


def benchmark_build(index: hnswlib.Index, data: np.ndarray, threads: int) -> float:
    """Benchmark index construction."""
    index.set_num_threads(threads)
    t0 = time.time()
    index.add_items(data)
    return time.time() - t0


def benchmark_search(
    index: hnswlib.Index,
    queries: np.ndarray,
    k: int,
    ef: int,
    threads: int,
    warmup: int = 1,
    iterations: int = 5,
) -> dict:
    """Benchmark search with multiple iterations."""
    index.set_ef(ef)
    index.set_num_threads(threads)

    # Warmup
    for _ in range(warmup):
        index.knn_query(queries[:100], k=k)

    # Actual benchmark
    times = []
    for _ in range(iterations):
        t0 = time.time()
        labels, distances = index.knn_query(queries, k=k)
        elapsed = time.time() - t0
        times.append(elapsed)

    qps = len(queries) / np.mean(times)
    latency_ms = (np.mean(times) / len(queries)) * 1000

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "qps": qps,
        "latency_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="HNSW Performance Benchmark")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimension")
    parser.add_argument(
        "--num_elements", type=int, default=100000, help="Number of vectors"
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=50)
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to pre-generated embeddings"
    )
    args = parser.parse_args()

    # Load or generate data
    if args.data_path is None:
        args.data_path = f"random_data_{args.num_elements}_{args.dim}.npy"

    train_data, test_data = load_or_generate_data(
        args.num_elements, args.dim, args.data_path
    )

    print(f"\n{'='*60}")
    print(f"HNSW Benchmark - hnswlib baseline")
    print(f"{'='*60}")
    print(f"Dataset: {len(train_data)} train, {len(test_data)} test")
    print(f"Dimension: {args.dim}")
    print(f"M={args.M}, ef_construction={args.ef_construction}")
    print(f"Threads: {args.threads}")

    # Build index
    index = hnswlib.Index(space="l2", dim=args.dim)
    index.init_index(
        max_elements=len(train_data), ef_construction=args.ef_construction, M=args.M
    )

    print(f"\nBuilding index...")
    build_time = benchmark_build(index, train_data, args.threads)
    print(f"Build time: {build_time:.2f}s ({len(train_data)/build_time:.0f} vectors/s)")

    # Search benchmark
    print(f"\nSearching with ef={args.ef_search}, k=10...")
    results = benchmark_search(
        index, test_data, k=10, ef=args.ef_search, threads=args.threads
    )

    print(f"QPS: {results['qps']:.0f}")
    print(f"Latency: {results['latency_ms']:.3f}ms per query")
    print(f"Time: {results['mean_time']:.3f}s ± {results['std_time']:.3f}s")

    # Save results
    log_file = f"benchmark_dim{args.dim}_n{args.num_elements}_t{args.threads}.csv"
    with open(log_file, "a") as f:
        f.write(
            f"{args.M},{args.ef_construction},{args.ef_search},"
            f"{build_time:.3f},{results['qps']:.0f},"
            f"{results['latency_ms']:.3f}\n"
        )

    print(f"\nResults appended to {log_file}")


if __name__ == "__main__":
    main()
