```
# Build

# Quick test
uv run tests/python/customtest.py  --dim 384 --num_elements 10000 --threads 8

# Real test with pre-generated embeddings
uv run tests/python/customtest.py  --dim 384 --num_elements 100000 --threads 8 \
  --data_path embeddings_100k.npy

# Parameter sweep
for ef in 10 20 50 100; do
  uv run tests/python/customtest.py  --dim 384 --ef_search $ef --threads 8
done
```