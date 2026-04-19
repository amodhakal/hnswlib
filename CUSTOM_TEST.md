```
# Quick test
cd build && cmake .. && uv pip install -e . && uv run tests/python/customtest.py  --dim 384 --num_elements 10000 --threads 4

# Real test with pre-generated embeddings
cd build && cmake .. && uv pip install -e . && uv run tests/python/customtest.py  --dim 384 --num_elements 100000 --threads 4 \
  --data_path embeddings_100k.npy

# Parameter sweep
for ef in 10 20 50 100; do
  cd build && cmake .. && uv pip install -e . && uv run tests/python/customtest.py  --dim 384 --ef_search $ef --threads 4
done
```