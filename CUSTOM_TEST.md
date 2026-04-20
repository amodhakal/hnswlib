```
# Compile
cd build && cmake .. && cd .. && uv pip install -e .
# Test
uv run tests/python/customtest.py --dim 384 --num_elements 1000000 --threads 8 --ef_values 10 20 50 100 --iterations 1 --warmup 0 --recall_queries 200
```
