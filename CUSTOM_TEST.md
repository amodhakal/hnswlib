```
Download dataset
http://ann-benchmarks.com/mnist-784-euclidean.hdf5
# Compile
cd build && cmake .. && cd .. && uv pip install -e .

# Test
uv run tests/python/customtest.py \
--threads 8 \
--ef_values 10 20 50 100 \
--iterations 5 \
--warmup 1 \
--recall_queries 200 \
--data_path data/mnist-784-euclidean.hdf5
```
