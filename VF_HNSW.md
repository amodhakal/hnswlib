The idea here is to treat HNSW graph as a flat one-level non-hierarchical graph and search on all layers at once. Without actual flattening, so let's call it VF-HNSW, Virtually Flattened Hierarchical Navigable Small World.

It can be simply described as "when searching on the Nth layer, instead of adding all node neighbors from the same layer to the candidate set, add all node neighbors on all layers to the candidate set". But this of course will cause too many unnecessary distance calculations, so an optimization would be "add neighbors from the layer k-1 only if any of the k layer neighbors was better than all k+1 layer neighbors".

This will allow searching all layers in one traversal without overshooting the target.