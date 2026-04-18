#include <sys/types.h>

typedef uint HNSWElement;

class HNSW
{
    /**
    Insert a value into HNSW

    @param newElement the new element (q)
    @param establishedConnectionCount number of established connections (M)
    @param maxConnectionEachElementPerLayer max number of connections for each element per layer (mmax)
    @param dynamicCandidateListSize size of dynamic candidate list (efConstruction)
    @param normalizatioNFactorLevelGeneration normalization factor for level generation (m)
    */
    void insert(HNSWElement newElement, uint establishedConnectionCount, uint maxConnectionEachElementPerLayer,
                uint dynamicCandidateListSize, uint normalizatioNFactorLevelGeneration);
};
