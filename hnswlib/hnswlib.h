#pragma once

/**
 * @file hnswlib.h
 * @brief HNSW (Hierarchical Navigable Small World) library header.
 *
 * This is the main header file that includes all HNSW library functionality.
 * It provides:
 *   - Core interfaces (SpaceInterface, AlgorithmInterface)
 *   - Distance function types (DISTFUNC)
 *   - Binary serialization utilities
 *   - SIMD detection (SSE, AVX, AVX512)
 *   - Filter and stop condition base classes
 *
 * ========================================================================
 * QUICK START
 * ========================================================================
 *
 * 1. Create a space (L2 or Inner Product):
 *    hnswlib::L2Space space(dim);  // for Euclidean distance
 *    // OR
 *    hnswlib::InnerProductSpace space(dim);  // for cosine/IP
 *
 * 2. Create an index:
 *    hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);
 *
 * 3. Add points:
 *    index.addPoint(vector_data, label);
 *
 * 4. Search:
 *    auto results = index.searchKnn(query_vector, k);
 *
 * ========================================================================
 * KEY CONCEPTS
 * ========================================================================
 *
 * HNSW is a graph-based approximate nearest neighbor (ANN) algorithm.
 *
 * The index builds a multi-layer graph where:
 *   - Each element is a node with a vector
 *   - Edges connect similar (nearby) elements
 *   - Higher layers have fewer but longer edges
 *   - Search starts from top layer and descends
 *
 * Parameters:
 *   - M: Number of connections per node (default 16)
 *   - efConstruction: Search width during build (default 200)
 *   - ef: Search width during query (default 10)
 *
 * Trade-offs:
 *   - Higher M: better recall, slower build, more memory
 *   - Higher efConstruction: better recall, slower build
 *   - Higher ef: better recall, slower query
 */

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)
#ifndef HNSWLIB_ERR_OVERRIDE
  #define HNSWERR std::cerr
#else
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

/**
 * @def NO_MANUAL_VECTORIZATION
 * @brief Define to disable SIMD optimizations.
 *
 * If defined, the library uses only scalar code.
 * Useful for debugging or non-x86 platforms.
 */

#ifndef NO_MANUAL_VECTORIZATION
/**
 * @def USE_SSE
 * @brief Enable SSE (Streaming SIMD Extensions) for distance computations.
 *
 * SSE provides 128-bit registers (4 floats at once).
 * Available on virtually all modern x86 CPUs.
 */

#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
/**
 * @def USE_AVX
 * @brief Enable AVX (Advanced Vector Extensions) for distance computations.
 *
 * AVX provides 256-bit registers (8 floats at once).
 * Available on Ivy Bridge and later Intel/AMD CPUs.
 */
#define USE_AVX
#ifdef __AVX512F__
/**
 * @def USE_AVX512
 * @brief Enable AVX-512 for distance computations.
 *
 * AVX-512 provides 512-bit registers (16 floats at once).
 * Available on Knights Landing and later Intel CPUs.
 */
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
/**
 * @brief CPU feature detection infrastructure.
 *
 * These functions detect CPU capabilities for SIMD at runtime.
 * Some CPUs support the instruction set but not the OS handling.
 * Both CPU and OS must support the instructions.
 */

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
/**
 * @brief Get CPUID information.
 * @param out Array to receive [eax, ebx, ecx, edx]
 * @param eax Function number
 * @param ecx Sub-function number
 */
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
/**
 * @brief Get XCR (Extended Control Register) value.
 * @param x XCR index
 * @return XCR value
 */
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

/**
 * @def PORTABLE_ALIGN32
 * @brief 32-byte aligned attribute for stack arrays.
 *
 * SIMD operations work best with aligned data.
 * Use this for temporary arrays to enable aligned loads.
 */

/**
 * @def PORTABLE_ALIGN64
 * @brief 64-byte aligned attribute for stack arrays.
 *
 * AVX-512 requires 64-byte alignment.
 */
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64)))
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK  0

/**
 * @brief Check if AVX is available on this system.
 *
 * AVX requires both:
 *   1. CPU support (checked via CPUID)
 *   2. OS support (checked via XCR)
 *
 * Even if the CPU supports AVX, the OS must enable the
 * YMM state for AVX to work. XSAVE/XRSTORE handles this.
 *
 * @return true if AVX can be used
 */
static bool AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

/**
 * @brief Check if AVX-512 is available on this system.
 *
 * AVX-512 requires:
 *   1. CPU support for AVX-512F (Foundation)
 *   2. OS support for full ZMM state
 *   3. AVX already working (prerequisite)
 *
 * @return true if AVX-512 can be used
 */
static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000001) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {

/**
 * @typedef labeltype
 * @brief Type for element labels (identifiers).
 *
 * Each element in the index has a label that identifies it.
 * This is typically an integer (size_t), but could be any unique ID.
 */
typedef size_t labeltype;

/**
 * @class BaseFilterFunctor
 * @brief Base class for filtering search results.
 *
 * Use this to filter results based on custom criteria.
 * For example, only return results from a specific set.
 *
 * USAGE:
 *    class MyFilter : public BaseFilterFunctor {
 *        std::set<labeltype> allowed;
 *    public:
 *        MyFilter(const std::set<labeltype>& allowed) : allowed(allowed) {}
 *        bool operator()(labeltype id) override {
 *            return allowed.count(id) > 0;
 *        }
 *    };
 */
class BaseFilterFunctor {
 public:
    /**
     * @brief Check if a label should be included in results.
     * @param id The label to check
     * @return true if included, false to exclude
     */
    virtual bool operator()(hnswlib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

/**
 * @class BaseSearchStopCondition
 * @brief Abstract class for controlling search termination.
 *
 * This allows custom early-stopping logic. The search algorithm
 * calls these methods to decide whether to continue or stop.
 *
 * Use cases:
 *   - Multi-document search (stop when k docs found)
 *   - Epsilon search (stop when distance > epsilon)
 *   - Time-bounded search
 *
 * @tparam dist_t Distance type (float, int, etc.)
 */
template<typename dist_t>
class BaseSearchStopCondition {
 public:
    /**
     * @brief Called when a candidate is added to results.
     * @param label The element's label
     * @param datapoint Pointer to the element's data
     * @param dist Distance to query
     */
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    /**
     * @brief Called when a candidate is removed from results.
     * @param label The element's label
     * @param datapoint Pointer to the element's data
     * @param dist Distance to query
     */
    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    /**
     * @brief Should we stop the search?
     * @param candidate_dist Distance of current candidate
     * @param lowerBound Best distance found so far
     * @return true to stop searching
     */
    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    /**
     * @brief Should we consider this candidate?
     * @param candidate_dist Distance of candidate
     * @param lowerBound Best distance found so far
     * @return true to add to candidates
     */
    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    /**
     * @brief Should we remove the worst result?
     * @return true to remove extra results
     */
    virtual bool should_remove_extra() = 0;

    /**
     * @brief Final filtering of results.
     * @param candidates Results to filter
     */
    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

/**
 * @class pairGreater
 * @brief Functor for max-heap ordering.
 *
 * std::priority_queue defaults to max-heap with less<>.
 * This gives us min-heap (smallest element on top).
 * Use this when you want the CLOSEST result on top.
 *
 * @tparam T The type stored in the pair
 */
template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

/**
 * @brief Write a POD type to binary stream.
 *
 * Used for saving index to disk.
 *
 * @tparam T Plain Old Data type
 * @param out Output stream
 * @param podRef Value to write
 */
template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

/**
 * @brief Read a POD type from binary stream.
 *
 * Used for loading index from disk.
 *
 * @tparam T Plain Old Data type
 * @param in Input stream
 * @param podRef Variable to receive read value
 */
template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

/**
 * @typedef DISTFUNC
 * @brief Type for distance functions.
 *
 * A distance function takes:
 *   - Pointer to first vector
 *   - Pointer to second vector
 *   - Pointer to dimension (size_t*)
 *
 * Returns: distance (float for L2/IP, int for integer)
 */
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

/**
 * @class SpaceInterface
 * @brief Abstract interface for distance spaces.
 *
 * A "space" defines how to compute distances between vectors.
 * Implement this to create custom distance metrics.
 *
 * @tparam MTYPE Distance type (float, int, etc.)
 */
template<typename MTYPE>
class SpaceInterface {
 public:
    /**
     * @brief Get size of vector data in bytes.
     */
    virtual size_t get_data_size() = 0;

    /**
     * @brief Get the distance function.
     */
    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    /**
     * @brief Get parameter for distance function (typically dimension).
     */
    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface() {}
};

/**
 * @class AlgorithmInterface
 * @brief Abstract interface for ANN algorithms.
 *
 * Both HNSW and BruteForce implement this interface,
 * making them interchangeable in code.
 *
 * @tparam dist_t Distance type
 */
template<typename dist_t>
class AlgorithmInterface {
 public:
    /**
     * @brief Add a point to the index.
     * @param datapoint Pointer to vector data
     * @param label Unique label for this point
     * @param replace_deleted If true, reuse deleted slots
     */
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    /**
     * @brief Search for k nearest neighbors.
     * @param query_data Pointer to query vector
     * @param k Number of results to return
     * @param isIdAllowed Optional filter
     * @return Priority queue of (distance, label) pairs, closest first
     */
    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    /**
     * @brief Search returning results closest-first order.
     *
     * Note: searchKnn returns in closest-first order by default
     * due to pairGreater comparator.
     *
     * @param query_data Pointer to query vector
     * @param k Number of results
     * @param isIdAllowed Optional filter
     * @return Vector of (distance, label), closest first
     */
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    /**
     * @brief Save index to disk.
     * @param location File path
     */
    virtual void saveIndex(const std::string &location) = 0;
    virtual ~AlgorithmInterface(){
    }
};

/**
 * @brief SearchKnnCloserFirst implementation.
 *
 * Converts priority queue results to vector in closest-first order.
 */
template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                         BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswlib

// Include all headers
#include "space_l2.h"        // Euclidean distance
#include "space_ip.h"         // Inner product distance
#include "stop_condition.h"  // Advanced search conditions
#include "bruteforce.h"     // Brute-force fallback
#include "hnswalg.h"       // Main HNSW implementation
