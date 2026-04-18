#pragma once
#include "hnswlib.h"

namespace hnswlib {

/**
 * @file space_l2.h
 * @brief Euclidean (L2) distance functions for HNSW.
 *
 * This file provides distance functions that measure how "far apart" two vectors are
 * using the Euclidean (L2) metric. This is the straight-line distance
 * between points in d-dimensional space.
 *
 * MATH:
 *   L2 distance = sqrt(sum((x_i - y_i)^2 for all i))
 *   We use L2^2 (squared distance, without sqrt) for efficiency since:
 *   - sqrt is expensive
 *   - ordering is preserved: if d1 < d2 then sqrt(d1) < sqrt(d2)
 *
 * VECTOR REPRESENTATION:
 *   - Vectors are arrays of float values
 *   - qty_ptr points to the dimension (number of floats)
 *   - All vectors must have the same dimension
 *
 * OPTIMIZATIONS:
 *   - Scalar: Simple loop (baseline)
 *   - SSE: 4 floats at a time with 128-bit registers
 *   - AVX: 8 floats at a time with 256-bit registers  
 *   - AVX512: 16 floats at a time with 512-bit registers
 *   - Residuals: Handle non-aligned dimensions
 */

/**
 * @brief Compute squared L2 (Euclidean) distance between two float vectors.
 *
 * This is a simple scalar implementation - baseline for correctness.
 * Processes one float at a time.
 *
 * @param pVect1v Pointer to first vector (array of floats)
 * @param pVect2v Pointer to second vector (array of floats)
 * @param qty_ptr Pointer to dimension (size_t). *qty_ptr = number of floats
 * @return Squared L2 distance: sum((x_i - y_i)^2)
 *
 * @note We return squared distance (not sqrt) for efficiency.
 *       sqrt is monotonic, so comparing d^2 is same as comparing d.
 */
static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX512)

/**
 * @brief AVX512 optimized squared L2 distance (16 floats at once).
 *
 * AVX512 processes 16 floats simultaneously using 512-bit registers.
 * This is the fastest option on modern CPUs.
 *
 * @param pVect1v Pointer to first vector
 * @param pVect2v Pointer to second vector  
 * @param qty_ptr Pointer to dimension
 * @return Squared L2 distance
 *
 * @note Requires CPU with AVX-512F (Foundation) support
 */
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

/**
 * @brief AVX optimized squared L2 distance (8 floats at once).
 *
 * AVX processes 8 floats simultaneously using 256-bit registers.
 * Processes 16 floats per iteration (two 8-float operations).
 *
 * @param pVect1v Pointer to first vector
 * @param pVect2v Pointer to second vector  
 * @param qty_ptr Pointer to dimension
 * @return Squared L2 distance
 *
 * @note Requires CPU with AVX support
 */
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

/**
 * @brief SSE optimized squared L2 distance (4 floats at once).
 *
 * SSE processes 4 floats simultaneously using 128-bit registers.
 * Processes 16 floats per iteration (four 4-float operations).
 *
 * @param pVect1v Pointer to first vector
 * @param pVect2v Pointer to second vector  
 * @param qty_ptr Pointer to dimension
 * @return Squared L2 distance
 *
 * @note Works on any CPU with SSE support (virtually all x86)
 */
static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)

/**
 * @brief Global pointer to the best available SIMD L2 function.
 *
 * Auto-selects based on CPU capabilities:
 * - AVX512 if available (fastest)
 * - AVX if available
 * - SSE otherwise (baseline)
 */
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

/**
 * @brief Handle dimension not divisible by 16 using SIMD with residuals.
 *
 * SIMD works best on 16-float chunks. For dimensions not divisible by 16,
 * we process the aligned portion with SIMD, then the remaining
 * elements with scalar code.
 *
 * @param pVect1v Pointer to first vector
 * @param pVect2v Pointer to second vector
 * @param qty_ptr Pointer to dimension (any size)
 * @return Squared L2 distance
 *
 * @example
 *   dim=50: process 48 with SIMD, 2 with scalar
 */
static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#if defined(USE_SSE)

/**
 * @brief SSE optimized squared L2 for dimension divisible by 4.
 *
 * Uses 4-float SSE registers. Slower than 16-ext but still
 * faster than scalar for moderately sized vectors.
 *
 * @param pVect1v Pointer to first vector
 * @param pVect2v Pointer to second vector  
 * @param qty_ptr Pointer to dimension (MUST be divisible by 4)
 * @return Squared L2 distance
 */
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

/**
 * @brief SSE 4-ext with residual handling.
 *
 * For dimensions not divisible by 16 but divisible by 4.
 *
 * @param pVect1v Pointer to first vector
 * @param pVect2v Pointer to second vector
 * @param qty_ptr Pointer to dimension
 * @return Squared L2 distance
 */
static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

/**
 * @class L2Space
 * @brief Space class for Euclidean (L2) distance with auto-SIMD selection.
 *
 * This is the main entry point for L2 distance calculations.
 * It automatically selects the fastest available SIMD implementation
 * based on CPU capabilities.
 *
 * USAGE:
 *   L2Space space(dim);        // Create space for dim-dimensional vectors
 *   dist = space.get_dist_func()(vec1, vec2, space.get_dist_func_param());
 *
 * AUTO-SELECTION LOGIC:
 *   1. If AVX512 available and dim%16==0: use AVX512 (16 floats)
 *   2. Else if AVX available and dim%16==0: use AVX (8 floats x2)
 *   3. Else if SSE available and dim%4==0: use SSE (4 floats)
 *   4. Else: use scalar baseline
 *
 * RESIDUAL HANDLING:
 *   For dimensions not divisible by SIMD width:
 *   - dim%16 != 0: L2SqrSIMD16ExtResiduals
 *   - dim%4 != 0: L2SqrSIMD4ExtResiduals
 */
class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;  ///< Selected distance function
    size_t data_size_;              ///< Size of vector in bytes
    size_t dim_;                   ///< Dimension (number of floats)

 public:
    /**
     * @brief Create an L2 space for vectors of given dimension.
     * @param dim Vector dimension (number of floats)
     *
     * Automatically selects fastest SIMD implementation.
     */
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    /**
     * @brief Get size of vector data in bytes.
     */
    size_t get_data_size() {
        return data_size_;
    }

    /**
     * @brief Get the selected distance function.
     */
    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    /**
     * @brief Get pointer to dimension (used as parameter to distance func).
     */
    void *get_dist_func_param() {
        return &dim_;
    }

    /**
     * @brief Destructor (nothing to free for this class).
     */
    ~L2Space() {}
};

/**
 * @brief Integer L2 distance with 4-byte batches.
 *
 * Optimized for unsigned char vectors, comparing 4 bytes at a time.
 * Returns squared distance as int (no floating point needed).
 *
 * @param pVect1 Pointer to first vector (unsigned chars)
 * @param pVect2 Pointer to second vector (unsigned chars)
 * @param qty_ptr Pointer to byte count
 * @return Squared L2 distance as int
 */
static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;
    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

/**
 * @brief Integer L2 distance (scalar, one byte at a time).
 *
 * Baseline integer L2 for arbitrary byte counts.
 *
 * @param pVect1 Pointer to first vector (unsigned chars)
 * @param pVect2 Pointer to second vector (unsigned chars)
 * @param qty_ptr Pointer to byte count
 * @return Squared L2 distance as int
 */
static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

/**
 * @class L2SpaceI
 * @brief Space class for integer (byte) vectors.
 *
 * Similar to L2Space but for unsigned char (byte) vectors.
 * Uses integer distance (no floats needed for byte values).
 *
 * Use case: Binary features, byte-encoded data.
 */
class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;      ///< Selected distance function
    size_t data_size_;              ///< Size in bytes
    size_t dim_;                   ///< Dimension (number of bytes)

 public:
    /**
     * @brief Create integer L2 space.
     * @param dim Number of bytes per vector
     */
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrI4x;
        } else {
            fstdistfunc_ = L2SqrI;
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    /**
     * @brief Get data size in bytes.
     */
    size_t get_data_size() {
        return data_size_;
    }

    /**
     * @brief Get distance function.
     */
    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    /**
     * @brief Get dimension parameter.
     */
    void *get_dist_func_param() {
        return &dim_;
    }

    /**
     * @brief Destructor.
     */
    ~L2SpaceI() {}
};
}  // namespace hnswlib
