#pragma once
#include "space_l2.h"
#include "space_ip.h"
#include <assert.h>
#include <unordered_map>

namespace hnswlib {

/**
 * @file stop_condition.h
 * @brief Advanced search stop conditions and multi-vector spaces.
 *
 * This file provides:
 * 1. BaseMultiVectorSpace: Space that supports multiple vectors per doc
 * 2. Custom stop conditions: Control when search terminates early
 * 3. Search filtering: Return diverse results from same document
 *
 * USE CASE: When one document has multiple vectors (e.g., multiple images,
 * paragraphs, or embeddings), and you want to search by document rather
 * than by individual vector.
 *
 * EXAMPLE:
 *   - Document A has vectors [v1, v2, v3]
 *   - Document B has vectors [u1, u2]
 *   - Without multi-vector: search returns v1, v2, u1, u2
 *   - With multi-vector: search returns Document A, Document B
 *     ( deduplicates by document ID)
 */

/**
 * @class BaseMultiVectorSpace
 * @brief Abstract space that tracks document IDs.
 *
 * In standard HNSW, each element has one vector. But sometimes
 * you want multiple vectors per document (e.g., image search
 * where one image has multiple regional crops).
 *
 * This abstracts a space where vectors are paired with doc IDs.
 * The doc ID is stored alongside the vector data.
 *
 * @tparam DOCIDTYPE Type for document ID (e.g., int, size_t)
 */
template<typename DOCIDTYPE>
class BaseMultiVectorSpace : public SpaceInterface<float> {
 public:
    /**
     * @brief Extract document ID from a data point.
     * @param datapoint Pointer to vector data (plus doc ID at end)
     * @return The document ID
     */
    virtual DOCIDTYPE get_doc_id(const void *datapoint) = 0;

    /**
     * @brief Set document ID in a data point.
     * @param datapoint Pointer to vector data
     * @param doc_id The document ID to store
     */
    virtual void set_doc_id(void *datapoint, DOCIDTYPE doc_id) = 0;
};


/**
 * @class MultiVectorL2Space
 * @brief L2 space with document ID tracking.
 *
 * Stores vector + DOCIDTYPE together. The doc ID is placed after
 * the vector data, so the total size is:
 *   data_size = dim * sizeof(float) + sizeof(DOCIDTYPE)
 *
 * This allows search to return results by document, not vector.
 *
 * @tparam DOCIDTYPE Type for document ID
 */
template<typename DOCIDTYPE>
class MultiVectorL2Space : public BaseMultiVectorSpace<DOCIDTYPE> {
    DISTFUNC<float> fstdistfunc_;      ///< Selected distance function
    size_t data_size_;                 ///< Total size (vector + doc ID)
    size_t vector_size_;              ///< Size of vector only
    size_t dim_;                      ///< Vector dimension

 public:
    /**
     * @brief Create multi-vector L2 space.
     * @param dim Vector dimension (NOT including doc ID)
     */
    MultiVectorL2Space(size_t dim) {
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
        vector_size_ = dim * sizeof(float);
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    }

    /**
     * @brief Get total data size (vector + doc ID).
     */
    size_t get_data_size() override {
        return data_size_;
    }

    /**
     * @brief Get selected distance function.
     */
    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    /**
     * @brief Get dimension parameter.
     */
    void *get_dist_func_param() override {
        return &dim_;
    }

    /**
     * @brief Get doc ID from data point.
     */
    DOCIDTYPE get_doc_id(const void *datapoint) override {
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    }

    /**
     * @brief Set doc ID in data point.
     */
    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) override {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    }

    /**
     * @brief Destructor.
     */
    ~MultiVectorL2Space() {}
};


/**
 * @class MultiVectorInnerProductSpace
 * @brief Inner Product space with document ID tracking.
 *
 * Similar to MultiVectorL2Space but with inner product distance.
 *
 * @tparam DOCIDTYPE Type for document ID
 */
template<typename DOCIDTYPE>
class MultiVectorInnerProductSpace : public BaseMultiVectorSpace<DOCIDTYPE> {
    DISTFUNC<float> fstdistfunc_;      ///< Selected distance function
    size_t data_size_;                 ///< Total size (vector + doc ID)
    size_t vector_size_;              ///< Size of vector only
    size_t dim_;                      ///< Vector dimension

 public:
    /**
     * @brief Create multi-vector Inner Product space.
     * @param dim Vector dimension (NOT including doc ID)
     */
    MultiVectorInnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        vector_size_ = dim * sizeof(float);
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    }

    /**
     * @brief Get total data size (vector + doc ID).
     */
    size_t get_data_size() override {
        return data_size_;
    }

    /**
     * @brief Get selected distance function.
     */
    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    /**
     * @brief Get dimension parameter.
     */
    void *get_dist_func_param() override {
        return &dim_;
    }

    /**
     * @brief Get doc ID from data point.
     */
    DOCIDTYPE get_doc_id(const void *datapoint) override {
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    }

    /**
     * @brief Set doc ID in data point.
     */
    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) override {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    }

    /**
     * @brief Destructor.
     */
    ~MultiVectorInnerProductSpace() {}
};


/**
 * @class MultiVectorSearchStopCondition
 * @brief Stop condition for multi-document search.
 *
 * PROBLEM: When one document has multiple vectors, we want to return
 * results by document, not by individual vector. But standard HNSW
 * search returns the top-k nearest VECTORS, which may all belong to
 * the same document.
 *
 * SOLUTION: Track which documents we've seen, and deduplicate!
 *
 * HOW IT WORKS:
 *   - We want num_docs_to_search unique documents
 *   - ef_collection is the search width (like ef in HNSW)
 *   - As we find candidates, we track their document IDs
 *   - When we've found enough documents, we can stop early
 *
 * @tparam DOCIDTYPE Type for document ID
 * @tparam dist_t Distance type (float, int, etc.)
 */
template<typename DOCIDTYPE, typename dist_t>
class MultiVectorSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    size_t curr_num_docs_;                ///< Current unique documents found
    size_t num_docs_to_search_;            ///< Target number of documents
    size_t ef_collection_;                ///< Search width (like ef in HNSW)
    std::unordered_map<DOCIDTYPE, size_t> doc_counter_;  ///< Count vectors per doc
    std::priority_queue<std::pair<dist_t, DOCIDTYPE>> search_results_;  ///< Deduped results
    BaseMultiVectorSpace<DOCIDTYPE>& space_;  ///< The space containing vectors

 public:
    /**
     * @brief Create multi-vector stop condition.
     * @param space The multi-vector space
     * @param num_docs_to_search Target unique documents to find
     * @param ef_collection Search width (default 10)
     */
    MultiVectorSearchStopCondition(
        BaseMultiVectorSpace<DOCIDTYPE>& space,
        size_t num_docs_to_search,
        size_t ef_collection = 10)
        : space_(space) {
            curr_num_docs_ = 0;
            num_docs_to_search_ = num_docs_to_search;
            ef_collection_ = std::max(ef_collection, num_docs_to_search);
        }

    /**
     * @brief Add a found point to results, deduplicating by document.
     */
    void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
        DOCIDTYPE doc_id = space_.get_doc_id(datapoint);
        if (doc_counter_[doc_id] == 0) {
            curr_num_docs_ += 1;
        }
        search_results_.emplace(dist, doc_id);
        doc_counter_[doc_id] += 1;
    }

    /**
     * @brief Remove a point from results.
     */
    void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) override {
        DOCIDTYPE doc_id = space_.get_doc_id(datapoint);
        doc_counter_[doc_id] -= 1;
        if (doc_counter_[doc_id] == 0) {
            curr_num_docs_ -= 1;
        }
        search_results_.pop();
    }

    /**
     * @brief Stop if we have enough documents and candidate can't improve.
     */
    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
        bool stop_search = candidate_dist > lowerBound && curr_num_docs_ == ef_collection_;
        return stop_search;
    }

    /**
     * @brief Consider candidate if we need more docs or it's better.
     */
    bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) override {
        bool flag_consider_candidate = curr_num_docs_ < ef_collection_ || lowerBound > candidate_dist;
        return flag_consider_candidate;
    }

    /**
     * @brief Remove extra if we have too many unique docs.
     */
    bool should_remove_extra() override {
        bool flag_remove_extra = curr_num_docs_ > ef_collection_;
        return flag_remove_extra;
    }

    /**
     * @brief Filter final results to requested number of documents.
     */
    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) override {
        while (curr_num_docs_ > num_docs_to_search_) {
            dist_t dist_cand = candidates.back().first;
            dist_t dist_res = search_results_.top().first;
            assert(dist_cand == dist_res);
            DOCIDTYPE doc_id = search_results_.top().second;
            doc_counter_[doc_id] -= 1;
            if (doc_counter_[doc_id] == 0) {
                curr_num_docs_ -= 1;
            }
            search_results_.pop();
            candidates.pop_back();
        }
    }

    /**
     * @brief Destructor.
     */
    ~MultiVectorSearchStopCondition() {}
};


/**
 * @class EpsilonSearchStopCondition
 * @brief Stop condition based on distance threshold (epsilon).
 *
 * PROBLEM: Standard HNSW returns exactly k results. But what if you want
 * all results within a certain distance threshold?
 *
 * SOLUTION: Use epsilon-based stopping!
 *
 * PARAMETERS:
 *   - epsilon: Maximum distance to consider
 *   - min_num_candidates: Minimum results to return before checking epsilon
 *   - max_num_candidates: Maximum results, even if within epsilon
 *
 * STOPPING CONDITIONS:
 *   1. We have max candidates and candidate is worse than best found
 *   2. Candidate distance > epsilon AND we have minimum candidates
 *
 * @tparam dist_t Distance type (float, int, etc.)
 *
 * @example
 *   epsilon=0.1, min=10, max=100
 *   - Keep searching until we have 100 results OR
 *   - Stop early if candidate > 0.1 AND we have 10+ results
 */
template<typename dist_t>
class EpsilonSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    float epsilon_;                    ///< Maximum distance threshold
    size_t min_num_candidates_;      ///< Minimum results before checking epsilon
    size_t max_num_candidates_;       ///< Maximum results to collect
    size_t curr_num_items_;           ///< Current results collected

 public:
    /**
     * @brief Create epsilon stop condition.
     * @param epsilon Maximum distance threshold
     * @param min_num_candidates Minimum results to return
     * @param max_num_candidates Maximum results to return
     */
    EpsilonSearchStopCondition(float epsilon, size_t min_num_candidates, size_t max_num_candidates) {
        assert(min_num_candidates <= max_num_candidates);
        epsilon_ = epsilon;
        min_num_candidates_ = min_num_candidates;
        max_num_candidates_ = max_num_candidates;
        curr_num_items_ = 0;
    }

    /**
     * @brief Add candidate to results.
     */
    void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
        curr_num_items_ += 1;
    }

    /**
     * @brief Remove candidate from results.
     */
    void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) override {
        curr_num_items_ -= 1;
    }

    /**
     * @brief Determine if search should stop.
     *
     * Two early-stop conditions:
     * 1. No room for improvement: candidate > lowerBound AND at max
     * 2. Outside epsilon: candidate > epsilon AND have minimum
     */
    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
        if (candidate_dist > lowerBound && curr_num_items_ == max_num_candidates_) {
            // new candidate can't improve found results
            return true;
        }
        if (candidate_dist > epsilon_ && curr_num_items_ >= min_num_candidates_) {
            // new candidate is out of epsilon region and
            // minimum number of candidates is checked
            return true;
        }
        return false;
    }

    /**
     * @brief Consider candidate if we have room or it's good.
     */
    bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) override {
        bool flag_consider_candidate = curr_num_items_ < max_num_candidates_ || lowerBound > candidate_dist;
        return flag_consider_candidate;
    }

    /**
     * @brief Remove if over max candidates.
     */
    bool should_remove_extra() {
        bool flag_remove_extra = curr_num_items_ > max_num_candidates_;
        return flag_remove_extra;
    }

    /**
     * @brief Filter results to those within epsilon.
     */
    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) override {
        while (!candidates.empty() && candidates.back().first > epsilon_) {
            candidates.pop_back();
        }
        while (candidates.size() > max_num_candidates_) {
            candidates.pop_back();
        }
    }

    /**
     * @brief Destructor.
     */
    ~EpsilonSearchStopCondition() {}
};
}  // namespace hnswlib
