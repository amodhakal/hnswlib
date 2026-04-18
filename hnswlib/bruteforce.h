#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace hnswlib {

/**
 * @file bruteforce.h
 * @brief Brute-force (exact) nearest neighbor search implementation.
 *
 * This provides an exact (= brute-force) ANN index as a fallback or comparison.
 * While not scalable, it gives perfect recall and is useful for:
 *   - Small datasets (up to ~10K elements)
 *   - Ground truth for evaluating HNSW accuracy
 *   - When exact results are required
 *
 * The implementation stores all vectors in a flat array and computes
 * distances sequentially during search.
 *
 * PERFORMANCE:
 *   - Add: O(1) amortized
 *   - Search: O(n) where n = number of elements
 *   - Memory: O(n * dim)
 *
 * This is the "dumb" baseline that HNSW is compared against.
 */

/**
 * @class BruteforceSearch
 * @brief Brute-force (exact) nearest neighbor search.
 *
 * Stores vectors in a contiguous array. For search, computes
 * distance to every vector and returns the k closest.
 *
 * Thread-safe with mutex locking.
 *
 * @tparam dist_t Distance type (float, int, etc.)
 */
template<typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
    char *data_;                        ///< Contiguous storage for all vectors + labels
    size_t maxelements_;               ///< Maximum elements (capacity)
    size_t cur_element_count;        ///< Current element count
    size_t size_per_element_;        ///< Bytes per element (vector + label)

    size_t data_size_;               ///< Size of vector data in bytes
    DISTFUNC <dist_t> fstdistfunc_;  ///< Distance function
    void *dist_func_param_;           ///< Parameter for distance function
    std::mutex index_lock;          ///< Mutex for thread safety

    /**
     * @brief Map from external label to internal index.
     *
     * External labels are user-provided IDs (any unique value).
     * Internal indices are positions in our data array.
     * This maps between them.
     */
    std::unordered_map<labeltype, size_t > dict_external_to_internal;


    /**
     * @brief Create empty brute-force index.
     * @param s The space (defines distance function)
     */
    BruteforceSearch(SpaceInterface <dist_t> *s)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
    }


    /**
     * @brief Load index from file.
     * @param s The space
     * @param location File path
     */
    BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
        loadIndex(location, s);
    }


    /**
     * @brief Create brute-force index with capacity.
     * @param s The space
     * @param maxElements Maximum number of elements
     */
    BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
        maxelements_ = maxElements;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxElements * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count = 0;
    }


    /**
     * @brief Destructor - free allocated memory.
     */
    ~BruteforceSearch() {
        free(data_);
    }


    /**
     * @brief Add a point to the index.
     *
     * If label already exists, update its vector.
     * Otherwise, add as new element.
     *
     * @param datapoint Pointer to vector data
     * @param label Unique identifier
     * @param replace_deleted Ignored (for API compatibility)
     */
    void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) {
        int idx;
        {
            std::unique_lock<std::mutex> lock(index_lock);

            auto search = dict_external_to_internal.find(label);
            if (search != dict_external_to_internal.end()) {
                idx = search->second;
            } else {
                if (cur_element_count >= maxelements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit\n");
                }
                idx = cur_element_count;
                dict_external_to_internal[label] = idx;
                cur_element_count++;
            }
        }
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
    }


    /**
     * @brief Remove a point from the index.
     *
     * Swaps with last element to maintain contiguity.
     *
     * @param cur_external Label to remove
     */
    void removePoint(labeltype cur_external) {
        std::unique_lock<std::mutex> lock(index_lock);

        auto found = dict_external_to_internal.find(cur_external);
        if (found == dict_external_to_internal.end()) {
            return;
        }

        dict_external_to_internal.erase(found);

        size_t cur_c = found->second;
        labeltype label = *((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
        dict_external_to_internal[label] = cur_c;
        memcpy(data_ + size_per_element_ * cur_c,
                data_ + size_per_element_ * (cur_element_count-1),
                data_size_+sizeof(labeltype));
        cur_element_count--;
    }


    /**
     * @brief Search for k nearest neighbors.
     *
     * Computes distance to ALL elements, returns k closest.
     * This is O(n) where n = element count.
     *
     * @param query_data Query vector
     * @param k Number of results
     * @param isIdAllowed Optional filter
     * @return Priority queue (dist, label) pairs, closest first
     */
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        assert(k <= cur_element_count);
        std::priority_queue<std::pair<dist_t, labeltype >> topResults;
        dist_t lastdist = std::numeric_limits<dist_t>::max();
        for (int i = 0; i < cur_element_count; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            if (dist <= lastdist || topResults.size() < k) {
                labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.emplace(dist, label);
                    if (topResults.size() > k)
                        topResults.pop();
                    if (!topResults.empty())
                        lastdist = topResults.top().first;
                }
            }
        }
        return topResults;
    }


    /**
     * @brief Save index to file.
     * @param location File path
     */
    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, maxelements_);
        writeBinaryPOD(output, size_per_element_);
        writeBinaryPOD(output, cur_element_count);

        output.write(data_, maxelements_ * size_per_element_);

        output.close();
    }


    /**
     * @brief Load index from file.
     * @param location File path
     * @param s The space
     */
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
        std::ifstream input(location, std::ios::binary);
        std::streampos position;

        readBinaryPOD(input, maxelements_);
        readBinaryPOD(input, size_per_element_);
        readBinaryPOD(input, cur_element_count);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

        input.read(data_, maxelements_ * size_per_element_);

        input.close();
    }
};
}  // namespace hnswlib
