#pragma once

/**
 * @file visited_list_pool.h
 * @brief Thread-safe pool of visited lists for graph traversal tracking.
 *
 * During HNSW graph search, we need to track which nodes (elements) have already been
 * visited to avoid revisiting them in the same search. This file provides a memory-efficient
 * way to do that.
 *
 * KEY CONCEPT: Why use a pool?
 * - Creating a new array for every search is expensive
 * - We reuse arrays by using different "markers" (curV)
 * - When curV overflows (wraps to 0), we reset all markers
 *
 * Each visited list stores a marker value for each element in the index.
 * Instead of clearing the whole array (O(n)), we just increment curV (O(1))!
 * If curV wraps around, we do a full memset reset.
 */

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {

/**
 * @typedef vl_type
 * @brief Marker type used to mark elements as visited.
 *
 * We use unsigned short to store visit markers. Each search gets a unique
 * marker value (curV). If element[i] has marker == curV, it's been visited.
 */
typedef unsigned short int vl_type;

/**
 * @class VisitedList
 * @brief A single visited list that tracks which elements have been visited.
 *
 * Think of this as a boolean array where each element has a "visit marker".
 * Instead of resetting to false, we just use a new marker value!
 *
 * MEMORY LAYOUT:
 *   - curV: current marker value (incremented each search)
 *   - mass: array of size numelements, each stores a marker
 *
 * TRICK: Why this works:
 *   - When we start a search, we do curV++ (not memset!)
 *   - To check if visited: visited_array[id] == curV ?
 *   - Only when curV overflows to 0 do we need to reset
 */
class VisitedList {
 public:
    vl_type curV;                  ///< Current marker value (incremented each search)
    vl_type *mass;                 ///< Array of markers, one per element
    unsigned int numelements;        ///< Number of elements in the index

    /**
     * @brief Construct a visited list for tracking numelements1 elements.
     * @param numelements1 Maximum number of elements in the HNSW index
     */
    VisitedList(int numelements1) {
        curV = -1;                       // Start at -1 so first real search uses 0
        numelements = numelements1;
        mass = new vl_type[numelements];   // Allocate marker array
    }

    /**
     * @brief Prepare this visited list for a new search.
     *
     * Instead of clearing the whole array (expensive!), we just increment
     * the marker value. This is O(1) instead of O(n).
     *
     * Edge case: If curV wraps to 0, we must do a full reset
     * because we can't distinguish "not visited" from "visited with marker 0".
     */
    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    /**
     * @brief Destructor - free the marker array
     */
    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

/**
 * @class VisitedListPool
 * @brief Thread-safe pool of reusable visited lists.
 *
 * PROBLEM: In multi-threaded search, each thread needs its own visited list
 * to avoid race conditions. Creating/destroying them each time is slow.
 *
 * SOLUTION: Pool of pre-allocated visited lists!
 * - Threads grab from pool (or create new if empty)
 * - Return to pool when done (for reuse)
 * - Thread-safe with mutex locking
 *
 * USAGE PATTERN:
 *   1. Thread calls getFreeVisitedList() - gets a visited list
 *   2. Thread does its search, marking visited elements
 *   3. Thread calls releaseVisitedList() - returns to pool
 *
 * This is much faster than allocating per-search.
 */
class VisitedListPool {
    std::deque<VisitedList *> pool;    ///< Pool of available visited lists
    std::mutex poolguard;               ///< Mutex for thread-safe pool access
    int numelements;                ///< Size of each visited list

 public:
    /**
     * @brief Create a pool of visited lists.
     * @param initmaxpools Initial number of visited lists to pre-allocate
     * @param numelements1 Maximum elements in the index (size of each list)
     */
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    /**
     * @brief Get a visited list for a new search.
     *
     * If pool has one available, reuse it. Otherwise create new.
     * The returned list is already reset() for immediate use.
     *
     * @return A fresh VisitedList ready for use in search
     */
    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    /**
     * @brief Return a visited list to the pool for reuse.
     * @param vl The visited list to return
     */
    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    /**
     * @brief Destructor - clean up all visited lists in the pool
     */
    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
