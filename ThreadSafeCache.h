#pragma once

#include <cassert>
#include <cmath>
#include <forward_list>
#include <vector>

#define expects(x) assert(x)
#define ensures(x) assert(x)

template <typename Key, typename T>
class ThreadSafeCache
{
public:
    using size_type = std::size_t;

    explicit ThreadSafeCache(size_type bucket_count)
    : m_buckets(std::max(bucket_count, size_type(1)))
    {
    }

    ThreadSafeCache()
    : ThreadSafeCache(k_default_bucket_count)
    {
    }

    template <typename Iterator>
    ThreadSafeCache(Iterator first, Iterator last, size_type bucket_count = k_default_bucket_count)
    : m_buckets(std::max(bucket_count, size_type(1)))
    {
    }

    template <typename F>
    const T& lookup(const Key& key, F&& creator) const
    {
        // Read lock on bucket list
        // Write lock on linked list

        // Lookup value. If there, return
        // Else create
        // No sentinel needed: we have a write lock

        // Read lock still active...
        // If we are over our bucket capacity
        // Create new

        // Release read lock
        // Swap
    }

    const T& lookup(const Key& key) const
    {
        // Read lock on bucket list
        // Read lock on linked list

        // Lookup value. If not there, throw
    }

    // While technically thread-safe, use this with caution if there are other active threads.
    void swap(ThreadSafeCache& other) noexcept
    {
        // Write lock on bucket list.
        // Swap
    }

    float load_factor() const
    {
        // TODO
        return 1.0f;
    }

    float max_load_factor() const noexcept
    {
        // Read lock
        return m_max_load_factor;
    }

    void max_load_factor(float ml) noexcept
    {
        // Write lock
        m_max_load_factor = ml;
    }

    void rehash(size_type count)
    {
    }

    void reserve(size_type count)
    {
    }

private:
    static constexpr size_type k_default_bucket_count{32};

    using ElementList = std::forward_list<T>;
    using BucketList  = std::vector<ElementList>;

    static double mod1(double x) noexcept
    {
        return x - std::floor(x);
    }

    static std::size_t get_bucket_index(std::size_t hash, std::size_t num_buckets) noexcept
    {
        // Hashing by multiplication
        constexpr double phi = 1.618033988749894848204;
        return static_cast<std::size_t>(num_buckets * mod1(hash * phi));
    }

    float m_max_load_factor{1.0f};
    BucketList m_buckets;
};