#pragma once

#include <cassert>
#include <cmath>
#include <forward_list>
#include <shared_mutex>
#include <vector>

#define expects(x) assert(x)
#define ensures(x) assert(x)

template <typename Key, typename T>
class ThreadSafeCache
{
public:
    using size_type       = std::size_t;
    using hasher          = std::hash<Key>; // TODO: template
    using value_type      = std::pair<const Key, T>;
    using reference       = value_type&;
    using const_reference = const value_type&;

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

    // TODO: can these return references? We have to be careful about rehashing not invalidating references.
    template <typename F>
    const T& find_or_create(const Key& key, F&& creator)
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto bucket_count = m_buckets.size();
        const auto bucket_idx   = get_bucket_index(hasher{}(key), bucket_count);

        LockingList& locking_list = m_buckets[bucket_idx];
        ElementList& element_list = locking_list.m_list;

        // Write lock on linked list.
        // There is a chance that we're just reading the value, in which case a read lock would be just fine, but we
        // don't have boost's upgrade lock to move from a read to a write lock.
        std::unique_lock<SharedMutex> list_lock(locking_list.m_mutex);

        // Lookup value. If there, return

        auto compare = [&key](const auto& r) { return r.first == key; };

        const auto it = std::find_if(element_list.cbegin(), element_list.cend(), compare);
        if (it != element_list.cend()) {
            return it->second;
        }

        // Else create
        // No sentinel needed: we have a write lock

        element_list.emplace_front(key, creator(key));

        // emplace_front does not return anything until C++17, so do it the hard way...
        const T& ret = element_list.front().second;

        std::size_t num_elements;
        {
            std::unique_lock<SharedMutex> num_elements_lock(m_num_elements_mutex);
            num_elements = ++m_num_elements;
        }

        if (num_elements > max_load_factor() * bucket_count) {
            // Undo list lock
            locking_list.m_mutex.unlock();
            // Undo bucket lock
            m_bucket_mutex.unlock();



            // TODO: max of this and load_factor
            rehash(bucket_count * 3u / 2u);
        }

        return ret;
    }

    // C++17: optional
#if 0
    const T& find(const Key& key) const
    {
        // Read lock on bucket list
        // Read lock on linked list

        // Lookup value. If not there, throw
    }
#endif

    const T& at(const Key& key) const
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto bucket_count = m_buckets.size();
        const auto bucket_idx   = get_bucket_index(hasher{}(key), bucket_count);

        LockingList& locking_list = m_buckets[bucket_idx];
        ElementList& element_list = locking_list.m_list;

        // Read lock on linked list
        std::shared_lock<SharedMutex> list_lock(locking_list.m_mutex);

        // Lookup value. If there, return
        auto compare = [&key](const auto& r) { return r.first == key; };

        const auto it = std::find_if(element_list.cbegin(), element_list.cend(), compare);
        if (it == element_list.cend()) {
            throw std::out_of_range{"No key for 'at'"};
        }
        return it->second;
    }

    void update(const Key& key, T value)
    {

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

    void rehash(size_type new_bucket_count)
    {
        std::unique_lock<SharedMutex> bucket_lock(m_bucket_mutex);
        if (m_buckets.size() >= new_bucket_count) {
            return;
        }
        BucketList new_locking_list(new_bucket_count);

        // We go through the trouble of splicing the lists so that we don't invalidate references held outside of the
        // class.
        for (auto& bucket : m_buckets) {
            ElementList& old_list = bucket.m_list;
            while (!old_list.empty()) {
                const auto hash = hasher{}(old_list.cbegin()->first);
                const auto new_bucket_idx = get_bucket_index(hash, new_bucket_count);
                ElementList& new_list = new_locking_list[new_bucket_idx].m_list;
                new_list.splice_after(new_list.cbefore_begin(), old_list, old_list.cbefore_begin());
            }
        }

        m_buckets.swap(new_locking_list);
    }

    void reserve(size_type element_count)
    {
        rehash(static_cast<size_type>(std::ceil(element_count / max_load_factor())));
    }

private:
    static constexpr size_type k_default_bucket_count{32};

    using SharedMutex = std::shared_timed_mutex;

    using ElementList = std::forward_list<value_type>;

    struct LockingList
    {
        ElementList         m_list;
        mutable SharedMutex m_mutex;
    };

    using BucketList = std::vector<LockingList>;

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

    mutable SharedMutex m_bucket_mutex;
    mutable SharedMutex m_num_elements_mutex;

    float       m_max_load_factor{1.0f};
    std::size_t m_num_elements{0};
    BucketList  m_buckets;
};