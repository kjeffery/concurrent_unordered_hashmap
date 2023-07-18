#pragma once

#include <cassert>
#include <cmath>
#include <forward_list>
#include <iomanip>
#include <iostream>
#include <shared_mutex>
#include <vector>

#define expects(x) assert(x)
#define ensures(x) assert(x)

template <typename ReturnType, typename OriginalType>
inline ReturnType bitCast(OriginalType val) noexcept
{
    static_assert(sizeof(ReturnType) == sizeof(OriginalType), "Types must be of the same size.");

    union
    {
        OriginalType in;
        ReturnType   out;
    };

    in = val;
    return out;
}

inline double bitsToDouble(uint64_t n) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559, "Format must be IEEE-754");

    // Set the exponent to 1023, but leave the sign as zero. With the bias, this
    // ultimately means the exponent bits are set to zero and the exponent is
    // therefore implicitly one.  This allows us to fill in the bits for a
    // number in [1, 2), which is uniformly distributed.
    constexpr uint64_t expMask = 1023ULL << 52ULL;

    // Use n's higher-order bits by shifting past the sign and exponent into
    // the fraction. This isn't strictly necessary, in the general case, but
    // it's important for some of the QMC algorithms.
    const uint64_t asInt = expMask | (n >> 12ULL);

    // Force our bits into a floating point representation, and subtract one,
    // to get in [0, 1).
    const double f = bitCast<double>(asInt) - 1.0;

    ensures(f >= 0.0 && f < 1.0);
    return f;
}

inline double bitsToDouble(uint32_t n) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559, "Format must be IEEE-754");
    return bitsToDouble(static_cast<uint64_t>(n) << 32);
}

template <typename Key, typename T>
class ConcurrentUnorderedMap
{
    enum class CreationType
    {
        DEFAULT,
        COPY,
        GENERATE
    };

public:
    using size_type       = std::size_t;
    using hasher          = std::hash<Key>; // TODO: template
    using value_type      = std::pair<const Key, T>;
    using reference       = value_type&;
    using const_reference = const value_type&;

    explicit ConcurrentUnorderedMap(size_type bucket_count)
    : m_buckets(std::max(bucket_count, size_type(1)))
    {
    }

    ConcurrentUnorderedMap()
    : ConcurrentUnorderedMap(k_default_bucket_count)
    {
    }

    template <typename Iterator>
    ConcurrentUnorderedMap(Iterator first, Iterator last, size_type bucket_count = k_default_bucket_count)
    : m_buckets(std::max(bucket_count, size_type(1)))
    {
        // TODO: implement
    }

    void histogram()
    {
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        for (std::size_t i = 0; i < m_buckets.size(); ++i) {
            std::cout << std::setw(4) << i << " | ";
            const auto&                   locking_list = m_buckets[i].m_list;
            std::shared_lock<SharedMutex> list_lock(m_buckets[i].m_mutex);
            const auto                    num_elements = std::distance(locking_list.cbegin(), locking_list.cend());
            std::cout << std::string(num_elements, '*') << '\n';
        }
    }

    // These return non-const references for maximum flexibility even though it's up to the user to make sure they are
    // accessed in a thead-safe manner. The map does nothing to prevent race conditions in modifying the returned
    // references. I suggest you copy them or store them in const values.
    template <typename F>
    T& find_or_generate(const Key& key, F&& creator)
    {
        return find_or_create_impl<CreationType::GENERATE>(key, std::forward<F>(creator));
    }

    T& find_or_create(const Key& key)
    {
        return find_or_create_impl<CreationType::DEFAULT>(key, DefaultConstruct{});
    }

    T& find_or_create(const Key& key, T&& model)
    {
        return find_or_create_impl<CreationType::COPY>(key, std::forward<T>(model));
    }

    T& find_or_create(const Key& key, const T& model)
    {
        return find_or_create_impl<CreationType::COPY>(key, model);
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

        const LockingList& locking_list = m_buckets[bucket_idx];
        const ElementList& element_list = locking_list.m_list;

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

    bool update(const Key& key, T value)
    {
        return update_impl(key, std::move(value));
    }

    bool update(const Key& key, T&& value)
    {
        return update_impl(key, std::forward<T>(value));
    }

    // While technically thread-safe, use this with caution if there are other active threads.
    void swap(ConcurrentUnorderedMap& other) noexcept
    {
        // Write lock on bucket list.
        // Swap
    }

    float load_factor() const
    {
        std::size_t bucket_count;
        {
            // Read lock on bucket list
            std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);
            bucket_count = m_buckets.size();
        }
        return static_cast<float>(m_num_elements) / static_cast<float>(bucket_count);
    }

    float max_load_factor() const noexcept
    {
        return m_max_load_factor;
    }

    void max_load_factor(float ml) noexcept
    {
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
                const auto   hash           = hasher{}(old_list.cbegin()->first);
                const auto   new_bucket_idx = get_bucket_index(hash, new_bucket_count);
                ElementList& new_list       = new_locking_list[new_bucket_idx].m_list;
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

    struct DefaultConstruct
    {
    };

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
        constexpr double phi  = 1.618033988749894848204;
        const auto       mult = mod1(bitsToDouble(hash) * phi);
        const auto       idx  = static_cast<std::size_t>(num_buckets * mult);
        ensures(idx < num_buckets);
        return idx;
    }

    template <CreationType>
    struct Construct;

    template <>
    struct Construct<CreationType::GENERATE>
    {
        template <typename F>
        static void construct(ElementList& list, const Key& key, F&& creator)
        {
            list.emplace_front(key, std::forward<F>(creator)(key));
        }
    };

    template <>
    struct Construct<CreationType::COPY>
    {
        static void construct(ElementList& list, const Key& key, const T& model)
        {
            list.emplace_front(key, model);
        }
    };

    template <>
    struct Construct<CreationType::DEFAULT>
    {
        static void construct(ElementList& list, const Key& key, DefaultConstruct)
        {
            list.emplace_front(key, T{});
        }
    };

    template <CreationType creation_type, typename F>
    T& find_or_create_impl(const Key& key, F&& creator)
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto num_buckets = m_buckets.size();
        const auto bucket_idx  = get_bucket_index(hasher{}(key), num_buckets);

        LockingList& locking_list = m_buckets[bucket_idx];
        ElementList& element_list = locking_list.m_list;

        // Write lock on linked list.
        // There is a chance that we're just reading the value, in which case a read lock would be just fine, but we
        // don't have boost's upgrade lock to move from a read to a write lock.
        std::unique_lock<SharedMutex> list_lock(locking_list.m_mutex);

        // Lookup value. If there, return

        auto compare = [&key](const auto& r) { return r.first == key; };

        auto it = std::find_if(element_list.begin(), element_list.end(), compare);
        if (it != element_list.cend()) {
            return it->second;
        }

        // Else create
        // No sentinel needed: we have a write lock

        Construct<creation_type>::construct(element_list, key, std::forward<F>(creator));
        // element_list.emplace_front(key, std::forward<F>(creator)(key));

        // emplace_front does not return anything until C++17, so do it the hard way...
        auto& result = element_list.front();

        const std::size_t num_elements    = ++m_num_elements;
        const auto        max_load_factor = m_max_load_factor.load(); // Only do atomic load once...
        if (num_elements > max_load_factor * num_buckets) {
            // Unlock list
            list_lock.unlock();

            // Unlock bucket
            bucket_lock.unlock();

            // 1. load_factor = num_elements / num_buckets
            // 2. num_buckets * load_factor = num_elements
            // 3. num_buckets = num_elements / load_factor

            const auto load_factor_required_buckets = static_cast<size_type>(num_elements / max_load_factor);

            rehash(std::max(num_buckets * 3u / 2u, load_factor_required_buckets * 2u));
        }

        return result.second;
    }

    bool update_impl(const Key& key, T&& value)
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto num_buckets = m_buckets.size();
        const auto bucket_idx  = get_bucket_index(hasher{}(key), num_buckets);

        LockingList& locking_list = m_buckets[bucket_idx];
        ElementList& element_list = locking_list.m_list;

        // Write lock on linked list.
        // There is a chance that we're just reading the value, in which case a read lock would be just fine, but we
        // don't have boost's upgrade lock to move from a read to a write lock.
        std::unique_lock<SharedMutex> list_lock(locking_list.m_mutex);

        // Lookup value.
        auto compare = [&key](const auto& r) { return r.first == key; };

        const auto it = std::find_if(element_list.cbegin(), element_list.cend(), compare);
        if (it != element_list.cend()) {
            it->second = std::move(value);
            return true;
        }
        return false;
    }

    mutable SharedMutex m_bucket_mutex;

    std::atomic<float>       m_max_load_factor{1.0f};
    std::atomic<std::size_t> m_num_elements{0};
    BucketList               m_buckets;
};