#pragma once

#include <mutex>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring> // memcpy
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

    ReturnType out;
    std::memcpy(&out, &val, sizeof(ReturnType));
    return out;
}

inline double bitsToDouble(uint64_t n) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559, "Format must be IEEE-754");

    // Set the exponent to 1023, but leave the sign as zero. With the bias, this ultimately means the exponent bits are
    // set to zero and the exponent is therefore implicitly one.  This allows us to fill in the bits for a number in [1,
    // 2), which is uniformly distributed.
    constexpr uint64_t expMask = 1023ULL << 52ULL;

    // Use n's higher-order bits by shifting past the sign and exponent into the fraction. This isn't strictly
    // necessary, in the general case, but it's important for some of the QMC algorithms. It's also important in this
    // implementation so that we keep the mask in a valid state.
    const uint64_t asInt = expMask | (n >> 12ULL);

    // Force our bits into a floating point representation, and subtract one, to get in [0, 1).
    const double f = bitCast<double>(asInt) - 1.0;

    ensures(f >= 0.0 && f < 1.0);
    return f;
}

inline double bitsToDouble(uint32_t n) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559, "Format must be IEEE-754");
    return bitsToDouble(static_cast<uint64_t>(n) << 32);
}

template <typename T>
inline T& get_key(T& v) noexcept
{
    return v;
}

template <typename T, typename U>
inline T& get_key(std::pair<T, U>& v) noexcept
{
    return v.first;
}

template <typename T>
inline const T& get_key(const T& v) noexcept
{
    return v;
}

template <typename T, typename U>
inline const T& get_key(const std::pair<T, U>& v) noexcept
{
    return v.first;
}

template <typename T>
inline T& get_value(T& v) noexcept
{
    return v;
}

template <typename T, typename U>
inline T& get_value(std::pair<T, U>& v) noexcept
{
    return v.second;
}

template <typename T>
inline const T& get_value(const T& v) noexcept
{
    return v;
}

template <typename T, typename U>
inline const T& get_value(const std::pair<T, U>& v) noexcept
{
    return v.second;
}

template <typename Traits>
class ConcurrentHashTable
{
public:
    using size_type       = std::size_t;
    using hasher          = typename Traits::hasher;
    using key_equal       = typename Traits::key_equal;
    using primary_type    = typename Traits::primary_type;
    using key_type        = typename Traits::key_type;
    using value_type      = typename Traits::value_type;
    using reference       = value_type&;
    using const_reference = const value_type&;

protected:
    using ElementList = std::forward_list<value_type>;

public:
    friend class iterator;
    friend class const_iterator;

    enum class IteratorType
    {
        CONST,
        NON_CONST
    };

    template <IteratorType iterator_type>
    class iterator_base
    {
        using list_iterator = std::conditional_t<iterator_type == IteratorType::CONST,
                                                 typename ElementList::const_iterator,
                                                 typename ElementList::iterator>;

        using table_type =
            std::conditional_t<iterator_type == IteratorType::CONST, const ConcurrentHashTable, ConcurrentHashTable>;

        // I don't like this bool, but it's the easiest way I can see doing a good end iterator. A value-initialized
        // list_iterator is not guaranteed to compare equal to the end iterator of a list. If we're really worried about
        // size, we could fit this into the least-significant bits of the pointer, but that's gross.
        bool m_end_iterator{true};

        list_iterator m_iterator;
        table_type*   m_table{nullptr};

    public:
        using iterator_category = typename std::iterator_traits<list_iterator>::iterator_category;
        using difference_type   = typename std::iterator_traits<list_iterator>::difference_type;
        using value_type        = typename std::iterator_traits<list_iterator>::value_type;
        using pointer           = typename std::iterator_traits<list_iterator>::pointer;
        using reference         = typename std::iterator_traits<list_iterator>::reference;

        iterator_base() = default;

        // TODO: private
        explicit iterator_base(table_type& table) noexcept
        : m_table(std::addressof(table))
        {
            ensures(is_valid_iterator());
        }

        // TODO: private
        iterator_base(table_type& table, list_iterator iterator) noexcept
        : m_end_iterator{false}
        , m_table(std::addressof(table))
        , m_iterator(std::move(iterator))
        {
            expects(is_valid_iterator());
            expects(iterator != list_iterator{}); // No value-initialized iterators, fool!
        }

        iterator_base(const iterator_base<IteratorType::NON_CONST>& other) noexcept
        : m_end_iterator(other.m_end_iterator)
        , m_table(other.m_table)
        , m_iterator(other.m_iterator)
        {
        }

        reference operator*() const noexcept
        {
            expects(!is_end_iterator());
            return *m_iterator;
        }

        pointer operator->() const noexcept
        {
            expects(!is_end_iterator());
            return m_iterator.operator->();
        }

        // Incrementing an end iterator results in undefined behavior.
        iterator_base operator++()
        {
            expects(m_table);
            expects(!is_end_iterator());
            expects(is_valid_iterator());

            std::shared_lock<SharedMutex> bucket_lock(m_table->m_bucket_mutex);
            const auto                    num_buckets = m_table->m_buckets.size();
            auto                          bucket_idx  = get_bucket_index(bucket_lock);

            // In the first step, we try to increment our iterator along our current list.
            {
                const auto&                   locking_list = m_table->m_buckets[bucket_idx].m_list;
                std::shared_lock<SharedMutex> list_lock(m_table->m_buckets[bucket_idx].m_mutex);
                ++m_iterator;
                if (m_iterator != locking_list.end()) {
                    return *this;
                } else {
                    // Potentially set up for next step, or find the end of the bucket list.
                    ++bucket_idx;
                    if (bucket_idx >= num_buckets) {
                        m_end_iterator = true;
                        return *this;
                    }
                }
            }

            // If incrementing along the list was unsuccessful, the next step is to go along the buckets until we find a
            // non-empty list. We're only interested in the begin iterator of each list.
            while (true) {
                const auto&                   locking_list = m_table->m_buckets[bucket_idx].m_list;
                std::shared_lock<SharedMutex> list_lock(m_table->m_buckets[bucket_idx].m_mutex);

                m_iterator = locking_list.begin();
                if (m_iterator == locking_list.end()) {
                    ++bucket_idx;
                    if (bucket_idx >= num_buckets) {
                        m_end_iterator = true;
                        break;
                    }
                } else {
                    // Valid iterator
                    break;
                }
            }

            ensures(is_valid_iterator());
            return *this;
        }

        iterator_base operator++(int)
        {
            iterator_base result(*this);
            this->        operator++();
            return result;
        }

        friend bool operator==(const iterator_base& a, const iterator_base& b) noexcept
        {
            const auto* const table_a = a.m_table;
            const auto* const table_b = b.m_table;

            // Value-constructed iterators
            if (table_a == nullptr && table_b == nullptr) {
                return true;
            }

            expects(a.m_table == b.m_table);
            return a.m_iterator == b.m_iterator;
        }

        friend bool operator!=(const iterator_base& a, const iterator_base& b) noexcept
        {
            return !(a == b);
        }

        bool is_valid_iterator() const noexcept
        {
            std::shared_lock<SharedMutex> bucket_lock(m_table->m_bucket_mutex);
            return is_valid_iterator(bucket_lock);
        }

    private:
        bool is_end_iterator() const noexcept
        {
            if (!m_table) {
                return true;
            }
            return m_end_iterator;
        }

        template <typename LockType>
        bool is_valid_iterator(const LockType& bucket_lock) const noexcept
        {
            expects(bucket_lock.owns_lock());
            if (is_end_iterator()) {
                return true;
            }

            const auto bucket_idx = get_bucket_index(bucket_lock);

            // Get bucket list
            const auto&                   locking_list = m_table->m_buckets[bucket_idx].m_list;
            std::shared_lock<SharedMutex> list_lock(m_table->m_buckets[bucket_idx].m_mutex);

            return m_iterator != locking_list.end();
        }

        template <typename LockType>
        std::size_t get_bucket_index(const LockType& bucket_lock) const noexcept
        {
            // We can't store the bucket index because we unlock when we're idle and the table may have been rehashed.
            expects(bucket_lock.owns_lock());
            expects(m_table);

            const auto  bucket_count = m_table->m_buckets.size();
            const auto& key          = get_key(*m_iterator);
            return table_type::get_bucket_index(hasher{}(key), bucket_count);
        }
    };

    using iterator       = iterator_base<IteratorType::NON_CONST>;
    using const_iterator = iterator_base<IteratorType::CONST>;

    explicit ConcurrentHashTable(size_type bucket_count)
    : m_buckets(std::max(bucket_count, size_type(1)))
    {
    }

    ConcurrentHashTable()
    : ConcurrentHashTable(k_default_bucket_count)
    {
    }

    template <typename Iterator>
    ConcurrentHashTable(Iterator first, Iterator last, size_type bucket_count = k_default_bucket_count)
    : m_buckets(std::max(bucket_count, size_type(1)))
    {
        // TODO: implement
    }

    void histogram() const
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

    // We don't have one shared ElementList, and things may change from another thread between this call and the caller.
    // We can't return a valid end iterator for a list that may not exist in the future, so we pair it up with a valid
    // flag.
    std::pair<typename ElementList::iterator, bool> find_impl(const key_type& key)
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto num_buckets = m_buckets.size();
        const auto bucket_idx  = get_bucket_index(hasher{}(key), num_buckets);

        LockingList& locking_list = m_buckets[bucket_idx];
        ElementList& element_list = locking_list.m_list;

        // Read lock on list
        std::shared_lock<SharedMutex> list_lock(locking_list.m_mutex);

        // Lookup value.
        auto compare = [&key](const auto& r) { return key_equal{}(get_key(r), key); };

        auto it = std::find_if(element_list.begin(), element_list.end(), compare);
        if (it != element_list.cend()) {
            return std::make_pair(it, true);
        } else {
            return std::make_pair(typename ElementList::iterator{}, false);
        }
    }

    // We don't have one shared ElementList, and things may change from another thread between this call and the caller.
    // We can't return a valid end iterator for a list that may not exist in the future, so we pair it up with a valid
    // flag.
    std::pair<typename ElementList::const_iterator, bool> find_impl(const key_type& key) const
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto num_buckets = m_buckets.size();
        const auto bucket_idx  = get_bucket_index(hasher{}(key), num_buckets);

        const LockingList& locking_list = m_buckets[bucket_idx];
        const ElementList& element_list = locking_list.m_list;

        // Read lock on list
        std::shared_lock<SharedMutex> list_lock(locking_list.m_mutex);

        // Lookup value.
        auto compare = [&key](const auto& r) { return key_equal{}(get_key(r), key); };

        auto it = std::find_if(element_list.begin(), element_list.end(), compare);
        if (it != element_list.cend()) {
            return std::make_pair(it, true);
        } else {
            return std::make_pair(typename ElementList::iterator{}, false);
        }
    }

    // While technically thread-safe, use this with caution if there are other active threads.
    void swap(ConcurrentHashTable& other) noexcept
    {
        // Write lock on bucket list.
        std::unique_lock<SharedMutex> bucket_lock(m_bucket_mutex);
        m_buckets.swap(other.m_buckets);
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
                const auto   hash           = hasher{}(get_key(old_list.front()));
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

    // This is a departure from std::unordered_map in that they don't have a size function.
    size_type size() const noexcept
    {
        return m_num_elements;
    }

protected:
    // Make the destructor protected so that we can inherit from it, but not be used virtually.
    ~ConcurrentHashTable() = default;

    static constexpr size_type k_default_bucket_count{32};

    using SharedMutex = std::shared_timed_mutex;

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
        // Implementations are allowed to use identity functions as std::hash implementations. This messes with our
        // conversion to double later, where we get really small values and end up setting everything to 0.
        hash *= 0x17f6eb688799f2cbU;

        // Hashing by multiplication
        // https://en.wikipedia.org/wiki/Hash_table#Hashing_by_multiplication
        constexpr double phi  = 1.618033988749894848204;
        const auto       mult = mod1(bitsToDouble(hash) * phi);
        const auto       idx  = static_cast<std::size_t>(num_buckets * mult);
        ensures(idx < num_buckets);
        return idx;
    }

    template <typename Creator, typename Value, typename... Args>
    std::pair<typename ElementList::iterator, bool> find_or_create_impl(Value&& key, Args&&... args)
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(m_bucket_mutex);

        const auto num_buckets = m_buckets.size();
        const auto bucket_idx  = get_bucket_index(hasher{}(key), num_buckets);

        LockingList& locking_list = m_buckets[bucket_idx];
        ElementList& element_list = locking_list.m_list;

        std::shared_lock<SharedMutex> list_lock_read(locking_list.m_mutex);

        // Lookup value. If there, return
        auto compare = [&key](const auto& r) { return key_equal{}(get_key(r), key); };

        auto it = std::find_if(element_list.begin(), element_list.end(), compare);
        if (it != element_list.cend()) {
            return std::make_pair(it, false);
        }

        list_lock_read.unlock();

        std::unique_lock<SharedMutex> list_lock_write(locking_list.m_mutex);

        // We have to check for existence again
        it = std::find_if(element_list.begin(), element_list.end(), compare);
        if (it != element_list.cend()) {
            return std::make_pair(it, false);
        }

        // Else create
        // No sentinel needed: we have a write lock

        Creator::construct(element_list, std::forward<Value>(key), std::forward<Args>(args)...);

        // emplace_front does not return anything until C++17, so do it the hard way...
        auto list_iterator = element_list.begin();

        // We can unlock before we create the iterator, because our list iterator will not be invalidated, and the
        // debug iterator does some check that require locks, leading to deadlock if we don't unlock.
        // We also want to unlock before we rehash.
        list_lock_write.unlock();
        bucket_lock.unlock();

        const std::size_t num_elements = m_num_elements.fetch_add(1u) + 1u;
        const auto max_load_factor = m_max_load_factor.load(); // Only do atomic load once...

        if (num_elements > max_load_factor * num_buckets) {
            // 1. load_factor = num_elements / num_buckets
            // 2. num_buckets * load_factor = num_elements
            // 3. num_buckets = num_elements / load_factor

            const auto load_factor_required_buckets = static_cast<size_type>(num_elements / max_load_factor);

            rehash(std::max(num_buckets * 3u / 2u, load_factor_required_buckets * 2u));
        }

        return std::make_pair(list_iterator, true);
    }

    mutable SharedMutex m_bucket_mutex;

    std::atomic<float>     m_max_load_factor{1.0f};
    std::atomic<size_type> m_num_elements{0};
    BucketList             m_buckets;
};

template <typename Key, typename Hash, typename KeyEqual>
struct ConcurrentUnorderedSetTraits
{
    using key_type     = Key;
    using primary_type = Key;
    using hasher       = Hash;
    using key_equal    = KeyEqual;
    using value_type   = Key;
};

template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class ConcurrentUnorderedSet : private ConcurrentHashTable<ConcurrentUnorderedSetTraits<Key, Hash, KeyEqual>>
{
public:
    using Traits         = ConcurrentUnorderedSetTraits<Key, Hash, KeyEqual>;
    using Base           = ConcurrentHashTable<Traits>;
    using key_type       = typename Traits::key_type;
    using value_type     = typename Traits::value_type;
    using primary_type   = typename Traits::primary_type;
    using hasher         = typename Traits::hasher;
    using key_equal      = typename Traits::key_equal;
    using iterator       = typename Base::const_iterator;
    using const_iterator = typename Base::const_iterator;
    using ElementList    = typename Base::ElementList;
    using LockingList    = typename Base::LockingList;
    using SharedMutex    = typename Base::SharedMutex;

private:
    struct IdentityCopy
    {
        template <typename F>
        static void construct(ElementList& list, const Key& key, F&& f)
        {
            std::forward<F>(f)();
            list.emplace_front(key);
        }
    };

    struct IdentityMove
    {
        template <typename F>
        static void construct(ElementList& list, Key&& key, F&& f)
        {
            // TODO: move the function call to after the emplace and pass the new element as argument?
            std::forward<F>(f)();
            list.emplace_front(std::move(key));
        }
    };

    struct ConstructCopy
    {
        static void construct(ElementList& list, const key_type& key)
        {
            list.emplace_front(key);
        }
    };

    struct ConstructMove
    {
        static void construct(ElementList& list, key_type&& key)
        {
            list.emplace_front(std::move(key));
        }
    };

public:
    using Base::Base;
    using Base::histogram;
    using Base::rehash;

    iterator begin()
    {
        // Read lock. Find first iterator
        // return iterator{*this, }
        return iterator{*this};
    }

    iterator end()
    {
        return iterator{*this};
    }

    iterator find(const Key& key)
    {
        auto result = Base::find_impl(key);

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return iterator{*this, result.first};
        } else {
            return end();
        }
    }

    const_iterator find(const Key& key) const
    {
        auto result = Base::find_impl(key);

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return const_iterator{*this, result.first};
        } else {
            return end();
        }
    }

    std::pair<iterator, bool> insert(const value_type& value)
    {
        auto result = Base::template find_or_create_impl<ConstructCopy>(value.first);

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{*this, result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }

    std::pair<iterator, bool> insert(value_type&& value)
    {
        auto result = Base::template find_or_create_impl<ConstructMove>(std::move(value));

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{*this, result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }

    template <typename F>
    std::pair<iterator, bool> insert_and_run(const Key& key, F&& f)
    {
        auto result = Base::template find_or_create_impl<IdentityCopy>(key, std::forward<F>(f));

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{*this, result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }

    // TODO: make sure find_or_create_impl takes Key rvalue references
    template <typename F>
    std::pair<iterator, bool> insert_and_run(Key&& key, F&& f)
    {
        auto result = Base::template find_or_create_impl<IdentityMove>(std::move(key), std::forward<F>(f));

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{*this, result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }
};

template <typename Key, typename T, typename Hash, typename KeyEqual>
struct ConcurrentUnorderedMapTraits
{
    using key_type     = Key;
    using primary_type = T;
    using hasher       = Hash;
    using key_equal    = KeyEqual;
    using value_type   = std::pair<const Key, T>;
};

template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class ConcurrentUnorderedMap : private ConcurrentHashTable<ConcurrentUnorderedMapTraits<Key, T, Hash, KeyEqual>>
{
public:
    using Traits         = ConcurrentUnorderedMapTraits<Key, T, Hash, KeyEqual>;
    using Base           = ConcurrentHashTable<Traits>;
    using key_type       = typename Traits::key_type;
    using value_type     = typename Traits::value_type;
    using primary_type   = typename Traits::primary_type;
    using hasher         = typename Traits::hasher;
    using key_equal      = typename Traits::key_equal;
    using iterator       = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;
    using ElementList    = typename Base::ElementList;
    using LockingList    = typename Base::LockingList;
    using SharedMutex    = typename Base::SharedMutex;

private:
    struct ConstructGenerator
    {
        template <typename F>
        static void construct(ElementList& list, const Key& key, F&& creator)
        {
            list.emplace_front(key, std::forward<F>(creator)(key));
        }
    };

    struct ConstructCopy
    {
        static void construct(ElementList& list, const key_type& key, const primary_type& model)
        {
            list.emplace_front(key, model);
        }
    };

    struct ConstructMove
    {
        static void construct(ElementList& list, const key_type& key, primary_type&& model)
        {
            list.emplace_front(key, std::move(model));
        }
    };

    struct ConstructDefault
    {
        static void construct(ElementList& list, const primary_type& key)
        {
            list.emplace_front(key, primary_type{});
        }
    };

public:
    using Base::Base;
    using Base::histogram;
    using Base::rehash;

    iterator begin()
    {
        // Read lock. Find first iterator
        // return iterator{*this, }
        return iterator{*this};
    }

    iterator end()
    {
        return iterator{*this};
    }

    const T& at(const Key& key) const
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(Base::m_bucket_mutex);

        const auto bucket_count = Base::m_buckets.size();
        const auto bucket_idx   = get_bucket_index(hasher{}(key), bucket_count);

        const LockingList& locking_list = Base::m_buckets[bucket_idx];
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

    bool update(const Key& key, primary_type value)
    {
        return update_impl(key, std::move(value));
    }

    bool update(const Key& key, primary_type&& value)
    {
        return update_impl(key, std::move(value));
    }

    // These return non-const references for maximum flexibility even though it's up to the user to make sure they are
    // accessed in a thead-safe manner. The map does nothing to prevent race conditions in modifying the returned
    // references. I suggest you copy them or store them in const values.
    template <typename F>
    std::pair<iterator, bool> generate(const Key& key, F&& creator)
    {
        auto result = Base::template find_or_create_impl<ConstructGenerator>(key, std::forward<F>(creator));

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{*this, result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }

    // These return non-const references for maximum flexibility even though it's up to the user to make sure they are
    // accessed in a thead-safe manner. The map does nothing to prevent race conditions in modifying the returned
    // references. I suggest you copy them or store them in const values.
    T& operator[](const Key& key)
    {
        auto result = Base::template find_or_create_impl<ConstructDefault>(key);
        return *result.first;
    }

    T& operator[](Key&& key)
    {
        auto result = Base::template find_or_create_impl<ConstructDefault>(std::move(key));
        return *result.first;
    }

    // These return non-const references for maximum flexibility even though it's up to the user to make sure they are
    // accessed in a thead-safe manner. The map does nothing to prevent race conditions in modifying the returned
    // references. I suggest you copy them or store them in const values.
    std::pair<iterator, bool> insert(const value_type& value)
    {
        auto result = Base::template find_or_create_impl<ConstructCopy>(value.first, value.second);

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{*this, result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }

    // These return non-const references for maximum flexibility even though it's up to the user to make sure they are
    // accessed in a thead-safe manner. The map does nothing to prevent race conditions in modifying the returned
    // references. I suggest you copy them or store them in const values.
    std::pair<iterator, bool> insert(value_type&& value)
    {
        auto result = Base::template find_or_create_impl<ConstructMove>(value.first, std::move(value.second));

        // Another thread may alter our lists between the base call and here (e.g., rehash), and we cannot rely on a
        // generic end iterator from the forward_list. Recreate the end iterator explicitly.
        if (result.second) {
            return std::make_pair(iterator{static_cast<Base&>(*this), result.first}, result.second);
        } else {
            return std::make_pair(end(), result.second);
        }
    }

private:
    bool update_impl(const Key& key, primary_type&& value)
    {
        // Read lock on bucket list
        std::shared_lock<SharedMutex> bucket_lock(Base::m_bucket_mutex);

        const auto num_buckets = Base::m_buckets.size();
        const auto bucket_idx  = get_bucket_index(hasher{}(key), num_buckets);

        LockingList& locking_list = Base::m_buckets[bucket_idx];
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
};
