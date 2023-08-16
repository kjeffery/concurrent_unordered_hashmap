#include "ConcurrentUnorderedMap.h"

#include <iostream>
#include <thread>

struct Verbose
{
    Verbose()
    {
        // std::cerr << __FUNCSIG__ << '\n';
    }

    explicit Verbose(int x)
    : m_id(x)
    {
        // std::cerr << __FUNCSIG__ << '\n';
    }

    Verbose(const Verbose& other)
    : m_id(other.m_id)
    {
        // std::cerr << __FUNCSIG__ << '\n';
    }

    Verbose(Verbose&& other)
    : m_id(other.m_id)
    {
        // std::cerr << __FUNCSIG__ << '\n';
    }

    Verbose& operator=(const Verbose& other)
    {
        // std::cerr << __FUNCSIG__ << '\n';
        m_id = other.m_id;
        return *this;
    }

    Verbose& operator=(Verbose&& other)
    {
        // std::cerr << __FUNCSIG__ << '\n';
        m_id = other.m_id;
        return *this;
    }

    ~Verbose()
    {
        // std::cerr << __FUNCSIG__ << '\n';
    }

    void report() const
    {
        std::cerr << this << '\t' << m_id << '\n';
    }

    int m_id{0};
};

void do_ints(ConcurrentUnorderedMap<int, int>& cache, int start, int end)
{
    for (int i = start; i < end; ++i) {
        cache.generate(i, [](int x) { return x * x; });
    }
}

void test_rehash()
{
    ConcurrentUnorderedMap<int, Verbose> cache(1);

    auto f = [](auto x) { return Verbose{x}; };

    const auto& a = cache.generate(1, f);
    a.first->second.report();
    const auto& b = cache.generate(2, f);
    b.first->second.report();
    const auto& c = cache.insert(std::make_pair(3, Verbose{8}));
    // c.first->second.report();
    // const auto& d = cache.find_or_create(3, Verbose{42});
    // d.first->second.report();

    Verbose v{101};
    // const auto& e = cache.find_or_create(3, v);
    // e.first->second.report();

    // cache.histogram();

    cache.rehash(100);
    a.first->second.report();
    b.first->second.report();
    // c.first->second.report();

    // cache.histogram();
    std::cout << "End of function\n";
}

void test_set()
{
    ConcurrentUnorderedSet<int> set;

    auto i0 = set.insert(3);
    auto i1 = set.insert(4);
    auto i2 = set.insert(5);
    auto i3 = set.insert(6);

    assert(i0.second == true);
    assert(i1.second == true);
    assert(i2.second == true);
    assert(i3.second == true);

    auto i4 = set.insert(4);
    assert(i4.second == false);

    int  count = 0;
    auto f     = [&count]() { ++count; };
    auto i5    = set.insert_and_run(7, f);
    auto i6    = set.insert_and_run(8, f);
    auto i7    = set.insert_and_run(7, f);

    auto i8 = set.find(8);
    auto i9 = set.find(12);

    assert(i8 != set.end());
    assert(i9 == set.end());
}

template <typename T>
struct MyContainer
{
    void f(T&& t)
    {
    }

    template <typename U>
    void g(U&& u)
    {
    }
};

int main()
{
    test_rehash();
#if 0
    ConcurrentUnorderedMap<int, int> cache;

    auto f = [](auto x) { return x * x; };

    for (auto x : {3, 4, 5, 3, 4, 5, 3, 4, 5}) {
        const auto& a = cache.find_or_create(x, f);
        std::cout << cache.load_factor() << '\n';
        std::cout << a << std::endl;
        std::cout << std::addressof(a) << std::endl;
    }

    const auto& a = cache.find_or_create(3, f);

    std::cout << a << std::endl;
    std::cout << std::addressof(a) << std::endl;

    cache.rehash(100);
    std::cout << a << std::endl;
    std::cout << std::addressof(a) << std::endl;

    cache.rehash(1000);
    std::cout << a << std::endl;
    std::cout << std::addressof(a) << std::endl;

    cache.rehash(10000);
    std::cout << a << std::endl;
    std::cout << std::addressof(a) << std::endl;
#endif

    test_set();

#if 1
    ConcurrentUnorderedMap<int, int> cache;
    constexpr int                    num_threads = 32;
    std::vector<std::thread>         threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&do_ints, std::ref(cache), i * 100, i * 100 + 2000);
    }

    for (auto& t : threads) {
        t.join();
    }

    cache.histogram();
#endif
}
