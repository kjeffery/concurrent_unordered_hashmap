#include "ConcurrentUnorderedMap.h"

#include <iostream>

struct Verbose
{
    Verbose()
    {
        std::cerr << __FUNCSIG__ << '\n';
    }

    explicit Verbose(int x)
    : m_id(x)
    {
        std::cerr << __FUNCSIG__ << '\n';
    }

    Verbose(const Verbose& other)
    : m_id(other.m_id)
    {
        std::cerr << __FUNCSIG__ << '\n';
    }

    Verbose(Verbose&& other)
    : m_id(other.m_id)
    {
        std::cerr << __FUNCSIG__ << '\n';
    }

    Verbose& operator=(const Verbose& other)
    {
        std::cerr << __FUNCSIG__ << '\n';
        m_id = other.m_id;
        return *this;
    }

    Verbose& operator=(Verbose&& other)
    {
        std::cerr << __FUNCSIG__ << '\n';
        m_id = other.m_id;
        return *this;
    }

    ~Verbose()
    {
        std::cerr << __FUNCSIG__ << '\n';
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
        cache.find_or_generate(i, [](int x) { return x * x; });
    }
}

void test_rehash()
{
    ConcurrentUnorderedMap<int, Verbose> cache(1);

    auto f = [](auto x) { return Verbose{x}; };

    const auto& a = cache.find_or_generate(1, f);
    a.report();
    const auto& b = cache.find_or_generate(2, f);
    b.report();
    const auto& c = cache.find_or_create(3);
    c.report();
    const auto& d = cache.find_or_create(3, Verbose{42});
    d.report();

    Verbose v{101};
    const auto& e = cache.find_or_create(3, v);
    e.report();

    cache.histogram();

    cache.rehash(100);
    a.report();
    b.report();
    c.report();

    cache.histogram();
    std::cout << "End of function\n";
}

int main()
{
    //test_rehash();
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

#if 1
    ConcurrentUnorderedMap<int, int> cache;
    constexpr int num_threads = 32;
    std::vector<std::thread> threads;
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
