#include "ThreadSafeCache.h"

#include <iostream>

void do_ints(ThreadSafeCache<int, int>& cache, int start, int end)
{
    for (int i = start; i < end; ++i) {
        cache.find_or_create(i, [](int x) { return x * x; });
    }
}

int main()
{
    ThreadSafeCache<int, int> cache;

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

#if 0
    constexpr int num_threads = 1;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&do_ints, std::ref(cache), i * 100, i * 100 + 2000);
    }

    for (auto& t : threads) {
        t.join();
    }
#endif


}
