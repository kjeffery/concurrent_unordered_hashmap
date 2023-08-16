cmake_minimum_required(VERSION 3.16)
#cmake_minimum_required(VERSION 3.26)
project(ThreadSafeCache)

set(CMAKE_CXX_STANDARD 14)

add_executable(ThreadSafeCache main.cpp
        ConcurrentUnorderedMap.h)

if (UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(ThreadSafeCache PRIVATE Threads::Threads atomic)
endif (UNIX)
