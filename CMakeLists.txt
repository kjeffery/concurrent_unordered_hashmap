cmake_minimum_required(VERSION 3.16)
#cmake_minimum_required(VERSION 3.26)
project(ThreadSafeCache)

set(CMAKE_CXX_STANDARD 14)

add_executable(ThreadSafeCache main.cpp
        ConcurrentUnorderedMap.h)
