cmake_minimum_required(VERSION 3.15)
project(hnswlib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hnswlib
    "src/hnswlib.cpp"
)

target_include_directories(hnswlib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE
)

add_executable(playground
    playground/main.cpp
)

target_link_libraries(playground PRIVATE hnswlib)
