cmake_minimum_required(VERSION 3.20)

project(lab4_path_tracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(embree_DIR "D:/ITMO.kal/vcpkg/installed/x64-windows/share/embree")

add_executable(path_tracer
    src/app.cpp
    src/core/path_tracer.cpp
    src/core/sampling.cpp
    src/accel/embree_scene.cpp
    src/io/obj_loader.cpp
    src/io/ppm_writer.cpp
    src/main.cpp
)

target_include_directories(path_tracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(path_tracer PRIVATE /W4 /permissive-)
    target_compile_definitions(path_tracer PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(path_tracer PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(embree CONFIG REQUIRED)
target_link_libraries(path_tracer PRIVATE embree)

option(PATH_TRACER_BUILD_TESTS "Build path tracer tests" ON)

if (PATH_TRACER_BUILD_TESTS)
    add_executable(path_tracer_tests
        src/accel/embree_scene.cpp
        src/core/sampling.cpp
        src/io/obj_loader.cpp
        tests/test_main.cpp
    )
    target_include_directories(path_tracer_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(path_tracer_tests PRIVATE embree)
    if (MSVC)
        target_compile_options(path_tracer_tests PRIVATE /W4 /permissive-)
        target_compile_definitions(path_tracer_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(path_tracer_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
