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
    src/io/bilateral_filter.cpp
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

get_filename_component(_embree_share_dir "${embree_DIR}" DIRECTORY)
get_filename_component(_embree_installed_dir "${_embree_share_dir}" DIRECTORY)
set(_vcpkg_bin_dir "${_embree_installed_dir}/bin")
file(GLOB _embree_runtime_dlls
    "${_vcpkg_bin_dir}/embree*.dll"
    "${_vcpkg_bin_dir}/tbb*.dll"
)
if (_embree_runtime_dlls)
    add_custom_command(TARGET path_tracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_embree_runtime_dlls}
            $<TARGET_FILE_DIR:path_tracer>
    )
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(path_tracer PRIVATE OpenMP::OpenMP_CXX)
endif()
