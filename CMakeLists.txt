cmake_minimum_required(VERSION 3.20)

project(QT_Qwen600 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "========================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Qt5 Widgets: ${Qt5Widgets_VERSION}")

# CUDA inference static library
add_library(qwen_inference STATIC
    inference_engine_impl.cu
)

target_compile_options(qwen_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(MSVC)
    target_compile_options(qwen_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W0>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -w>
    )
else()
    target_compile_options(qwen_inference PRIVATE -O3 -w)
endif()

target_link_libraries(qwen_inference PRIVATE CUDA::cublas)

# Qt GUI executable
add_executable(QT_Qwen600
    main.cpp
    mainwindow.cpp
    mainwindow.h
)

target_link_libraries(QT_Qwen600 PRIVATE
    qwen_inference
    CUDA::cublas
    CUDA::cudart
    Qt5::Widgets
)

if(MSVC)
    target_compile_options(QT_Qwen600 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W0>
    )
else()
    target_compile_options(QT_Qwen600 PRIVATE -O3 -w)
endif()
