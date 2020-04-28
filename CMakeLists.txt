cmake_minimum_required(VERSION 3.10)
project(main)

find_package(OpenCV 4.0.0 REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin/)

add_compile_options(-std=c++11)
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(./src SOURCES)

add_executable(main  ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})