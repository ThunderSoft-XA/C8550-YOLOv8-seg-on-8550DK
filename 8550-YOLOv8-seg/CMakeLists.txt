cmake_minimum_required(VERSION 3.10)
project(YOLOv8s)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)

include_directories(
    ./
    /usr/include/SNPE
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    test
    ./main.cpp
    ./SNPETask.cpp
    ./YOLOv8s.cpp
)

target_link_libraries(
    test
    pthread
    dl
    ${OpenCV_LIBS}
    /usr/lib/libSNPE.so
)
