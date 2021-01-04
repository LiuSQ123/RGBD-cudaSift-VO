cmake_minimum_required(VERSION 3.10)
project(myBundleAdjustment)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE DEBUG)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pthread -fopenmp")

#ADD_DEFINITIONS(-DDEBUG)
#ADD_DEFINITIONS(-DSHOWMATRIX)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS})
message("$PROJECT_SOURCE_DIR")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY /home/liushiqi/ClionProjects/CUDA_project1/lib)
add_library(BA src/costFunction.cpp src/problem.cpp include/lossFunction.hpp)
add_executable(myBundleAdjustment test_sample/main.cpp)
target_link_libraries(myBundleAdjustment BA ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES})