cmake_minimum_required(VERSION 3.10)

PROJECT(testCUDA)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE DEBUG)
#暂且设置为 release 模式
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pthread -fopenmp")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

set(OpenCV_DIR /home/liushiqi/opencv/build)
set(OpenCV_INCLUDE_DIRS /home/liushiqi/opencv/include)
set(OpenCV_LIBRARIES /home/liushiqi/opencv/build/build/lib)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        cv_bridge
        camera_model
        )

find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/cudaSift_include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/myBundleAdjustment/include)

set(cuda_sources
        ${PROJECT_SOURCE_DIR}/cudaSift_src/geomFuncs.cpp
        ${PROJECT_SOURCE_DIR}/cudaSift_src/cudaImage.cu
        ${PROJECT_SOURCE_DIR}/cudaSift_src/cudaSiftH.cu
        ${PROJECT_SOURCE_DIR}/cudaSift_src/matching.cu
        )

list(APPEND CUDA_NVCC_FLAGS
        -lineinfo;
        --compiler-options;
        -O2;
        -D_FORCE_INLINES;
        -DVERBOSE_NOT;
        -arch=compute_61;
        -code=sm_61
        )

#添加自制BA库
add_subdirectory(myBundleAdjustment)
#
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
CUDA_ADD_LIBRARY(cusift SHARED ${cuda_sources})

ADD_LIBRARY(VO src/featureTracking.cpp
            src/frame.cpp
            src/feature.cpp
            src/visualization.cpp)

ADD_EXECUTABLE(test2
        test2.cpp
        )
ADD_EXECUTABLE(VOtest VOTest.cpp)
ADD_EXECUTABLE(VOtest2 VOTest2.cpp)
ADD_EXECUTABLE(BAtest BAtest.cpp)
ADD_EXECUTABLE(mainSift mainSift.cpp)

CUDA_ADD_EXECUTABLE(opencvtest opencvCUDATest.cpp)
target_link_libraries(opencvtest ${OpenCV_LIBS} ${CUDA_LIBS})


target_link_libraries(test2 cusift ${OpenCV_LIBS} ${CUDA_LIBS} )
target_link_libraries(VOtest VO BA cusift ${OpenCV_LIBS} ${CUDA_LIBS} ${catkin_LIBRARIES} ${Sophus_LIBRARIES})
target_link_libraries(VOtest2 VO BA cusift ${OpenCV_LIBS} ${CUDA_LIBS} ${catkin_LIBRARIES} ${Sophus_LIBRARIES})
target_link_libraries(BAtest VO BA cusift ${OpenCV_LIBS} ${CUDA_LIBS} ${catkin_LIBRARIES} ${Sophus_LIBRARIES})
target_link_libraries(mainSift ${OpenCV_LIBS} cusift )