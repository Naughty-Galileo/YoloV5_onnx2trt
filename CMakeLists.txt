cmake_minimum_required(VERSION 3.5)
project(demo)

set(CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/cuda-11.3/targets/x86_64-linux/include/)

#查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)


# 编译detector.so
list(APPEND lib_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/SampleDetector.cpp)

add_library (detector SHARED ${lib_SOURCES})
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        glog
        nvinfer
        nvonnxparser
        -L/usr/local/cuda/lib64/ -lcudart 
)

add_executable(demo main.cpp)
target_link_libraries(demo
        ${OpenCV_LIBS}
        detector
        glog
)
