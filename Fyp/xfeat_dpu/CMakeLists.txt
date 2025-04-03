cmake_minimum_required(VERSION 3.10)
project(xfeat_dpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(OpenCV REQUIRED)
find_package(vitis_ai_library REQUIRED)
find_package(Glog REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

add_executable(xfeat_dpu
    src/XFextractorDPU.cpp
    src/demo_xfeat.cpp
)


target_link_libraries(xfeat_dpu
    ${OpenCV_LIBS}
    vitis_ai_library::dpu_task
    glog::glog
)

add_executable(throughput_analysis
    src/XFextractorDPU.cpp
    src/throughput_analysis.cpp
)
target_link_libraries(throughput_analysis
    ${OpenCV_LIBS}
    vitis_ai_library::dpu_task
)


install(TARGETS xfeat_dpu DESTINATION bin)