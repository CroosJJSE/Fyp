cmake_minimum_required(VERSION 3.10)
project(SuperPointProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set all required package versions
set(OpenCV_VERSION_REQUIRED 4.5.4)
set(VART_VERSION_REQUIRED 3.5.0)
set(UNILOG_VERSION_REQUIRED 3.5.0)
set(XIR_VERSION_REQUIRED 3.5.0)
set(GLOG_VERSION_REQUIRED 0.5.0)

# Find packages with version checking
find_package(OpenCV ${OpenCV_VERSION_REQUIRED} REQUIRED)
find_package(VART ${VART_VERSION_REQUIRED} REQUIRED)
find_package(UNILOG ${UNILOG_VERSION_REQUIRED} REQUIRED)
find_package(XIR ${XIR_VERSION_REQUIRED} REQUIRED)
find_package(Glog ${GLOG_VERSION_REQUIRED} REQUIRED)
find_package(Threads REQUIRED)
find_package(vitis_ai_library REQUIRED)

# Print versions for verification
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "VART version: ${VART_VERSION}")
message(STATUS "UNILOG version: ${UNILOG_VERSION}")
message(STATUS "GLOG version: ${GLOG_VERSION}")
message(STATUS "XIR version: ${XIR_VERSION}")

# Verify found versions meet requirements
if(OpenCV_VERSION VERSION_LESS OpenCV_VERSION_REQUIRED)
    message(FATAL_ERROR "OpenCV version ${OpenCV_VERSION} found, but ${OpenCV_VERSION_REQUIRED} required")
endif()

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

set(MODEL_SOURCES
    include/src/superpoint.cpp 
)

# Executables
add_executable(demo
    ${MODEL_SOURCES}
    include/src/demo_superpoint.cpp
)

add_executable(evaluate_accuracy
    ${MODEL_SOURCES}
    include/src/superpoint_hpatches_eval.cpp
)

# Common link libraries for both executables
foreach(TARGET demo evaluate_accuracy)
    target_link_libraries(${TARGET}
        ${OpenCV_LIBS}
        vart::runner
        vart::util
        xir::xir
        unilog::unilog
        glog::glog
        Threads::Threads
        vitis_ai_library::dpu_task
        vitis_ai_library::math
    )
    # Add additional target-specific properties if needed
endforeach()