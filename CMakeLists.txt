cmake_minimum_required(VERSION 2.8)

project(pcl_conversion_depth_image_to_surface_normal LANGUAGES CXX C)

# Set build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

add_compile_options(-Wall -Wextra)
add_compile_options(
    "$<$<CONFIG:Debug>:-Og>"
    "$<$<CONFIG:Release>:-O3>"
    "$<$<CONFIG:None>:-O3>")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find Packages
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

set(USE_BACKWARD_CPP OFF CACHE BOOL "Enable backward-cpp")

if(USE_BACKWARD_CPP)
    find_package(Backward REQUIRED)
    message(STATUS "backward-cpp: ENABLED")
else()
    message(STATUS "backward-cpp: DISABLED")
endif()

# Setup for popl
set(popl_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/popl/include)

# Build an example executable
add_executable(pcl_conversion_depth_image_to_surface_normal src/main.cpp)
set(EXECUTABLE_TARGET pcl_conversion_depth_image_to_surface_normal)

target_include_directories(${EXECUTABLE_TARGET}
    PRIVATE
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/popl/include>
)
target_link_libraries(${EXECUTABLE_TARGET}
    PRIVATE
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    yaml-cpp
)

if(USE_BACKWARD_CPP)
    target_link_libraries(${EXECUTABLE_TARGET} Backward::Backward)
endif()