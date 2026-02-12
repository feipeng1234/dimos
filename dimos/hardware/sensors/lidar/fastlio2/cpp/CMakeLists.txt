cmake_minimum_required(VERSION 3.14)
project(fastlio2_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# FAST-LIO-NON-ROS source directory (local reference)
set(FASTLIO_DIR "$ENV{HOME}/coding/FAST-LIO-NON-ROS")

# OpenMP for parallel processing
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# MP defines (same logic as FAST-LIO)
message("CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  if(N GREATER 4)
    add_definitions(-DMP_EN -DMP_PROC_NUM=3)
  elseif(N GREATER 3)
    add_definitions(-DMP_EN -DMP_PROC_NUM=2)
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Fetch dimos-lcm for C++ message headers
include(FetchContent)
FetchContent_Declare(dimos_lcm
  GIT_REPOSITORY https://github.com/dimensionalOS/dimos-lcm.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dimos_lcm)

# LCM
find_package(PkgConfig REQUIRED)
pkg_check_modules(LCM REQUIRED lcm)

# Eigen3
find_package(Eigen3 REQUIRED)

# PCL
find_package(PCL 1.8 REQUIRED)

# nlohmann_json (FAST-LIO config parsing)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Livox SDK2 (installed to /usr/local)
find_library(LIVOX_SDK livox_lidar_sdk_shared HINTS /usr/local/lib)
if(NOT LIVOX_SDK)
  message(FATAL_ERROR "Livox SDK2 not found. Install from https://github.com/Livox-SDK/Livox-SDK2")
endif()

add_executable(fastlio2_native
  main.cpp
  ${FASTLIO_DIR}/src/preprocess.cpp
  ${FASTLIO_DIR}/include/ikd-Tree/ikd_Tree.cpp
)

target_include_directories(fastlio2_native PRIVATE
  ${FASTLIO_DIR}/include
  ${FASTLIO_DIR}/src
  ${dimos_lcm_SOURCE_DIR}/generated/cpp_lcm_msgs
  ${LCM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/include  # Livox SDK2 headers
)

target_link_libraries(fastlio2_native PRIVATE
  ${LCM_LIBRARIES}
  ${LIVOX_SDK}
  ${PCL_LIBRARIES}
  nlohmann_json::nlohmann_json
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fastlio2_native PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_directories(fastlio2_native PRIVATE
  ${LCM_LIBRARY_DIRS}
)
