# Created on 2019/12/20 by Kihiro

cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Bug)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify build type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# global include
include_directories("${PROJECT_SOURCE_DIR}")

# run path for installed executables
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# paths to external libraries
set(METIS_ROOT "" CACHE PATH "METIS root")
set(HDF5_ROOT "" CACHE PATH "HDF5 root")
set(LEGION_ROOT "" CACHE PATH "Legion root")

include_directories(
        "${METIS_ROOT}/include"
        "${HDF5_ROOT}/include"
        "${LEGION_ROOT}/include")
link_directories(
        "${METIS_ROOT}/lib"
        "${HDF5_ROOT}/lib"
        "${LEGION_ROOT}/lib")

# specify default cmake options
option(USE_DOUBLES "Use double precision" ON)

# set preprocessor definitions
if(USE_DOUBLES)
    add_compile_definitions(USE_DOUBLES)
endif()

add_executable(exec
        mesh.cpp mesh_data.cpp solution_data.cpp main.cpp)
target_link_libraries(exec PRIVATE
        metis hdf5 hdf5_cpp
        legion realm
        pthread z dl rt)

install(TARGETS exec RUNTIME DESTINATION ${PROJECT_SOURCE_DIR})