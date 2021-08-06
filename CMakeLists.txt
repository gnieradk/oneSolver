cmake_minimum_required(VERSION 3.9)

project("one-solver" CXX)
# Option is default OFF in cmake
# Call cmake with option enabled -DWITH_SYCL=ON/OFF ..
option(WITH_SYCL "Build with the SYCL enabled")
if (WITH_SYCL)
	add_definitions(-DWITH_SYCL)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Intel MPI compilers
# https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/command-reference/compiler-commands.html

if(NOT WITH_SYCL)
	set(CMAKE_C_COMPILER mpiicc)
	set(CMAKE_CXX_COMPILER mpiicpc)
	message(STATUS "SYCL is NOT enabled in compilation")
else()
	set(ENV{I_MPI_CXX} "dpcpp")
	message(STATUS "the value of I_MPI_CXX: " $ENV{I_MPI_CXX})
	set(CMAKE_C_COMPILER mpicc)
	set(CMAKE_CXX_COMPILER mpiicpc)
	set(CMAKE_CXX_FLAGS "-fsycl")
	message(STATUS "SYCL is enabled in compilation")
endif()

# set(CMAKE_CXX_FLAGS "-fsycl")

# find_package(Boost 1.65)

add_subdirectory(src)
# add_subdirectory(tests)