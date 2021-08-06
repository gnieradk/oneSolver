#find_package(Boost 1.65 COMPONENTS program_options REQUIRED)

set(MODULE_DIRECTORY "${PROJECT_SOURCE_DIR}/src")
set(SOURCE_DIRECTORY "${MODULE_DIRECTORY}")


if(NOT WITH_SYCL)
	file(GLOB SOURCES ${SOURCE_DIRECTORY}/*mpi*.cpp)
	message(STATUS "File - no SYCL")
else()
	file(GLOB SOURCES ${SOURCE_DIRECTORY}/*mpi-sycl*.cpp)
	message(STATUS "File - with SYCL")
endif()

#file(GLOB SOURCES  "${SOURCE_DIRECTORY}/*.cpp")
foreach(sourcefile ${SOURCES})
    get_filename_component(filename ${sourcefile} NAME_WE) 
    add_executable(${filename} ${sourcefile})
    target_include_directories(${filename} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(${filename} ${LIBRARY_NAME})
#    target_link_libraries(${filename} Boost::program_options)
endforeach(sourcefile ${SOURCES})
