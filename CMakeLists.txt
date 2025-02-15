cmake_minimum_required(VERSION 3.20)

function(submodule_error module)
  message(FATAL_ERROR "\
  The ${module} submodule is not present.\n \
  If in git repository run the following two commands:\n \
  git submodule init\n \
  git submodule update")
endfunction()

if(DEFINED PROTO_LOADED)
  return() # Proto has already been loaded by encompassing CMake project
endif()
# checks if Proto is submodule of larger CMake project but hasn't been loaded yet
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set(PROTO_LOADED "${Proto_VERSION_MAJOR}.${Proto_VERSION_MINOR}" PARENT_SCOPE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # if OFF, compilation will revert to older standard if C++11 unavailable
set(CMAKE_CXX_EXTENSIONS OFF)  # if OFF, compilation flag uses c++11, not gnu++11

# setup basics
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose build type, \
options are: Debug Release RelWithDebInfo" FORCE)
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
set(CMAKE_POSITION_INDEPENEDENT_CODE TRUE)
option(AMR "Build AMR libraries" OFF)
option(MMB "Build MMB libraries" OFF)
option(OPS "Build Operator libraries" OFF)
option(ENABLE_MPI "Build using MPI" ON)
option(ENABLE_OPENMP "Build using OpenMP" OFF)
option(ENABLE_CUDA "Build using CUDA" OFF)
option(ENABLE_HIP "Build using HIP" OFF)
option(ENABLE_HDF5 "Build using HDF5" ON)
option(ENABLE_TESTS "Build default tests only" ON)
option(ENABLE_ALL_TESTS "Build all tests" OFF)
option(ENABLE_EXAMPLES "Build examples" ON)
option(ENABLE_DOCS "Build Proto documentation" OFF)
option(MEMCHECK "turns on code in BoxData that checks that copying/aliasing is working correctly" OFF)
option(MEMTRACK "print the amount of data allocated per protoMalloc" OFF)
option(TIMERS "whether to turn on timers" ON)

set(DIM 2 CACHE STRING "Dimensionality of examples")
set(STACK 4294967296 CACHE STRING "Stack size")
message(STATUS "Dimensionality: ${DIM}\n   Stack size: ${STACK}")

add_compile_definitions(DIM=${DIM})
add_compile_definitions(PR_STACK_ALLOC=${STACK})

if (AMR)
    message(STATUS "AMR libraries are enabled")
    add_compile_definitions(PR_AMR)
endif()

if (MMB)
    message(STATUS "MMB libraries are enabled")
    add_compile_definitions(PR_MMB)
endif()

if (OPS)
    message(STATUS "Operator libraries are enabled")
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/blis/include)
        execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/configure_blis)
    endif()
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/blis/include)
        submodule_error("BLIS")
    else()
        file(GLOB CONFIG RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/blis/include/ blis/include/*)
        include_directories(./blis/include/${CONFIG})
        link_directories(./blis/lib/${CONFIG})
        link_libraries(blis)
    endif()
    add_compile_definitions(PR_OPS)
    link_libraries(lapack)
endif()

if(MEMCHECK)
    message(STATUS "MEMCHECK is enabled")
    add_compile_definitions(PROTO_MEM_CHECK)
endif()

if(MEMTRACK)
    message(STATUS "MEMTRACK is enabled")
    add_compile_definitions(memoryFootPrint)
endif()

if(NOT TIMERS)
    message(STATUS "Proto Timers are disabled")
    add_compile_definitions(PR_TURN_OFF_TIMERS)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")  # disables compiler warings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" CACHE STRING "compiler flags")
endif()

project(Proto VERSION 4.0 DESCRIPTION "Proto examples") # LANGUAGES CXX) # HDF5 fails with LANGUAGES argument
enable_language(CXX)

# setup packages
if(ENABLE_CUDA OR ENABLE_HIP)
  add_compile_definitions(PROTO_CUDA)
  if(ENABLE_CUDA AND ENABLE_HIP)
    message(FATAL_ERROR "Must specify at most one of CUDA and HIP")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    add_compile_definitions(THRUST_DEBUG)
  endif()
  if(ENABLE_HIP)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to find HIP")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to find HIP")
    endif()
    if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
      set(CMAKE_PREFIX_PATH "${HIP_PATH} ${HIP_PATH}/.." CACHE PATH "Path to find Rocm")
    else()
      set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} CACHE PATH "Path to find Rocm")
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    list(APPEND LIB_DEP hip)
    add_compile_definitions(PROTO_HIP)
  else()
    if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
      set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
    else()
      set(ENV{CUDACXX} nvcc)
    endif()
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "Setting for BLT")
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "Compute architecture needed by NVCC")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10)
      message(FATAL_ERROR "CUDA version >=10 needed, found ${CMAKE_CUDA_COMPILER_VERSION}")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL 11)
      if(CMAKE_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 9)
        message(FATAL_ERROR "GCC version 9 needed, found ${CMAKE_CXX_COMPILER_VERSION}")
      endif()
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
    list(APPEND LIB_DEP cuda)
  endif()
endif()

if(ENABLE_MPI)
  add_compile_definitions(PR_MPI)
  list(APPEND LIB_DEP mpi)
  set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC TRUE)  # needed by BLT to make mpiexec commands
  if(DEFINED CMAKE_APPLE_SILICON_PROCESSOR)
    get_filename_component(MPI_LINK /usr/local/bin/mpiexec REALPATH)
    get_filename_component(MPI_PATH ${MPI_LINK} DIRECTORY)
    set(MPIEXEC_EXECUTABLE ${MPI_PATH}/mpiexec CACHE FILEPATH "full path for mpiexec")
   endif()
endif()


if(ENABLE_OPENMP)
    list(APPEND LIB_DEP openmp)
endif()

if(ENABLE_HDF5)
    add_compile_definitions(PR_HDF5)
    if(ENABLE_MPI)
        set(HDF5_PREFER_PARALLEL TRUE)
    endif()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      get_filename_component(HDF5_LINK /usr/local/include/hdf5.h REALPATH)
      string(REGEX MATCH [0-9]+\\.[0-9]+\\.[0-9]+ HDF5_VERSION ${HDF5_LINK})
      if(NOT DEFINED ENV{HDF5_ROOT})
        if(ENABLE_MPI)
          set(HDF5_ROOT /usr/local/Cellar/hdf5-mpi/${HDF5_VERSION}/ CACHE PATH "parallel HDF5 directory")
        elseif(DEFINED CMAKE_APPLE_SILICON_PROCESSOR)
          set(HDF5_ROOT /usr/local/hdf5/${HDF5_VERSION} CACHE PATH "serial HDF5 directory")
        else()
          set(HDF5_ROOT /usr/local/Cellar/hdf5/${HDF5_VERSION}/ CACHE PATH "serial HDF5 directory")
        endif()
        if(NOT IS_DIRECTORY HDF5_ROOT AND NOT IS_DIRECTORY ${HDF5_ROOT})
          message(FATAL_ERROR "HDF5 is installed but not in /usr/local/Cellar or stored in $HDF5_ROOT")
        endif()
      endif()
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      if(NOT DEFINED ENV{HDF5_ROOT} AND NOT DEFINED ENV{OLCF_HDF5_ROOT} AND NOT DEFINED ENV{CRAY_HDF5_DIR})
        if(ENABLE_MPI)
          set(HDF5_INC /usr/include/hdf5/openmpi FILEPATH)
          set(HDF5_LIB /usr/lib/x86_64-linux-gnu/hdf5/openmpi FILEPATH)
        else()
          set(HDF5_INC /usr/include/hdf5/serial FILEPATH)
          set(HDF5_LIB /usr/lib/x86_64-linux-gnu/hdf5/serial FILEPATH)
        endif()
        include_directories(${HDF5_INC}/include)
        link_directories(${HDF5_LIB}/lib)
        link_libraries(hdf5)
        set(ENABLE_HDF5 OFF)
      endif()
    endif()
    if(ENABLE_HDF5)  # only turned off on Linux laptops with no hdf5 module
      find_package(HDF5 REQUIRED)
    endif()
endif()

if (DEFINED ENV{OLCF_NETLIB_LAPACK_ROOT})
  set(LAPACK_DIR $ENV{OLCF_NETLIB_LAPACK_ROOT})
elseif(DEFINED ENV{MKLROOT})
  set(LAPACK_DIR $ENV{MKLROOT})
  add_compile_definitions(MKL)
elseif(DEFINED ENV{LAPACK_DIR})  # must do on MacOS after `brew install lapack` to get lapacke.h
  set(LAPACK_DIR $ENV{LAPACK_DIR})
endif()
find_package(LAPACK REQUIRED)

if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        submodule_error("BLT")
    endif ()
  endif ()

# Default to C++11 if not set so GTest/GMock can build
  if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
  endif()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
  if (${_variableName} MATCHES ".*_EXECUTABLE")
    mark_as_advanced(${_variableName})
  endif()
endforeach()

add_custom_target(dataclean COMMAND cd "${CMAKE_BINARY_DIR}/bin" && rm -rf *.hdf5 *.vtk *.table pout*)

# setup Proto config
add_subdirectory(include)

if(ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if(ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif()
