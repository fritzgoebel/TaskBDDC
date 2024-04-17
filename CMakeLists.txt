cmake_minimum_required(VERSION 3.23)
project(TaskCG)

link_directories($ENV{HOME}/install/lib $ENV{HOME}/install/lib64 )

find_package(MPI REQUIRED)
find_package(Ginkgo REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(FetchContent)
FetchContent_Declare(
      googletest
        # Specify the commit you depend on and update it regularly.
          URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
          
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${MPI_INCLUDE_PATH} $ENV{HOME}/install/include)

add_executable(task_bddc main.cpp)

target_include_directories(task_bddc PRIVATE ${TaskCG_BINARY_DIR})
target_link_libraries(task_bddc PRIVATE ginkgo ginkgo_cuda ginkgo_reference ginkgo_hip ginkgo_dpcpp ginkgo_omp ginkgo_device)
#target_link_libraries(task_bddc PRIVATE ginkgod ginkgo_cudad ginkgo_referenced ginkgo_hipd ginkgo_dpcppd ginkgo_ompd ginkgo_deviced)

add_subdirectory(matrices)

add_subdirectory(test)
include(CTest)
