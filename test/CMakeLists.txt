function(build_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} gtest_main ginkgo ginkgo_cuda ginkgo_reference ginkgo_hip ginkgo_dpcpp ginkgo_omp ginkgo_device)
    add_test(NAME ${test_name}_test COMMAND ${test_name})
endfunction()

build_test(block_matrix)
build_test(cg)
