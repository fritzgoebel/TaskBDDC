#include "../overlapping_vector.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>
#include <vector>

#include "ginkgo/core/distributed/partition.hpp"
#include "ginkgo/core/matrix/dense.hpp"

overlapping_vector setup_three_part_vector() {
    using part = gko::experimental::distributed::Partition<int, int>;
    std::vector<std::vector<int>> inner_idxs{{0, 1, 2, 3, 4, 5, 6}, {10, 11, 12, 13, 14, 15, 16}, {20, 21, 22, 23, 24, 25, 26}};
    std::vector<std::vector<int>> bndry_idxs{{7, 8, 9, 17, 19, 27, 29}, {7, 9, 17, 18, 19, 28, 29}, {8, 9, 18, 19, 27, 28, 29}};
    auto exec = gko::ReferenceExecutor::create();
    auto partition = gko::share(part::build_from_global_size_uniform(exec, 3, 30));

    auto vec = overlapping_vector(inner_idxs, bndry_idxs, partition, 30);
    return vec;
}

TEST(OverlappingVector, ConstructorSinglePart) {
    using part = gko::experimental::distributed::Partition<int, int>;
    std::vector<std::vector<int>> inner_idxs{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
    std::vector<std::vector<int>> bndry_idxs{{}};
    auto exec = gko::ReferenceExecutor::create();
    auto partition = gko::share(part::build_from_global_size_uniform(exec, 1, 10));

    auto vec = overlapping_vector(inner_idxs, bndry_idxs, partition, 10);
    ASSERT_EQ(vec.get_size()[0], 10);
    ASSERT_EQ(vec.get_size()[1], 1);
    ASSERT_EQ(vec.get_num_parts(), 1);
}

TEST(OverlappingVector, ConstructorMultipleParts) {
    auto vec = setup_three_part_vector();

    ASSERT_EQ(vec.get_size()[0], 30);
    ASSERT_EQ(vec.get_size()[1], 1);
    ASSERT_EQ(vec.get_num_parts(), 3);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(vec.get_inner_size(i), 7);
        ASSERT_EQ(vec.get_bndry_size(i), 7);
        ASSERT_EQ(vec.get_owning_bndry_size(i), 3);
    }       
}

TEST(OverlappingVector, Fill) {
    auto vec = setup_three_part_vector();
    vec.fill(1.0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 7; j++) {
            ASSERT_EQ(vec.inner_data[i]->at(j, 0), 1.0);
            ASSERT_EQ(vec.bndry_data[i]->at(j, 0), 1.0);
        }
    }
}

TEST(OverlappingVector, MakeConsistent) {
    auto vec = setup_three_part_vector();
    vec.fill(1.0);
#pragma omp parallel
    {
#pragma omp single
        {
            vec.make_consistent();
        }
#pragma omp taskwait
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 7; j++) {
            ASSERT_EQ(vec.inner_data[i]->at(j, 0), 1.0);
        }
    }

    ASSERT_EQ(vec.global_bndry->at(0, 0), 2.0);
    ASSERT_EQ(vec.global_bndry->at(1, 0), 2.0);
    ASSERT_EQ(vec.global_bndry->at(2, 0), 3.0);
    ASSERT_EQ(vec.global_bndry->at(3, 0), 2.0);
    ASSERT_EQ(vec.global_bndry->at(4, 0), 2.0);
    ASSERT_EQ(vec.global_bndry->at(5, 0), 3.0);
    ASSERT_EQ(vec.global_bndry->at(6, 0), 2.0);
    ASSERT_EQ(vec.global_bndry->at(7, 0), 2.0);
    ASSERT_EQ(vec.global_bndry->at(8, 0), 3.0);
}

TEST(OverlappingVector, Dot) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto vec1 = std::make_shared<overlapping_vector>(setup_three_part_vector());
    auto vec2 = std::make_shared<overlapping_vector>(setup_three_part_vector());
    vec1->fill(1.0);
    vec2->fill(2.0);
    auto res = gko::share(gko::initialize<vec>({0.0}, exec));
    vec1->dot(vec2, res);

    ASSERT_EQ(res->at(0, 0), 60.0);
}
