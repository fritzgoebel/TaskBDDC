#include "../overlapping_vector.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>
#include <vector>

#include "ginkgo/core/matrix/dense.hpp"
#include "vector_utils.hpp"

TEST(OverlappingVector, ConstructorSinglePart) {
    using part = gko::experimental::distributed::Partition<int, int>;
    std::vector<std::vector<int>> inner_idxs{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
    std::vector<std::vector<int>> bndry_idxs{{}};
    auto exec = gko::ReferenceExecutor::create();
    auto partition = gko::share(part::build_from_global_size_uniform(exec, 1, 10));

    overlapping_vector vec;
#pragma omp parallel
    {
#pragma omp single
        {
            vec = overlapping_vector(inner_idxs, bndry_idxs, partition, 10, exec);
        }
    }
    ASSERT_EQ(vec.get_size()[0], 10);
    ASSERT_EQ(vec.get_size()[1], 1);
    ASSERT_EQ(vec.get_num_parts(), 1);
}

TEST(OverlappingVector, ConstructorMultipleParts) {
    overlapping_vector vec;
#pragma omp parallel
    {
#pragma omp single
        {
            vec = setup_three_part_vector();
        }
    }

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
    overlapping_vector vec;
#pragma omp parallel
    {
#pragma omp single
        {
            vec = setup_three_part_vector();
            vec.fill(1.0);
        }
    }

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

TEST(OverlappingVector, RestrictProlongate) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto ref = gko::share(vec::create(exec, gko::dim<2>{30, 1}));
    std::iota(ref->get_values(), ref->get_values() + 30, 0.0);
    auto x = gko::share(clone(ref));
    x->fill(0.0);
    auto y = setup_three_part_vector();
#pragma omp parallel
    {
#pragma omp single
        {
            y.restrict(ref);
            y.prolongate(x);
        }
    }
    for (int i = 0; i < 30; i++) {
        ASSERT_EQ(ref->at(i, 0), x->at(i, 0));
    }
}

TEST(OverlappingVector, Scale) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto ref = gko::share(vec::create(exec, gko::dim<2>{30, 1}));
    std::iota(ref->get_values(), ref->get_values() + 30, 0.0);
    auto alpha = gko::share(gko::initialize<vec>({2.0}, exec));
    auto x = gko::share(clone(ref));
    x->fill(0.0);
    auto y = setup_three_part_vector();
#pragma omp parallel
    {
#pragma omp single
        {
            y.restrict(ref);
            y.scale(alpha);
            y.prolongate(x);
        }
    }
    ref->scale(alpha);
    for (int i = 0; i < 30; i++) {
        ASSERT_EQ(ref->at(i, 0), x->at(i, 0));
    }
}

TEST(OverlappingVector, AddScaled) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto ref = gko::share(vec::create(exec, gko::dim<2>{30, 1}));
    auto ref1 = gko::share(gko::clone(ref));
    std::iota(ref->get_values(), ref->get_values() + 30, 0.0);
    ref1->fill(1.0);
    auto alpha = gko::share(gko::initialize<vec>({2.0}, exec));
    auto x = gko::share(clone(ref));
    x->fill(0.0);
    auto y = setup_three_part_vector();
    auto y1 = std::make_shared<overlapping_vector>(setup_three_part_vector());
#pragma omp parallel
    {
#pragma omp single
        {
            y.restrict(ref);
            y1->restrict(ref1);
            y.add_scaled(alpha, y1);
            y.prolongate(x);
        }
    }
    ref->add_scaled(alpha, ref1);
    for (int i = 0; i < 30; i++) {
        ASSERT_EQ(ref->at(i, 0), x->at(i, 0));
    }
}
