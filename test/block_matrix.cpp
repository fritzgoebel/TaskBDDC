#include "../block_matrix.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>
#include <memory>

#include "ginkgo/core/base/executor.hpp"
#include "ginkgo/core/base/mtx_io.hpp"
#include "ginkgo/core/matrix/dense.hpp"
#include "matrices/config.hpp"
#include "util.hpp"


// Test the constructor
TEST(BlockMatrix, ConstructorSinglePart) {
#pragma omp parallel
    {
#pragma omp single
        {
            auto matrix = util::create_tridiag_matrix(1, 7);
#pragma omp taskwait
            util::assume_correct_matrix(1, 7, matrix);
        }
    }
}

TEST(BlockMatrix, ConstructorMultipleParts) {
    auto matrix = util::create_tridiag_matrix(3, 10);
    util::assume_correct_matrix(3, 10, matrix);
}

// Test the apply method
TEST(BlockMatrix, ApplySinglePart) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto matrix = util::create_tridiag_matrix(1, 7);
    auto x = gko::share(gko::initialize<vec>(7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, exec));
    auto y = gko::share(gko::clone(x));
#pragma omp parallel
    {
#pragma omp single
        {
            matrix.apply(x, y);
#pragma omp taskwait
        }
    }
    ASSERT_EQ(y->at(0, 0), 1.0);
    ASSERT_EQ(y->at(1, 0), 1.0);
    ASSERT_EQ(y->at(2, 0), 0.0);
    ASSERT_EQ(y->at(3, 0), 0.0);
    ASSERT_EQ(y->at(4, 0), 0.0);
    ASSERT_EQ(y->at(5, 0), 7.0);
    ASSERT_EQ(y->at(6, 0), 7.0);
}

TEST(BlockMatrix, ApplyMultipleParts) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto matrix = util::create_tridiag_matrix(3, 3);
    auto x = gko::share(gko::initialize<vec>(7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, exec));
    auto y = gko::share(gko::clone(x));
#pragma omp parallel
    {
#pragma omp single
        {
            matrix.apply(x, y);
#pragma omp taskwait
        }
    }
    ASSERT_EQ(y->at(0, 0), 1.0);
    ASSERT_EQ(y->at(1, 0), 1.0);
    ASSERT_EQ(y->at(2, 0), 0.0);
    ASSERT_EQ(y->at(3, 0), 0.0);
    ASSERT_EQ(y->at(4, 0), 0.0);
    ASSERT_EQ(y->at(5, 0), 7.0);
    ASSERT_EQ(y->at(6, 0), 7.0);
}

TEST(BlockMatrix, ApplyLargeParts) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    int n_parts = 8;
    int local_size = 432;
    auto matrix = util::create_tridiag_matrix(n_parts, local_size);
    auto x = gko::share(vec::create(exec, gko::dim<2>{n_parts * (local_size - 1) + 1, 1}));
    std::iota(x->get_values(), x->get_values() + x->get_size()[0], 1);
    auto y = gko::share(gko::clone(x));
#pragma omp parallel
    {
#pragma omp single
        {
            matrix.apply(x, y);

#pragma omp taskwait
        }
    }
    ASSERT_EQ(y->at(0, 0), 1.0);
    ASSERT_EQ(y->at(1, 0), 1.0);
    for (int i = 2; i < y->get_size()[0] - 2; i++) {
        ASSERT_EQ(y->at(i, 0), 0.0);
    }
    ASSERT_EQ(y->at(y->get_size()[0] - 1, 0), n_parts * (local_size - 1) + 1);
}

TEST(BlockMatrix, ApplyOverlappingVector) {
    using part = gko::experimental::distributed::Partition<int, int>;
    std::shared_ptr<overlapping_vector> x, y;
#pragma omp parallel shared(x, y)
    {
#pragma omp single
        {
            auto exec = gko::ReferenceExecutor::create();
            int n_parts = 3;
            int local_size = 10;
            std::vector<std::vector<int>> inner_idxs{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {10, 11, 12, 13, 14, 15, 16, 17}, {19, 20, 21, 22, 23, 24, 25, 26, 27}};
            std::vector<std::vector<int>> bndry_idxs{{9}, {9, 18}, {18}};
            auto matrix = util::create_tridiag_matrix(n_parts, local_size);
            gko::array<int> map(exec, 28);
            for (auto i = 0; i < 28; i++) {
                map.get_data()[i] = i < 10 ? 0 : i < 19 ? 1 : 2;
            }
            auto partition = gko::share(part::build_from_mapping(exec, map, 3));
            x = std::make_shared<overlapping_vector>(overlapping_vector(inner_idxs, bndry_idxs, partition, 28));
            y = std::make_shared<overlapping_vector>(overlapping_vector(inner_idxs, bndry_idxs, partition, 28));
            x->fill(1.0);
            y->fill(0.0);
            matrix.apply(x, y);
#pragma omp taskwait
        }
    }
    ASSERT_EQ(y->data[0]->at(0,0), 1.0);
    ASSERT_EQ(y->data[0]->at(1,0), 1.0);
    for (int i = 2; i < 10; i++) {
        ASSERT_EQ(y->data[0]->at(i,0), 0.0);
    }
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(y->data[1]->at(i,0), 0.0);
    }
    for (int i = 0; i < 7; i++) {
        ASSERT_EQ(y->data[2]->at(i,0), 0.0);
    }
    ASSERT_EQ(y->data[2]->at(7,0), 1.0);
    ASSERT_EQ(y->data[2]->at(8,0), 1.0);
    ASSERT_EQ(y->data[2]->at(9,0), 0.0);
}

TEST(BlockMatrix, ApplyOverlappingVectorLarge) {
    using part = gko::experimental::distributed::Partition<int, int>;
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;
    
    auto exec = gko::ReferenceExecutor::create();
    std::ifstream in{global_location};
    auto global = gko::read_binary<mtx>(in, exec);
    auto x_ref = gko::share(vec::create(exec, gko::dim<2>{global->get_size()[0], 1}));
    x_ref->fill(1.0);
    auto y_ref = vec::create(exec, gko::dim<2>{global->get_size()[0], 1});
    y_ref->fill(0.0);
    global->apply(x_ref, y_ref);

    std::shared_ptr<overlapping_vector> x, y;
    auto local = matrices::local();
    auto partition = matrices::build_partition();
    auto inner = matrices::inner();
    auto bndry = matrices::bndry();
    auto A = block_matrix(local, inner, bndry);
    x = std::make_shared<overlapping_vector>(overlapping_vector(A.inner_idxs, A.bndry_idxs, partition, global->get_size()[0]));
    y = std::make_shared<overlapping_vector>(overlapping_vector(A.inner_idxs, A.bndry_idxs, partition, global->get_size()[0]));
    x->restrict(x_ref);
    y->fill(0.0);
    A.apply(x, y);
    y->prolongate(x_ref);

    for (int i = 0; i < y_ref->get_size()[0]; i++) {
        ASSERT_LE(std::abs(y_ref->at(i, 0) - x_ref->at(i, 0)), 1e-9);
    }
}

TEST(BlockMatrix, AdvancedApplyOverlappingVectorLarge) {
    using part = gko::experimental::distributed::Partition<int, int>;
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;
    
    auto exec = gko::ReferenceExecutor::create();
    std::ifstream in{global_location};
    auto global = gko::read_binary<mtx>(in, exec);
    auto x_ref = gko::share(vec::create(exec, gko::dim<2>{global->get_size()[0], 1}));
    x_ref->fill(1.0);
    auto y_ref = vec::create(exec, gko::dim<2>{global->get_size()[0], 1});
    y_ref->fill(1.0);
    auto alpha = gko::share(gko::initialize<vec>(1, {2.0}, exec));
    auto beta = gko::share(gko::initialize<vec>(1, {3.0}, exec));
    global->apply(alpha, x_ref, beta, y_ref);

    std::shared_ptr<overlapping_vector> x, y;
    auto local = matrices::local();
    auto partition = matrices::build_partition();
    auto inner = matrices::inner();
    auto bndry = matrices::bndry();
    auto A = block_matrix(local, inner, bndry);
    x = std::make_shared<overlapping_vector>(overlapping_vector(A.inner_idxs, A.bndry_idxs, partition, global->get_size()[0]));
    y = std::make_shared<overlapping_vector>(overlapping_vector(A.inner_idxs, A.bndry_idxs, partition, global->get_size()[0]));

#pragma omp parallel
    {
#pragma omp single
        {
            x->restrict(x_ref);
            y->copy_from(x);
            A.apply(alpha, x, beta, y);
            y->prolongate(x_ref);
        }
    }

    for (int i = 0; i < y_ref->get_size()[0]; i++) {
        ASSERT_LE(std::abs(y_ref->at(i, 0) - x_ref->at(i, 0)), 1.5e-9);
    }
}

