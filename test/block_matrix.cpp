#include "../block_matrix.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>

#include "util.hpp"


// Test the constructor
TEST(BlockMatrix, ConstructorSinglePart) {
    auto matrix = util::create_tridiag_matrix(1, 7);
    util::assume_correct_matrix(1, 7, matrix);
}

TEST(BlockMatrix, ConstructorMultipleParts) {
    auto matrix = util::create_tridiag_matrix(3, 30);
    util::assume_correct_matrix(3, 30, matrix);
}

// Test the apply method
TEST(BlockMatrix, ApplySinglePart) {
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto matrix = util::create_tridiag_matrix(1, 7);
    auto x = gko::share(gko::initialize<vec>(7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, exec));
    auto y = gko::share(gko::clone(x));
    matrix.apply(x, y);
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
    matrix.apply(x, y);
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
    matrix.apply(x, y);

    ASSERT_EQ(y->at(0, 0), 1.0);
    ASSERT_EQ(y->at(1, 0), 1.0);
    for (int i = 2; i < y->get_size()[0] - 2; i++) {
        ASSERT_EQ(y->at(i, 0), 0.0);
    }
    ASSERT_EQ(y->at(y->get_size()[0] - 1, 0), n_parts * (local_size - 1) + 1);
}
