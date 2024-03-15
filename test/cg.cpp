#include "../cg.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>
#include <memory>

#include "util.hpp"

TEST(cg, ApplySinglePart)
{
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto matrix = std::make_shared<block_matrix>(util::create_tridiag_matrix(1, 7));
    auto x = gko::share(gko::initialize<vec>(7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, exec));
    auto y = gko::share(gko::clone(x));
    y->fill(0.0);
    auto b = gko::share(gko::clone(x));
    matrix->apply(x, b);

    auto solver = cg(matrix, 100, 1e-8);

    solver.apply(b, y);

    util::assume_vectors_near(y, x, 1e-14);
}

TEST(cg, ApplyMultipleParts)
{
    using vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto matrix = std::make_shared<block_matrix>(util::create_tridiag_matrix(3, 3));
    auto x = gko::share(gko::initialize<vec>(7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, exec));
    auto y = gko::share(gko::clone(x));
    y->fill(0.0);
    auto b = gko::share(gko::clone(x));
    matrix->apply(x, b);

    auto solver = cg(matrix, 100, 1e-8);

    solver.apply(b, y);

    util::assume_vectors_near(y, x, 1e-14);
}
