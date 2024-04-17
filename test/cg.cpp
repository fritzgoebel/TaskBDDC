#include "../cg.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>
#include <memory>

#include "ginkgo/core/stop/residual_norm.hpp"
#include "util.hpp"
#include "matrices/config.hpp"

TEST(cg, ApplyEquivalentToGinkgo)
{
    using part = gko::experimental::distributed::Partition<int, int>;
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;
    
    auto exec = gko::ReferenceExecutor::create();
    std::ifstream in{global_location};
    auto global = gko::share(gko::read_binary<mtx>(in, exec));
    auto x_ref = gko::share(vec::create(exec, gko::dim<2>{global->get_size()[0], 1}));
    x_ref->fill(1.0);
    auto y_ref = gko::share(vec::create(exec, gko::dim<2>{global->get_size()[0], 1}));
    global->apply(x_ref, y_ref);
    x_ref->fill(0.0);
    auto y_check = y_ref->clone();
    auto solver_ref = gko::solver::Cg<double>::build()
        .with_criteria(
            gko::stop::ResidualNorm<double>::build()
                .with_reduction_factor(1e-8).on(exec),
            gko::stop::Iteration::build().with_max_iters(1000u).on(exec))
        .on(exec)->generate(global);

    std::shared_ptr<overlapping_vector> x, y;
    auto local = matrices::local();
    auto partition = matrices::build_partition();
    auto inner = matrices::inner();
    auto bndry = matrices::bndry();
    auto A = std::make_shared<block_matrix>(block_matrix(local, inner, bndry));
    x = std::make_shared<overlapping_vector>(overlapping_vector(inner, bndry, partition, global->get_size()[0]));
    y = std::make_shared<overlapping_vector>(overlapping_vector(inner, bndry, partition, global->get_size()[0]));
#pragma omp parallel 
    {
#pragma omp single
        {
            x->restrict(x_ref);
            y->restrict(y_ref);
            auto solver = cg(A, 100, 1e-8, y);
            solver.apply(y, x);
            x->prolongate(y_ref);
        }
    }
    
    global->apply(y_ref, x_ref);
    for (int i = 0; i < y_ref->get_size()[0]; i++) {
        ASSERT_LE(std::abs(x_ref->at(i, 0) - y_check->at(i, 0)), 1e-6);
    }
}
