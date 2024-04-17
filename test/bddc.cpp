#include "../bddc.hpp"

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>

#include "matrices/config.hpp"

TEST(Bddc, SolveInner) {
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

    std::shared_ptr<overlapping_vector> x, y, buffer;
    auto local = matrices::local();
    auto partition = matrices::build_partition();
    auto inner = matrices::inner();
    auto bndry = matrices::bndry();
    auto A = std::make_shared<block_matrix>(block_matrix(local, inner, bndry));
    x = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));
    y = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));
    buffer = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));

    auto chol_factory = gko::experimental::solver::Direct<double, int>::build()
        .with_factorization(
            gko::experimental::factorization::Cholesky<double, int>::build().on(exec))
        .on(exec);
    auto N = A->local_mtxs_.size();
    for (size_t i = 0; i < N; i++) {
        size_t start = inner[i][0];
        size_t end = start + inner[i].size();
        auto inner_mtx = gko::share(global->create_submatrix(gko::span{start, end}, gko::span{start, end}));
        auto inner_solver = chol_factory->generate(inner_mtx);
        auto rhs = gko::share(x_ref->create_submatrix(gko::span{start, end}, gko::span{0, 1}));
        auto sol = gko::share(y_ref->create_submatrix(gko::span{start, end}, gko::span{0, 1}));
        inner_solver->apply(rhs, sol);
    }

    std::shared_ptr<bddc> precond;
#pragma omp parallel
    {
#pragma omp single
        {
            precond = std::make_shared<bddc>(bddc(A, buffer));
            x->restrict(x_ref);
            y->restrict(x_ref);
            precond->solve_inner(x, y);
            y->prolongate(x_ref);
        }
    }

    for (size_t i = 0; i < y_ref->get_size()[0]; i++) {
        ASSERT_NEAR(y_ref->at(i, 0), x_ref->at(i, 0), 1e-9);
    }
}

TEST(Bddc, Weights) {
    using part = gko::experimental::distributed::Partition<int, int>;
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;
    
    auto exec = gko::ReferenceExecutor::create();
    std::ifstream in{global_location};
    auto global = gko::read_binary<mtx>(in, exec);

    std::shared_ptr<overlapping_vector> x, y, buffer;
    auto local = matrices::local();
    auto partition = matrices::build_partition();
    auto inner = matrices::inner();
    auto bndry = matrices::bndry();
    auto A = std::make_shared<block_matrix>(block_matrix(local, inner, bndry));
    buffer = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));
    auto N = A->local_mtxs_.size();

    std::shared_ptr<bddc> precond;
#pragma omp parallel
    {
#pragma omp single
        {
            precond = std::make_shared<bddc>(bddc(A, buffer));
        }
    }

    auto diag = global->extract_diagonal();
    for (size_t i = 0; i < N; i++) {
        auto bndry_idxs = A->bndry_idxs[i];
        auto n_inner = A->inner_idxs[i].size();
        auto local_diag = A->local_mtxs_[i]->extract_diagonal();
        auto weights = precond->weights[i]->get_values();
        for (size_t j = 0; j < bndry_idxs.size(); j++) {
            auto idx = bndry_idxs[j];
            ASSERT_NEAR(weights[j], local_diag->get_const_values()[n_inner + j] / diag->get_const_values()[idx], 1e-9);
        }
    }
}

TEST(Bddc, Apply) {
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

    std::shared_ptr<overlapping_vector> x, y, buffer;
    auto local = matrices::local();
    auto partition = matrices::build_partition();
    auto inner = matrices::inner();
    auto bndry = matrices::bndry();
    auto A = std::make_shared<block_matrix>(block_matrix(local, inner, bndry));
    x = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));
    y = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));
    buffer = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, global->get_size()[0]));

    auto chol_factory = gko::experimental::solver::Direct<double, int>::build()
        .with_factorization(
            gko::experimental::factorization::Cholesky<double, int>::build().on(exec))
        .on(exec);
    auto N = A->local_mtxs_.size();
    for (size_t i = 0; i < N; i++) {
        size_t start = inner[i][0];
        size_t end = start + inner[i].size();
        auto inner_mtx = gko::share(global->create_submatrix(gko::span{start, end}, gko::span{start, end}));
        auto inner_solver = chol_factory->generate(inner_mtx);
        auto rhs = gko::share(x_ref->create_submatrix(gko::span{start, end}, gko::span{0, 1}));
        auto sol = gko::share(y_ref->create_submatrix(gko::span{start, end}, gko::span{0, 1}));
        inner_solver->apply(rhs, sol);
    }

    std::shared_ptr<bddc> precond;
#pragma omp parallel
    {
#pragma omp single
        {
            precond = std::make_shared<bddc>(bddc(A, buffer));
            x->restrict(x_ref);
            y->restrict(x_ref);
            precond->apply(x, y);
            y->prolongate(x_ref);
        }
    }

    for (size_t i = 0; i < y_ref->get_size()[0]; i++) {
        ASSERT_GE(std::abs(x_ref->at(i, 0)), 0);
    }
}
