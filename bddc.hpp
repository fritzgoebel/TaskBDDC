#include <ginkgo/ginkgo.hpp>
#include <set>
#include <vector>
#include "block_matrix.hpp"
#include "ginkgo/core/base/array.hpp"
#include "ginkgo/core/matrix/permutation.hpp"
#include "ginkgo/core/reorder/amd.hpp"

struct bddc{
    using mtx = gko::matrix::Csr<double>;
    using vec = gko::matrix::Dense<double>;
    using diag = gko::matrix::Diagonal<double>;

    bddc(std::shared_ptr<block_matrix> A, std::shared_ptr<overlapping_vector> buffer) : A(A), interfaces{A->interfaces}, local_interfaces{A->local_interfaces} {
        auto N = A->local_mtxs_.size();
        auto A_size = A->size_;
        inner_solvers.resize(N);
        inner_amd.resize(N);
        inner_rhs.resize(N);
        inner_sol.resize(N);
        A_gi.resize(N);
        A_ig.resize(N);
        workspace_1 = buffer->clone();
        workspace_2 = buffer->clone();
        workspace_3 = buffer->clone();
#pragma omp taskwait
        for (size_t i = 0; i < N; i++) {
#pragma omp task shared(inner_solvers)
            {
                // Set up inner solvers
                auto local_mtx = A->local_mtxs_[i];
                auto inner_mtx = gko::share(A->inner_mtxs_[i]->create_submatrix(gko::span{0, A->inner_mtxs_[i]->get_size()[0]}, gko::span{0, A->inner_mtxs_[i]->get_size()[0]}));
                auto exec = local_mtx->get_executor();
                inner_amd[i] = gko::experimental::reorder::Amd<int>::build().on(exec)->generate(inner_mtx);
                auto perm_A = gko::share(inner_mtx->permute(inner_amd[i]));
                inner_solvers[i] = gko::experimental::solver::Direct<double, int>::build()
                    .with_factorization(
                        gko::experimental::factorization::Cholesky<double, int>::build().on(exec))
                    .on(exec)->generate(perm_A);
                inner_rhs[i] = vec::create(exec, gko::dim<2>{inner_mtx->get_size()[0], 1});
                inner_sol[i] = vec::create(exec, gko::dim<2>{inner_mtx->get_size()[0], 1});
            }
        }

        one = gko::initialize<vec>({1.0}, gko::ReferenceExecutor::create());
        neg_one = gko::initialize<vec>({-1.0}, gko::ReferenceExecutor::create());

        // Set up stiffness scaling operators
        weights.resize(N);
        std::vector<std::shared_ptr<vec>> local_diag_vec(N);
        for (size_t i = 0; i < N; i++) {
#pragma omp task shared (local_diag_vec, buffer) depend (out: buffer->bndry_data[i])
            {
                auto local_mtx = A->local_mtxs_[i];
                auto exec = local_mtx->get_executor();
                auto local_diag = gko::share(local_mtx->extract_diagonal());
                auto local_diag_array = gko::make_const_array_view(exec, local_diag->get_size()[0], local_diag->get_const_values());
                local_diag_vec[i] = gko::clone(vec::create_const(exec, gko::dim<2>{local_diag->get_size()[0], 1}, std::move(local_diag_array), 1));
                buffer->data[i]->copy_from(local_diag_vec[i]);
            }
        }
#pragma omp taskwait
        buffer->make_consistent();
#pragma omp taskwait
        for (size_t i = 0; i < N; i++) {
#pragma omp task shared (local_diag_vec, buffer) depend (in: buffer->bndry_data[i])
            {
                auto exec = A->local_mtxs_[i]->get_executor();
                auto n = buffer->bndry_data[i]->get_size()[0];
                auto n_inner = buffer->inner_data[i]->get_size()[0];
                auto global_diag_array = gko::make_const_array_view(exec, n, buffer->bndry_data[i]->get_const_values());
                auto global_diag = diag::create_const(exec, n, std::move(global_diag_array));
                auto weights_vec = vec::create(exec, gko::dim<2>{n, 1});
                auto local_bndry_diag = local_diag_vec[i]->create_submatrix(gko::span{n_inner, n_inner + n}, gko::span{0, 1});
                global_diag->inverse_apply(local_bndry_diag, weights_vec);
                auto weights_array = gko::make_const_array_view(exec, n, weights_vec->get_const_values());
                weights[i] = gko::clone(diag::create_const(exec, n, std::move(weights_array)));
            }
        }

        // Set up restriction operators, constraints and local solvers
        C.resize(N);
        CT.resize(N);
        local_solvers.resize(N);
        local_amd.resize(N);
        local_rhs.resize(N);
        local_sol.resize(N);
        local_schur_solvers.resize(N);
        phi.resize(N);
        phi_t.resize(N);
        schur_rhs.resize(N);
        schur_sol.resize(N);
        gko::matrix_data<double, int> coarse_data(gko::dim<2>{interfaces.size(), interfaces.size()});
        for (size_t i = 0; i < N; i++) {
#pragma omp task shared (coarse_data)
            {
                auto local_mtx = A->local_mtxs_[i];
                auto exec = local_mtx->get_executor();
                size_t n_inner_dofs = A->inner_idxs[i].size();
                size_t n_edges = 0;
                size_t n_edge_dofs = 0;
                size_t n_corner_dofs = 0;
                for (size_t j = 0; j < local_interfaces[i].size(); j++) {
                    auto interf_size = interfaces[local_interfaces[i][j]].second.size();
                    if (interf_size > 1) {
                        n_edges++;
                        n_edge_dofs += interf_size;
                    } else {
                        n_corner_dofs++;
                    }
                }

                gko::matrix_data<double, int> C_data(gko::dim<2>{n_edges, n_inner_dofs + n_edge_dofs});
                size_t start = n_inner_dofs;
                for (size_t j = 0; j < n_edges; j++) {
                    auto interf_size = interfaces[local_interfaces[i][j]].second.size();
                    double val = 1.0 / interf_size;
                    for (size_t k = start; k < start + interf_size; k++) {
                        C_data.nonzeros.emplace_back(j, k, val);
                    }
                    start += interf_size;
                }
                C[i] = mtx::create(exec);
                C[i]->read(C_data);
                CT[i] = gko::as<mtx>(C[i]->transpose());

                auto A_ee = gko::share(local_mtx->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{0, n_inner_dofs + n_edge_dofs}));
                auto A_ec = gko::share(local_mtx->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{n_inner_dofs + n_edge_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}));
                auto A_ce = gko::share(local_mtx->create_submatrix(gko::span{n_inner_dofs + n_edge_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}, gko::span{0, n_inner_dofs + n_edge_dofs}));
                auto A_cc = gko::share(local_mtx->create_submatrix(gko::span{n_inner_dofs + n_edge_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}, gko::span{n_inner_dofs + n_edge_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}));
                A_gi[i] = gko::share(local_mtx->create_submatrix(gko::span{n_inner_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}, gko::span{0, n_inner_dofs}));
                A_ig[i] = gko::share(local_mtx->create_submatrix(gko::span{0, n_inner_dofs}, gko::span{n_inner_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}));
                local_amd[i] = gko::experimental::reorder::Amd<int>::build().on(exec)->generate(A_ee);
                auto perm_A = gko::share(A_ee->permute(local_amd[i]));
                local_solvers[i] = gko::experimental::solver::Direct<double, int>::build()
                    .with_factorization(
                        gko::experimental::factorization::Cholesky<double, int>::build().on(exec))
                    .on(exec)->generate(perm_A);
                local_rhs[i] = vec::create(exec, gko::dim<2>{n_inner_dofs + n_edge_dofs, 1});
                local_sol[i] = vec::create(exec, gko::dim<2>{n_inner_dofs + n_edge_dofs, 1});
                auto setup_solvers = std::vector<std::shared_ptr<gko::LinOp>>(n_edges + n_corner_dofs);
                /* for (auto k = 0; k < n_edges + n_corner_dofs; k++) { */
/* #pragma omp task shared (setup_solvers) */
                /*     setup_solvers[k] = gko::clone(local_solvers[i]); */
                /* } */
/* #pragma omp taskwait */

                std::shared_ptr<vec> local_schur_complement = vec::create(exec, gko::dim<2>{n_edges, n_edges});
                auto dense_CT = gko::share(vec::create(exec));
                dense_CT->copy_from(CT[i]);
                auto intermediate = gko::share(dense_CT->clone());
                for (size_t j = 0; j < n_edges; j++) {
/* #pragma omp task shared(dense_CT, intermediate, setup_solvers) */
/*                     { */
                        auto rhs = gko::share(dense_CT->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{j, j + 1}));
                        auto sol = gko::share(intermediate->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{j, j + 1}));
                        rhs->permute(local_amd[i], local_rhs[i], gko::matrix::permute_mode::rows);
                        local_solvers[i]->apply(local_rhs[i], local_sol[i]);
                        local_sol[i]->permute(local_amd[i], sol, gko::matrix::permute_mode::inverse_rows);
                        /* setup_solvers[j]->apply(rhs, sol); */
                    /* } */
                }
//#pragma omp taskwait
                C[i]->apply(intermediate, local_schur_complement);
                auto ls = gko::share(mtx::create(exec));
                ls->copy_from(local_schur_complement);
                local_schur_solvers[i] = gko::experimental::solver::Direct<double, int>::build()
                    .with_factorization(
                        gko::experimental::factorization::Cholesky<double, int>::build().on(exec))
                    .on(exec)->generate(ls);
                schur_rhs[i] = vec::create(exec, gko::dim<2>{n_edges, 1});
                schur_sol[i] = vec::create(exec, gko::dim<2>{n_edges, 1});

                auto phi_whole = vec::create(exec, gko::dim<2>{n_inner_dofs + n_edge_dofs + n_corner_dofs, n_edges + n_corner_dofs});
                auto lambda = vec::create(exec, gko::dim<2>{n_edges + n_corner_dofs, n_edges + n_corner_dofs});
                phi_whole->fill(0.0);
                lambda->fill(0.0);
                for (size_t j = 0; j < n_corner_dofs; j++) {
                    phi_whole->at(n_inner_dofs + n_edge_dofs + j, n_edges + j) = 1.0;
                }
                auto phi_e = gko::share(phi_whole->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{0, n_edges + n_corner_dofs}));
                auto phi_c = gko::share(phi_whole->create_submatrix(gko::span{n_inner_dofs + n_edge_dofs, n_inner_dofs + n_edge_dofs + n_corner_dofs}, gko::span{0, n_edges + n_corner_dofs}));
                auto lambda_e = gko::share(lambda->create_submatrix(gko::span{0, n_edges}, gko::span{0, n_edges + n_corner_dofs}));
                auto lambda_c = gko::share(lambda->create_submatrix(gko::span{n_edges, n_edges + n_corner_dofs}, gko::span{0, n_edges + n_corner_dofs}));
                
                auto rhs = gko::share(vec::create(exec, gko::dim<2>{n_inner_dofs + n_edge_dofs, n_edges + n_corner_dofs}));
                auto schur_rhs = gko::share(vec::create(exec, gko::dim<2>{n_edges, n_edges + n_corner_dofs}));
                schur_rhs->fill(0.0);
                for (size_t j = 0; j < n_edges; j++) {
                    schur_rhs->at(j, j) = 1.0;
                }
                auto schur_interm = rhs->clone();
                schur_interm->fill(0.0);
                A_ec->apply(phi_c, rhs);
                rhs->scale(neg_one);
                for (size_t j = 0; j < n_edges + n_corner_dofs; j++) {
/* #pragma omp task shared(rhs, schur_interm, setup_solvers) */
/*                     { */
                        auto rhs_sub = rhs->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{j, j + 1});
                        auto sol_sub = schur_interm->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{j, j + 1});
                        rhs_sub->permute(local_amd[i], local_rhs[i], gko::matrix::permute_mode::rows);
                        local_solvers[i]->apply(local_rhs[i], local_sol[i]);
                        local_sol[i]->permute(local_amd[i], sol_sub, gko::matrix::permute_mode::inverse_rows);
                        /* setup_solvers[j]->apply(rhs_sub, sol_sub); */
                    /* } */
                }
//#pragma omp taskwait
                C[i]->apply(one, schur_interm, neg_one, schur_rhs);
                for (size_t j = 0; j < n_edges + n_corner_dofs; j++) {
                    auto rhs_sub = schur_rhs->create_submatrix(gko::span{0, n_edges}, gko::span{j, j + 1});
                    auto sol_sub = lambda_e->create_submatrix(gko::span{0, n_edges}, gko::span{j, j + 1});
                    local_schur_solvers[i]->apply(rhs_sub, sol_sub);
                }
                CT[i]->apply(neg_one, lambda_e, one, rhs);
                for (size_t j = 0; j < n_edges + n_corner_dofs; j++) {
/* #pragma omp task shared(rhs, phi_e, setup_solvers) */
/*                     { */
                        auto rhs_sub = rhs->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{j, j + 1});
                        auto phi_sub = phi_e->create_submatrix(gko::span{0, n_inner_dofs + n_edge_dofs}, gko::span{j, j + 1});
                        rhs_sub->permute(local_amd[i], local_rhs[i], gko::matrix::permute_mode::rows);
                        local_solvers[i]->apply(local_rhs[i], local_sol[i]);
                        local_sol[i]->permute(local_amd[i], phi_sub, gko::matrix::permute_mode::inverse_rows);
                        /* setup_solvers[j]->apply(rhs_sub, phi_sub); */
                    /* } */
                }
//#pragma omp taskwait
                A_cc->apply(phi_c, lambda_c);
                A_ce->apply(neg_one, phi_e, neg_one, lambda_c);
                gko::matrix_data<double, int> phi_data(gko::dim<2>{n_edge_dofs + n_corner_dofs, interfaces.size()});
                for (size_t j = 0; j < n_edge_dofs + n_corner_dofs; j++) {
                    for (size_t k = 0; k < n_edges + n_corner_dofs; k++) {
                        phi_data.nonzeros.emplace_back(j, local_interfaces[i][k], phi_whole->at(n_inner_dofs + j, k));
                    }
                }
                phi_data.remove_zeros();
                phi[i] = mtx::create(exec);
                phi[i]->read(phi_data);
                phi_t[i] = gko::as<mtx>(phi[i]->transpose());
#pragma omp critical
                {
                    for (size_t j = 0; j < local_interfaces[i].size(); j++) {
                        coarse_data.nonzeros.emplace_back(local_interfaces[i][j], local_interfaces[i][j], -lambda->at(j, j));
                        for (size_t k = j + 1; k < local_interfaces[i].size(); k++) {
                            coarse_data.nonzeros.emplace_back(local_interfaces[i][j], local_interfaces[i][k], -lambda->at(j, k));
                            coarse_data.nonzeros.emplace_back(local_interfaces[i][k], local_interfaces[i][j], -lambda->at(j, k));
                        }
                    }
                }
            }
        }
#pragma omp taskwait
        auto exec = A->local_mtxs_[0]->get_executor();
        coarse_data.sum_duplicates();
        coarse_data.remove_zeros();
        auto A_coarse = gko::share(mtx::create(exec));
        A_coarse->read(coarse_data);
        coarse_solver = gko::experimental::solver::Direct<double, int>::build()
            .with_factorization(
                gko::experimental::factorization::Cholesky<double, int>::build().on(exec))
            .on(exec)->generate(A_coarse);
        coarse_rhs = vec::create(exec, gko::dim<2>{interfaces.size(), 1});
        coarse_sol = vec::create(exec, gko::dim<2>{interfaces.size(), 1});
    }


    void solve_inner(std::shared_ptr<overlapping_vector> b, std::shared_ptr<overlapping_vector> x)
    {
        auto N = A->local_mtxs_.size();
        for (size_t i = 0; i < N; i++) {
#pragma omp task shared(inner_solvers) depend (in: b->inner_data[i]) depend(out: x->inner_data[i])
            {
                b->inner_data[i]->permute(inner_amd[i], inner_rhs[i], gko::matrix::permute_mode::rows);
                inner_solvers[i]->apply(inner_rhs[i], inner_sol[i]);
                inner_sol[i]->permute(inner_amd[i], x->inner_data[i], gko::matrix::permute_mode::inverse_rows);
            }
        }
    }

    void static_condensation_1(std::shared_ptr<overlapping_vector> b, std::shared_ptr<overlapping_vector> x)
    {
        auto N = A->local_mtxs_.size();
        solve_inner(b, x);
        for (size_t i = 0; i < N; i++) {
#pragma omp task depend (in: x->inner_data[i]) depend(out: x->bndry_data[i])
            {
                auto exec = A->local_mtxs_[i]->get_executor();
                x->bndry_data[i]->copy_from(b->bndry_data[i]);
                A_gi[i]->apply(neg_one, workspace_1->inner_data[i], one, x->bndry_data[i]);
            }
        }
    }

    void static_condensation_2(std::shared_ptr<overlapping_vector> b, std::shared_ptr<overlapping_vector> x)
    {
        auto N = A->local_mtxs_.size();
        for (size_t i = 0; i < N; i++) {
#pragma omp task depend (in: b->bndry_data[i]) depend (out: b->inner_data[i], x->bndry_data[i])
            {
                x->bndry_data[i]->copy_from(b->bndry_data[i]);
                A_ig[i]->apply(neg_one, b->bndry_data[i], one, b->inner_data[i]);
            }
        }
        solve_inner(b, x);
    }

    void apply(std::shared_ptr<overlapping_vector> b, std::shared_ptr<overlapping_vector> x)
    {
        auto N = A->local_mtxs_.size();
        workspace_1->fill(0.0);
        static_condensation_1(b, workspace_1);
        coarse_rhs->fill(0.0);
        for (size_t i = 0; i < N; i++) {
#pragma omp task depend (inout: workspace_1->bndry_data[i])
            {
                weights[i]->apply(workspace_1->bndry_data[i], workspace_1->bndry_data[i]);
            }
#pragma omp task depend (in: workspace_1->bndry_data[i]) depend (out: this->coarse_rhs)
            {
#pragma omp critical
                phi_t[i]->apply(one, workspace_1->bndry_data[i], one, coarse_rhs);
            }
#pragma omp task depend (in: workspace_1->bndry_data[i]) depend (out: workspace_2->bndry_data[i])
            {
                workspace_1->inner_data[i]->fill(0.0);
                auto e_end = local_solvers[i]->get_size()[0];
                auto c_end = workspace_1->data[i]->get_size()[0];
                auto e_rhs_orig = workspace_1->data[i]->create_submatrix(gko::span{0, e_end}, gko::span{0, 1});
                auto e_rhs = workspace_3->data[i]->create_submatrix(gko::span{0, e_end}, gko::span{0, 1});
                e_rhs->copy_from(e_rhs_orig);
                auto e_lhs = workspace_2->data[i]->create_submatrix(gko::span{0, e_end}, gko::span{0, 1});
                e_rhs->permute(local_amd[i], local_rhs[i], gko::matrix::permute_mode::rows);
                local_solvers[i]->apply(local_rhs[i], local_sol[i]);
                local_sol[i]->permute(local_amd[i], e_lhs, gko::matrix::permute_mode::inverse_rows);
                C[i]->apply(e_lhs, schur_rhs[i]);
                local_schur_solvers[i]->apply(schur_rhs[i], schur_sol[i]);
                CT[i]->apply(neg_one, schur_sol[i], one, e_rhs);
                e_rhs->permute(local_amd[i], local_rhs[i], gko::matrix::permute_mode::rows);
                local_solvers[i]->apply(local_rhs[i], local_sol[i]);
                local_sol[i]->permute(local_amd[i], e_lhs, gko::matrix::permute_mode::inverse_rows);
                auto c_lhs = workspace_2->data[i]->create_submatrix(gko::span{e_end, c_end}, gko::span{0, 1});
                c_lhs->fill(0.0);
            }
        }
#pragma omp task shared (coarse_rhs, coarse_sol) depend (in: this->coarse_rhs) depend (out: this->coarse_sol)
        {
            coarse_solver->apply(coarse_rhs, coarse_sol);
        }

        for (size_t i = 0; i < N; i++) {
#pragma omp task depend (in: this->coarse_sol) depend (inout: workspace_2->bndry_data[i])
            {
                phi[i]->apply(one, coarse_sol, one, workspace_2->bndry_data[i]);
                weights[i]->apply(workspace_2->bndry_data[i], workspace_2->bndry_data[i]);
                workspace_2->inner_data[i]->copy_from(b->inner_data[i]);
            }
        }

        workspace_2->make_consistent();
        static_condensation_2(workspace_2, x);
    }

    std::shared_ptr<block_matrix> A;
    std::vector<std::shared_ptr<mtx>> A_gi;
    std::vector<std::shared_ptr<mtx>> A_ig;
    std::vector<std::shared_ptr<gko::LinOp>> inner_solvers;
    std::vector<std::shared_ptr<gko::matrix::Permutation<int>>> inner_amd;
    std::vector<std::shared_ptr<vec>> inner_rhs;
    std::vector<std::shared_ptr<vec>> inner_sol;
    std::shared_ptr<overlapping_vector> workspace_1;
    std::shared_ptr<overlapping_vector> workspace_2;
    std::shared_ptr<overlapping_vector> workspace_3;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> interfaces;
    std::vector<std::vector<int>> bndry_to_interfaces;
    std::vector<std::shared_ptr<mtx>> C;
    std::vector<std::shared_ptr<mtx>> CT;
    std::vector<std::shared_ptr<mtx>> R;
    std::vector<std::shared_ptr<mtx>> RT;
    std::vector<std::shared_ptr<gko::LinOp>> local_solvers;
    std::vector<std::shared_ptr<gko::matrix::Permutation<int>>> local_amd;
    std::vector<std::shared_ptr<vec>> local_rhs;
    std::vector<std::shared_ptr<vec>> local_sol;
    std::vector<std::shared_ptr<gko::LinOp>> local_schur_solvers;
    std::vector<std::shared_ptr<vec>> schur_rhs;
    std::vector<std::shared_ptr<vec>> schur_sol;
    std::shared_ptr<gko::LinOp> coarse_solver;
    std::shared_ptr<vec> coarse_rhs;
    std::shared_ptr<vec> coarse_sol;
    std::vector<std::vector<int>> local_interfaces;
    std::shared_ptr<vec> one;
    std::shared_ptr<vec> neg_one;
    std::vector<std::shared_ptr<mtx>> phi;
    std::vector<std::shared_ptr<mtx>> phi_t;
    std::vector<std::shared_ptr<diag>> weights;
};
