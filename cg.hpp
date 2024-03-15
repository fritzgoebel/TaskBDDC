#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <omp.h>
#include <vector>

#include "block_matrix.hpp"
#include "ginkgo/core/matrix/dense.hpp"

struct cg {
    using mtx = gko::matrix::Csr<double>;
    using vec = gko::matrix::Dense<double>;

    cg(std::shared_ptr<block_matrix> A, int max_it, double tol)
        : A(A), max_it(max_it), tol(tol) {
            auto exec = gko::ReferenceExecutor::create();
            auto vec_size = gko::dim<2>{A->size_[0], 1};
            r = vec::create(exec, vec_size);
            p = vec::create(exec, vec_size);
            q = vec::create(exec, vec_size);
            rho = vec::create(exec, gko::dim<2>{1, 1});
            rho_old = vec::create(exec, gko::dim<2>{1, 1});
            alpha = vec::create(exec, gko::dim<2>{1, 1});
            beta = vec::create(exec, gko::dim<2>{1, 1});
            one_op = gko::initialize<vec>({1.0}, exec);
            neg_one_op = gko::initialize<vec>({-1.0}, exec);
        }

    void apply(std::shared_ptr<vec> b, std::shared_ptr<vec> x) {
        r->copy_from(b);
        A->apply(neg_one_op, x, one_op, r);
        p->copy_from(r);
        p->compute_dot(p, rho);
        rho_old->copy_from(rho);

        for (int i = 0; i < max_it; i++) {
            if (rho->at(0,0) < tol * tol) {
                std::cout << "CG converged in " << i << " iterations\n";
                std::cout << "Implicit residual norm: " << sqrt(rho->at(0,0)) << "\n";
                break;
            }
            A->apply(p, q);
            p->compute_dot(q, alpha);
            alpha->at(0,0) = rho->at(0,0) / alpha->at(0,0);
            x->add_scaled(alpha, p);
            r->sub_scaled(alpha, q);
            rho_old->copy_from(rho);
            r->compute_dot(r, rho);
            beta->at(0,0) = rho->at(0,0) / rho_old->at(0,0);
            p->scale(beta);
            p->add_scaled(one_op, r);
        } 
    }

    std::shared_ptr<block_matrix> A;
    std::shared_ptr<vec> r;
    std::shared_ptr<vec> p;
    std::shared_ptr<vec> q;
    std::shared_ptr<vec> rho;
    std::shared_ptr<vec> rho_old;
    std::shared_ptr<vec> alpha;
    std::shared_ptr<vec> beta;
    std::shared_ptr<vec> one_op;
    std::shared_ptr<vec> neg_one_op;
    int max_it;
    double tol;
};
