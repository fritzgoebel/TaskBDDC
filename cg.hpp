#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <omp.h>
#include <vector>

#include "ginkgo/core/matrix/dense.hpp"

#include "bddc.hpp"

struct cg {
    using mtx = gko::matrix::Csr<double>;
    using vec = gko::matrix::Dense<double>;

    cg(std::shared_ptr<block_matrix> A, int max_it, double tol, std::shared_ptr<overlapping_vector> ref, bool nsp = false)
        : A(A), max_it(max_it), tol(tol), nsp(nsp) {
            auto exec = gko::ReferenceExecutor::create();
            auto vec_size = gko::dim<2>{A->size_[0], 1};
            r = ref->clone();
            p = ref->clone();
            q = ref->clone();
            z = ref->clone();
            rho = vec::create(exec, gko::dim<2>{1, 1});
            rho_old = vec::create(exec, gko::dim<2>{1, 1});
            alpha = vec::create(exec, gko::dim<2>{1, 1});
            beta = vec::create(exec, gko::dim<2>{1, 1});
            one_op = gko::initialize<vec>({1.0}, exec);
            neg_one_op = gko::initialize<vec>({-1.0}, exec);
            prec = std::make_shared<bddc>(bddc(A, z));
            if (nsp) {
                nullsp = ref->clone();
                nullsp->fill(1.0);
                nullsp->dot(nullsp, rho);
                rho->at(0,0) = 1.0 / sqrt(rho->at(0,0));
                nullsp->scale(rho);
            }
        }

    void apply(std::shared_ptr<overlapping_vector> b, std::shared_ptr<overlapping_vector> x) {
        /* prec->apply(b, x); */
        /* x->dot(x, rho); */
        /* std::cout << "Initial residual norm: " << sqrt(std::abs(rho->at(0,0))) << "\n"; */
        r->copy_from(b);
        A->apply(neg_one_op, x, one_op, r);
        rho_old->at(0,0) = 1.0;
        /* //p->copy_from(r); */
        /* prec->apply(r, p); */
        /* r->dot(p, rho); */
        /* rho_old->copy_from(rho); */
        int i = 0;
        for (i = 0; i < max_it; i++) {
            /* z->copy_from(r); */
            prec->apply(r, z);
            r->dot(z, rho);
            if (std::abs(rho->at(0,0)) < tol * tol) {
                break;
            }
            beta->at(0,0) = rho->at(0,0) / rho_old->at(0,0);
            p->scale(beta);
            p->add_scaled(one_op, z);
            A->apply(p, q);
            p->dot(q, alpha);
            alpha->at(0,0) = rho->at(0,0) / alpha->at(0,0);
            x->add_scaled(alpha, p);
            r->sub_scaled(alpha, q);
            rho_old->copy_from(rho);
        }
        if (std::abs(rho->at(0,0)) < tol * tol) {
            std::cout << "CG converged in " << i << " iterations\n";
            std::cout << "Implicit residual norm: " << sqrt(std::abs(rho->at(0,0))) << "\n";
        } else {
            std::cout << "CG did not converge in " << max_it << " iterations\n";
            std::cout << "Implicit residual norm: " << sqrt(std::abs(rho->at(0,0))) << "\n";
        }
    }

    std::shared_ptr<block_matrix> A;
    std::shared_ptr<bddc> prec;
    std::shared_ptr<overlapping_vector> r;
    std::shared_ptr<overlapping_vector> p;
    std::shared_ptr<overlapping_vector> q;
    std::shared_ptr<overlapping_vector> z;
    std::shared_ptr<vec> rho;
    std::shared_ptr<vec> rho_old;
    std::shared_ptr<vec> alpha;
    std::shared_ptr<vec> beta;
    std::shared_ptr<vec> one_op;
    std::shared_ptr<vec> neg_one_op;
    int max_it;
    double tol;
    bool nsp;
    std::shared_ptr<overlapping_vector> nullsp;
};
