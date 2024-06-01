#include "cg.hpp"
#include "ginkgo/core/base/mtx_io.hpp"
#include "ginkgo/core/distributed/partition.hpp"
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <omp.h>

#include "omptasktool.h"

using mat_data = gko::matrix_data<double, int>;
using mtx = gko::matrix::Csr<double, int>;
using part = gko::experimental::distributed::Partition<int, int>;
using vec = gko::matrix::Dense<double>;

std::vector<std::vector<int>> read_idxs(const char* dir, int n) {
    std::vector<std::vector<int>> idxs(n);
    for (int i = 0; i < n; i++) {
//#pragma omp task shared(idxs)
        {
            std::string filename = std::string(dir) + "/local_idxs_" + std::to_string(i) + ".txt";
            std::ifstream in{filename};
            int idx;
            while (in >> idx) {
                idxs[i].push_back(idx - 1);
            }
        }
    }
//#pragma omp taskwait
    return idxs;
}

std::vector<std::vector<int>> read_inner(const char* dir, int n) {
    std::vector<std::vector<int>> idxs(n);
    for (int i = 0; i < n; i++) {
//#pragma omp task shared(idxs)
        {
            std::string filename = std::string(dir) + "/inner_" + std::to_string(i) + ".txt";
            std::ifstream in{filename};
            int idx;
            while (in >> idx) {
                idxs[i].push_back(idx - 1);
            }
        }
    }
//#pragma omp taskwait
    return idxs;
}

std::vector<std::vector<int>> read_bndry(const char* dir, int n) {
    std::vector<std::vector<int>> idxs(n);
    for (int i = 0; i < n; i++) {
//#pragma omp task shared(idxs)
        {
            std::string filename = std::string(dir) + "/bndry_" + std::to_string(i) + ".txt";
            std::ifstream in{filename};
            int idx;
            while (in >> idx) {
                idxs[i].push_back(idx - 1);
            }
        }
    }
//#pragma omp taskwait
    return idxs;
}

std::vector<mat_data> read_matrices(const char* dir, int n) {
    std::vector<mat_data> matrices(n);
    for (int i = 0; i < n; i++) {
//#pragma omp task shared(matrices)
        {
            std::string filename = std::string(dir) + "/local_" + std::to_string(i) + ".gko";
            std::ifstream in{filename};
            auto data = gko::read_binary_raw(in);
            matrices[i] = data;
        }
    }
//#pragma omp taskwait
    return matrices;
}

std::shared_ptr<vec> read_rhs(const char* dir) {
    std::string filename = std::string(dir) + "/rhs.gko";
    std::ifstream in{filename};
    auto data = gko::share(gko::read_binary<vec>(in, gko::ReferenceExecutor::create()));
    return data;
}

int main(const int argc, const char *argv[]) {
/* #pragma omp parallel */
/*     { */
/* #pragma omp single */
/*         { */
            auto exec = gko::ReferenceExecutor::create();

            const auto dir = argv[1];
            const auto n = std::stoi(argv[2]);
            const auto max_it = std::stoi(argv[3]);
            const auto tol = std::stod(argv[4]);
            const auto rep = std::stoi(argv[5]);

            std::cout << "Setting up problem " << dir << " for " << n << " subdomains" << std::endl;
            std::cout << "Reading data..." << std::endl;
            auto idxs = read_idxs(dir, n);
            auto inner = read_inner(dir, n);
            auto bndry = read_bndry(dir, n);
            auto local_data = read_matrices(dir, n);
            auto rhs = read_rhs(dir);
            auto sol = gko::share(gko::clone(rhs));
/* #pragma omp taskwait */
            std::cout << "done" << std::endl;
            std::cout << "Setting up partition..." << std::endl;

            auto n_rows = rhs->get_size()[0];
            gko::array<int> map{exec, n_rows};
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < idxs[i].size(); j++) {
                    map.get_data()[idxs[i][j]] = i;
                }
            }

            auto partition = gko::share(part::build_from_mapping(exec, map, n));
            std::cout << "done" << std::endl;
            std::cout << "Setting up matrix..." << std::endl;
            auto A = std::make_shared<block_matrix>(block_matrix(local_data, inner, bndry));
/* #pragma omp taskwait */
            std::cout << "done" << std::endl;
            std::cout << "Setting up vectors..." << std::endl;
            auto b = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, n_rows));
            auto x = std::make_shared<overlapping_vector>(overlapping_vector(A->inner_idxs, A->bndry_idxs, partition, n_rows));
            x->fill(0.0);
            b->restrict(rhs);
/* #pragma omp taskwait */
            std::cout << "done" << std::endl;
            std::cout << "Setting up solver..." << std::endl;
            std::shared_ptr<cg> solver;
            solver = std::make_shared<cg>(cg(A, max_it, tol, x));
#pragma omp parallel
            {
#pragma omp single
                {
                    std::cout << "done" << std::endl;
                    std::cout << "Solving..." << std::endl;
                    solver->apply(b, x);
#pragma omp taskwait
                }
            }
/* #pragma omp taskwait */
            std::cout << "done" << std::endl;
            std::cout << "Running " << rep << " benchmark runs..." << std::endl;
            double duration = 0.0;
            for (int i = 0; i < rep; i++) {
                std::cout << "Run " << i << std::endl;
                x->fill(0.0);
/* #pragma omp taskwait */
                auto start = omp_get_wtime();
                solver->apply(b, x);
/* #pragma omp taskwait */
                auto end = omp_get_wtime();
                duration += end - start;
            }
            std::cout << "Average duration: " << duration / rep << "s" << std::endl;
            std::cout << "done" << std::endl;
            std::cout << "Prolongating solution..." << std::endl;
            x->prolongate(sol);
/* #pragma omp taskwait */
            std::cout << "done" << std::endl;
            std::cout << "Writing solution..." << std::endl;

            std::ofstream out{"sol.mtx"};
            gko::write(out, sol);
            std::cout << "done" << std::endl;
        /* } */
    /* } */
}
