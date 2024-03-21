#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <omp.h>

#include "overlapping_vector.hpp"

struct block_matrix {
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;

    block_matrix(std::vector<gko::matrix_data<double, int>> local_data, std::vector<std::vector<int>> inner_idxs, std::vector<std::vector<int>> boundary_idxs)
    {
#pragma omp parallel
        {
#pragma omp single
            {
                size_ = local_data[0].size;
                local_mtxs_.resize(local_data.size());
                inner_mtxs_.resize(local_data.size());
                bndry_mtxs_.resize(local_data.size());
                R_.resize(local_data.size());
                RIT_.resize(local_data.size());
                RGT_.resize(local_data.size());
                buf1.resize(local_data.size());
                buf2.resize(local_data.size());
                one = gko::initialize<vec>({1.0}, gko::ReferenceExecutor::create());
                neg_one = gko::initialize<vec>({-1.0}, gko::ReferenceExecutor::create());
                for (int i = 0; i < local_data.size(); i++) {
#pragma omp task
                    {
                        auto exec = gko::ReferenceExecutor::create();
                        std::map<int, int> global_to_local;
                        std::vector<int> local_to_global;
                        int idx = 0;
                        for (auto inner_idx : inner_idxs[i]) {
                            global_to_local[inner_idx] = idx;
                            local_to_global.push_back(inner_idx);
                            idx++;
                        }
                        for (auto boundary_idx : boundary_idxs[i]) {
                            global_to_local[boundary_idx] = idx;
                            local_to_global.push_back(boundary_idx);
                            idx++;
                        }
                        gko::matrix_data<double, int> data(gko::dim<2>{global_to_local.size(), global_to_local.size()});
                        gko::matrix_data<double, int> inner_data(gko::dim<2>{inner_idxs[i].size(), global_to_local.size()});
                        gko::matrix_data<double, int> bndry_data(gko::dim<2>{boundary_idxs[i].size(), global_to_local.size()});
                        for (auto entry : local_data[i].nonzeros) {
                            data.nonzeros.emplace_back(global_to_local[entry.row], global_to_local[entry.column], entry.value);
                            if (global_to_local[entry.row] < inner_idxs[i].size()) {
                                inner_data.nonzeros.emplace_back(global_to_local[entry.row], global_to_local[entry.column], entry.value);
                            } else {
                                bndry_data.nonzeros.emplace_back(global_to_local[entry.row] - inner_idxs.size(), global_to_local[entry.column], entry.value);
                            }
                        }
                        gko::matrix_data<double, int> R_data(gko::dim<2>{global_to_local.size(), size_[1]});
                        gko::matrix_data<double, int> RIT_data(gko::dim<2>{size_[1], global_to_local.size()});
                        gko::matrix_data<double, int> RGT_data(gko::dim<2>{size_[1], global_to_local.size()});
                        for (int j = 0; j < inner_idxs[i].size(); j++) {
                            R_data.nonzeros.emplace_back(j, inner_idxs[i][j], 1.0);
                            RIT_data.nonzeros.emplace_back(inner_idxs[i][j], j, 1.0);
                        }
                        for (int j = 0; j < boundary_idxs[i].size(); j++) {
                            R_data.nonzeros.emplace_back(j + inner_idxs[i].size(), boundary_idxs[i][j], 1.0);
                            RGT_data.nonzeros.emplace_back(boundary_idxs[i][j], j + inner_idxs[i].size(), 1.0);
                        }
                        data.sort_row_major();
                        local_mtxs_[i] = mtx::create(exec);
                        local_mtxs_[i]->read(data);
                        R_data.sort_row_major();
                        R_[i] = mtx::create(exec);
                        R_[i]->read(R_data);
                        RIT_data.sort_row_major();
                        RIT_[i] = mtx::create(exec);
                        RIT_[i]->read(RIT_data);
                        RGT_data.sort_row_major();
                        RGT_[i] = mtx::create(exec);
                        RGT_[i]->read(RGT_data);
                        buf1[i] = vec::create(exec, gko::dim<2>{global_to_local.size(), 1});
                        buf2[i] = vec::create(exec, gko::dim<2>{global_to_local.size(), 1});
                    }
                }
            }
        }
    }

    void apply_local(int i, std::shared_ptr<vec> x, std::shared_ptr<vec> y)
    {
        R_[i]->apply(x, buf1[i]);
        local_mtxs_[i]->apply(buf1[i], buf2[i]);
#pragma omp critical
        {
            RIT_[i]->apply(one, buf2[i], one, y);
            RGT_[i]->apply(one, buf2[i], one, y);
        }
    }

    void apply(std::shared_ptr<vec> x, std::shared_ptr<vec> y)
    {
#pragma omp parallel
        {
#pragma omp single
            {
                y->fill(0.0);
                for (int i = 0; i < local_mtxs_.size(); i++) {
#pragma omp task shared(x, y)
                    apply_local(i, x, y);
                }
#pragma omp taskwait
            }
        }
    }

    void apply(std::shared_ptr<vec> alpha, std::shared_ptr<vec> x, std::shared_ptr<vec> beta, std::shared_ptr<vec> y)
    {
        auto y_clone = gko::share(y->clone());
        this->apply(x, y_clone);
        y->scale(beta);
        y->add_scaled(alpha, y_clone);
    }

    void apply_bndry(std::shared_ptr<overlapping_vector> x, std::shared_ptr<overlapping_vector> y)
    {
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < local_mtxs_.size(); i++) {
#pragma omp task depend (in: x->inner_data[i], x->bndry_data[i]) depend(out: y->bndry_data[i])
                    bndry_mtxs_[i]->apply(x->bndry_data[i], y->bndry_data[i]);
                }
#pragma omp task
                y->make_consistent();
            }
        }
    }

    void apply(std::shared_ptr<overlapping_vector> x, std::shared_ptr<overlapping_vector> y)
    {
#pragma omp parallel
        {
#pragma omp single
            {
                apply_bndry(x, y);
                for (int i = 0; i < local_mtxs_.size(); i++) {
#pragma omp task depend (in: x->inner_data[i], x->bndry_data[i]) depend(out: y->inner_data[i])
                    inner_mtxs_[i]->apply(x->data[i], y->inner_data[i]);
                }
            }
        }
    }


    std::vector<std::shared_ptr<mtx>> local_mtxs_;
    std::vector<std::shared_ptr<mtx>> inner_mtxs_;
    std::vector<std::shared_ptr<mtx>> bndry_mtxs_;
    std::vector<std::shared_ptr<mtx>> R_;
    std::vector<std::shared_ptr<mtx>> RIT_;
    std::vector<std::shared_ptr<mtx>> RGT_;
    std::vector<std::shared_ptr<vec>> buf1;
    std::vector<std::shared_ptr<vec>> buf2;
    std::shared_ptr<vec> one;
    std::shared_ptr<vec> neg_one;
    gko::dim<2> size_;
};
