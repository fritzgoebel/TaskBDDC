#include <ginkgo/ginkgo.hpp>
#include <omp.h>

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
                        for (auto entry : local_data[i].nonzeros) {
                            data.nonzeros.emplace_back(global_to_local[entry.row], global_to_local[entry.column], entry.value);
                        }
                        gko::matrix_data<double, int> R_data(gko::dim<2>{global_to_local.size(), size_[1]});
                        gko::matrix_data<double, int> RIT_data(gko::dim<2>{size_[1], global_to_local.size()});
                        gko::matrix_data<double, int> RGT_data(gko::dim<2>{size_[1], global_to_local.size()});
                        for (int j = 0; j < inner_idxs[i].size(); j++) {
                            R_data.nonzeros.emplace_back(j, local_to_global[j], 1.0);
                            RIT_data.nonzeros.emplace_back(local_to_global[j], j, 1.0);
                        }
                        for (int j = 0; j < boundary_idxs[i].size(); j++) {
                            R_data.nonzeros.emplace_back(j + inner_idxs[i].size(), local_to_global[j + inner_idxs[i].size()], 1.0);
                            RGT_data.nonzeros.emplace_back(local_to_global[j + inner_idxs[i].size()], j + inner_idxs[i].size(), 1.0);
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


    std::vector<std::shared_ptr<mtx>> local_mtxs_;
    std::vector<std::shared_ptr<mtx>> R_;
    std::vector<std::shared_ptr<mtx>> RIT_;
    std::vector<std::shared_ptr<mtx>> RGT_;
    std::vector<std::shared_ptr<vec>> buf1;
    std::vector<std::shared_ptr<vec>> buf2;
    std::shared_ptr<vec> one;
    std::shared_ptr<vec> neg_one;
    gko::dim<2> size_;
};
