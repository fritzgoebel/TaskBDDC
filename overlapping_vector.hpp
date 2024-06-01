#include "ginkgo/core/base/executor.hpp"
#include "ginkgo/core/base/types.hpp"
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <map>
#include <vector>

template <typename IndexType>
inline auto find_part(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition,
    IndexType idx)
{
    auto range_bounds = partition->get_range_bounds();
    auto range_parts = partition->get_part_ids();
    auto num_ranges = partition->get_num_ranges();

    auto it =
        std::upper_bound(range_bounds + 1, range_bounds + num_ranges + 1, idx);
    auto range = std::distance(range_bounds + 1, it);
    return range_parts[range];
}

struct overlapping_vector {
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;
    using part = gko::experimental::distributed::Partition<int, int>;
    using mat_data = gko::matrix_data<double, int>;

    overlapping_vector() = default;

    overlapping_vector(std::vector<std::vector<int>>& inner_idxs, std::vector<std::vector<int>>& bndry_idxs, std::shared_ptr<const part> partition, int size)
        : size_{size, 1}, inner_idxs_{inner_idxs}, bndry_idxs_{bndry_idxs}
    {
        int N = inner_idxs.size();
        auto part_sizes = partition->get_part_sizes();
        int n_inner = 0;
        int cnt = 0;
        for (auto i = 0; i < N; i++) {
            n_inner += inner_idxs[i].size();
            for (auto idx : bndry_idxs[i]) {
                if (find_part(partition, idx) == i) {
                    local_to_global_bndry[idx] = cnt;
                    cnt++;
                }
            }    
        }
        global_bndry = vec::create(gko::ReferenceExecutor::create(), gko::dim<2>{size - n_inner, 1});
        one = gko::initialize<vec>({1.0}, gko::ReferenceExecutor::create());
        data.resize(N);
        inner_data.resize(N);
        bndry_data.resize(N);
        owning_bndry_idxs_.resize(N);
        inner_results.resize(N);
        bndry_results.resize(N);
        R.resize(N);
        RT.resize(N);
        for (gko::size_type i = 0; i < N; i++) {
//#pragma omp task 
            {
                auto exec = gko::ReferenceExecutor::create();
                gko::size_type local_size = inner_idxs[i].size() + bndry_idxs[i].size();
                gko::size_type local_inner = inner_idxs[i].size();
                auto local_vec = gko::share(vec::create(exec, gko::dim<2>{local_size, 1}));
                auto inner_vec = gko::share(local_vec->create_submatrix(gko::span{0, local_inner}, gko::span{0, 1}));
                auto bndry_vec = gko::share(local_vec->create_submatrix(gko::span{local_inner, local_size}, gko::span{0, 1}));
                data[i] = local_vec;
                inner_data[i] = inner_vec;
                bndry_data[i] = bndry_vec;
                inner_results[i] = gko::share(vec::create(exec, gko::dim<2>{1, 1}));
                bndry_results[i] = gko::share(vec::create(exec, gko::dim<2>{1, 1}));
                mat_data R_data(gko::dim<2>{bndry_idxs[i].size(), cnt});
                mat_data RT_data(gko::dim<2>{cnt, bndry_idxs[i].size()});
                bndry_idxs_[i].resize(bndry_idxs[i].size());
                for (auto j = 0; j < bndry_idxs[i].size(); j++) {
                    auto idx = local_to_global_bndry[bndry_idxs[i][j]];
                    R_data.nonzeros.emplace_back(j, idx, 1.0);
                    RT_data.nonzeros.emplace_back(idx, j, 1.0);
                }
                R_data.sort_row_major();
                auto restriction = gko::share(mtx::create(exec));
                restriction->read(R_data);
                RT_data.sort_row_major();
                auto prolongation = gko::share(mtx::create(exec));
                prolongation->read(RT_data);
                R[i] = restriction;
                RT[i] = prolongation;
                int owning_size = part_sizes[i] - local_inner;
                owning_bndry_idxs_[i].resize(2 * owning_size);
                size_t cnt = 0;
                for (size_t j = 0; j < bndry_idxs[i].size(); j++) {
                    if (find_part(partition, bndry_idxs[i][j]) == i) {
                        owning_bndry_idxs_[i][2 * cnt] = j;
                        owning_bndry_idxs_[i][2 * cnt + 1] = bndry_idxs[i][j];
                        cnt++;
                    }
                }
            }
        }
//#pragma omp taskwait
    }

    void dot(std::shared_ptr<overlapping_vector> other, std::shared_ptr<vec> result)
    {
        result->at(0,0) = 0.0;
        for (int i = 0; i < data.size(); i++) {
#pragma omp task shared(result) depend (in: other->inner_data[i], this->inner_data[i]) //depend (out: result)
            {
                inner_data[i]->compute_dot(other->inner_data[i], inner_results[i]);
#pragma omp atomic
                result->at(0,0) += inner_results[i]->at(0, 0);
            }
#pragma omp task shared(result) depend (in: other->bndry_data[i], this->bndry_data[i]) //depend (out: result)
            {
                double local_res = 0.0;
                for (size_t j = 0; j < owning_bndry_idxs_[i].size() / 2; j++) {
                    auto idx = owning_bndry_idxs_[i][2 * j];
                    local_res += bndry_data[i]->at(idx, 0) * other->bndry_data[i]->at(idx, 0);
                }
#pragma omp atomic
                result->at(0,0) += local_res;//bndry_results[i]->at(0, 0);
            }
        }
/* #pragma omp task depend (in: result) */
/*         { */
/*             int i = 0; */
/*         } */
#pragma omp taskwait
    }

    void scale(std::shared_ptr<vec> alpha)
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: alpha, this->inner_data[i]) depend (out: this->inner_data[i])
            {
                inner_data[i]->scale(alpha);
            }
#pragma omp task depend (in: alpha, this->bndry_data[i]) depend (out: this->bndry_data[i])
            {
                bndry_data[i]->scale(alpha);
            }
        }
    }

    void add_scaled(std::shared_ptr<vec> alpha, std::shared_ptr<overlapping_vector> other)
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: other->inner_data[i], alpha, this->inner_data[i]) depend (out: this->inner_data[i])
            {
                inner_data[i]->add_scaled(alpha, other->inner_data[i]);
            }
#pragma omp task depend (in: other->bndry_data[i], alpha, this->bndry_data[i]) depend (out: this->bndry_data[i])
            {
                bndry_data[i]->add_scaled(alpha, other->bndry_data[i]);
            }
        }
    }
    
    void sub_scaled(std::shared_ptr<vec> alpha, std::shared_ptr<overlapping_vector> other)
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: other->inner_data[i], alpha, this->inner_data[i]) depend (out: this->inner_data[i])
            {
                inner_data[i]->sub_scaled(alpha, other->inner_data[i]);
            }
#pragma omp task depend (in: other->bndry_data[i], alpha, this->bndry_data[i]) depend (out: this->bndry_data[i])
            {
                bndry_data[i]->sub_scaled(alpha, other->bndry_data[i]);
            }
        }
    }


    void make_consistent()
    {
        bool filled = false;
#pragma omp task shared(this->global_bndry) depend(in: this->global_bndry) depend(out: filled)
        global_bndry->fill(0.0);
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: this->bndry_data[i], filled) depend(out: this->global_bndry) 
            {
                for (int j = 0; j < bndry_idxs_[i].size(); j++) {
#pragma omp atomic
                    global_bndry->at(local_to_global_bndry[bndry_idxs_[i][j]], 0) += bndry_data[i]->at(j, 0);
                }
/* #pragma omp critical */
/*                 { */
/*                     RT[i]->apply(one, bndry_data[i], one, global_bndry); */
/*                 } */
            }
        }
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: this->global_bndry) depend (out: this->bndry_data[i])
            {
                R[i]->apply(global_bndry, bndry_data[i]);
            }
        }
    }

    void fill(double value)
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend(out: this->inner_data[i], this->bndry_data[i])
            {
                data[i]->fill(value);
            }
        }
    }

    void restrict(std::shared_ptr<vec> other) 
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: other) depend (out: this->inner_data[i], this->bndry_data[i])
            {
                for (int j = 0; j < inner_idxs_[i].size(); j++) {
                    inner_data[i]->at(j, 0) = other->at(inner_idxs_[i][j], 0);
                }
                for (int j = 0; j < bndry_idxs_[i].size(); j++) {
                    bndry_data[i]->at(j, 0) = other->at(bndry_idxs_[i][j], 0);
                }
            }
        }
    }

    void prolongate(std::shared_ptr<vec> other) 
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: this->inner_data[i], this->bndry_data[i]) depend (out: other)
            {
                for (int j = 0; j < inner_idxs_[i].size(); j++) {
                    other->at(inner_idxs_[i][j], 0) = inner_data[i]->at(j, 0);
                }
                for (int j = 0; j < owning_bndry_idxs_[i].size() / 2; j++) {
                    other->at(owning_bndry_idxs_[i][2 * j + 1], 0) = bndry_data[i]->at(owning_bndry_idxs_[i][2 * j], 0);
                }
            }
        }
    }

    gko::dim<2> get_size() const { return size_; }

    int get_inner_size(int i) const { return inner_data[i]->get_size()[0]; }

    int get_bndry_size(int i) const { return bndry_data[i]->get_size()[0]; }

    int get_owning_bndry_size(int i) const { return owning_bndry_idxs_[i].size() / 2; }

    int get_num_parts() const { return data.size(); }

    std::shared_ptr<overlapping_vector> clone() const
    {
        std::shared_ptr<overlapping_vector> ret;
        overlapping_vector other;
        ret = std::make_shared<overlapping_vector>(other);
        ret->size_ = size_;
        auto N = data.size();
        ret->data.resize(N);
        ret->inner_data.resize(N);
        ret->bndry_data.resize(N);
        ret->owning_bndry_idxs_.resize(N);
        ret->bndry_idxs_.resize(N);
        ret->inner_idxs_.resize(N);
        ret->inner_results.resize(N);
        ret->bndry_results.resize(N);
        ret->R.resize(N);
        ret->RT.resize(N);
        ret->one = gko::share(one->clone());
        ret->global_bndry = gko::share(global_bndry->clone());
        ret->local_to_global_bndry = local_to_global_bndry;
        for (int i = 0; i < N; i++) {
#pragma omp task depend (in: this->inner_data[i], this->bndry_data[i]) depend (out: ret->inner_data[i], ret->bndry_data[i])
            {
                ret->data[i] = gko::share(gko::clone(data[i]));
                ret->inner_data[i] = gko::share(ret->data[i]->create_submatrix(gko::span{0, inner_data[i]->get_size()[0]}, gko::span{0, 1}));
                ret->bndry_data[i] = gko::share(ret->data[i]->create_submatrix(gko::span{inner_data[i]->get_size()[0], data[i]->get_size()[0]}, gko::span{0, 1}));
                ret->inner_results[i] = gko::share(vec::create(gko::ReferenceExecutor::create(), gko::dim<2>{1, 1}));
                ret->bndry_results[i] = gko::share(vec::create(gko::ReferenceExecutor::create(), gko::dim<2>{1, 1}));
                ret->R[i] = gko::share(gko::clone(R[i]));
                ret->RT[i] = gko::share(gko::clone(RT[i]));
                ret->inner_idxs_[i] = inner_idxs_[i];
                ret->bndry_idxs_[i] = bndry_idxs_[i];
                ret->owning_bndry_idxs_[i] = owning_bndry_idxs_[i];
            }
        }
//#pragma omp taskwait
        return ret;
    }

    void copy_from(std::shared_ptr<overlapping_vector> other)
    {
        for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: other->inner_data[i]) depend (out: this->inner_data[i])
            {
                inner_data[i]->copy_from(other->inner_data[i]);
            }
#pragma omp task depend (in: other->bndry_data[i]) depend (out: this->bndry_data[i])
            {
                bndry_data[i]->copy_from(other->bndry_data[i]);
            }
        }
    }



    gko::dim<2> size_;
    std::vector<std::shared_ptr<vec>> data;
    std::vector<std::shared_ptr<vec>> inner_data;
    std::vector<std::shared_ptr<vec>> bndry_data;
    std::vector<std::shared_ptr<vec>> inner_results;
    std::vector<std::shared_ptr<vec>> bndry_results;
    std::shared_ptr<vec> one;
    std::shared_ptr<vec> global_bndry;
    std::map<int, int> local_to_global_bndry;
    std::vector<std::shared_ptr<mtx>> R;
    std::vector<std::shared_ptr<mtx>> RT;
    std::vector<std::vector<int>> inner_idxs_;
    std::vector<std::vector<int>> bndry_idxs_;
    std::vector<std::vector<int>> owning_bndry_idxs_;
};
