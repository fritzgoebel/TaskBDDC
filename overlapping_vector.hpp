#include "ginkgo/core/base/executor.hpp"
#include <ginkgo/ginkgo.hpp>
#include <memory>

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

    overlapping_vector(std::vector<std::vector<int>>& inner_idxs, std::vector<std::vector<int>>& bndry_idxs, std::shared_ptr<const part> partition, int size)
        : size_{size, 1}
    {
#pragma omp parallel
        {
#pragma omp single
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
                owning_bndry_data.resize(N);
                inner_results.resize(N);
                bndry_results.resize(N);
                R.resize(N);
                RT.resize(N);
                for (int i = 0; i < N; i++) {
#pragma omp task 
                    {
                        auto exec = gko::ReferenceExecutor::create();
                        int local_size = inner_idxs[i].size() + bndry_idxs[i].size();
                        int local_inner = inner_idxs[i].size();
                        auto local_vec = gko::share(vec::create(exec, gko::dim<2>{local_size, 1}));
                        auto inner_vec = gko::share(local_vec->create_submatrix(gko::span{0, local_inner}, gko::span{0, 1}));
                        auto bndry_vec = gko::share(local_vec->create_submatrix(gko::span{local_inner, local_size}, gko::span{0, 1}));
                        data[i] = local_vec;
                        inner_data[i] = inner_vec;
                        bndry_data[i] = bndry_vec;
                        inner_results[i] = gko::share(vec::create(exec, gko::dim<2>{1, 1}));
                        bndry_results[i] = gko::share(vec::create(exec, gko::dim<2>{1, 1}));
                        int owning_size = part_sizes[i] - local_inner;
                        auto owning_bndry_vec = gko::share(local_vec->create_submatrix(gko::span{local_inner, local_inner + owning_size}, gko::span{0, 1}));
                        owning_bndry_data[i] = owning_bndry_vec;
                        mat_data R_data(gko::dim<2>{bndry_idxs[i].size(), cnt});
                        mat_data RT_data(gko::dim<2>{cnt, bndry_idxs[i].size()});
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
                    }
                }
            }
#pragma omp taskwait
        }
    }

    void dot(std::shared_ptr<overlapping_vector> other, std::shared_ptr<vec> result)
    {
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: other->inner_data[i], this->inner_data[i])
                    {
                        inner_data[i]->compute_dot(other->inner_data[i], inner_results[i]);
#pragma omp atomic
                        result->at(0,0) += inner_results[i]->at(0, 0);
                    }
#pragma omp task depend (in: other->bndry_data[i], this->bndry_data[i])
                    {
                        owning_bndry_data[i]->compute_dot(other->owning_bndry_data[i], bndry_results[i]);
#pragma omp atomic
                        result->at(0,0) += bndry_results[i]->at(0, 0);
                    }
                }
#pragma omp taskwait
            }
        }
    }

    void make_consistent()
    {
#pragma omp parallel
        {
#pragma omp single
            {
                global_bndry->fill(0.0);
                for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (in: this->bndry_data[i])
                    {
#pragma omp critical
                        {
                            RT[i]->apply(one, bndry_data[i], one, global_bndry);
                        }
                    }
                }
#pragma omp taskwait
                for (int i = 0; i < data.size(); i++) {
#pragma omp task depend (out: this->bndry_data[i])
                    {
                        R[i]->apply(global_bndry, bndry_data[i]);
                    }
                }
            }
        }
    }

    void fill(double value)
    {
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < data.size(); i++) {
#pragma omp task
                    {
                        data[i]->fill(value);
                    }
                }
            }
#pragma omp taskwait
        }
    }

    gko::dim<2> get_size() const { return size_; }

    int get_inner_size(int i) const { return inner_data[i]->get_size()[0]; }

    int get_bndry_size(int i) const { return bndry_data[i]->get_size()[0]; }

    int get_owning_bndry_size(int i) const { return owning_bndry_data[i]->get_size()[0]; }

    int get_num_parts() const { return data.size(); }

    gko::dim<2> size_;
    std::vector<std::shared_ptr<vec>> data;
    std::vector<std::shared_ptr<vec>> inner_data;
    std::vector<std::shared_ptr<vec>> bndry_data;
    std::vector<std::shared_ptr<vec>> owning_bndry_data;
    std::vector<std::shared_ptr<vec>> inner_results;
    std::vector<std::shared_ptr<vec>> bndry_results;
    std::shared_ptr<vec> one;
    std::shared_ptr<vec> global_bndry;
    std::map<int, int> local_to_global_bndry;
    std::vector<std::shared_ptr<mtx>> R;
    std::vector<std::shared_ptr<mtx>> RT;
};
