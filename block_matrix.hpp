#include <algorithm>
#include <ginkgo/ginkgo.hpp>
#include <iterator>
#include <memory>
#include <omp.h>

#include "overlapping_vector.hpp"

#include <set>

struct block_matrix {
    using mtx = gko::matrix::Csr<double, int>;
    using vec = gko::matrix::Dense<double>;

    std::vector<std::pair<std::vector<int>, std::vector<int>>> intersect(std::vector<std::pair<std::vector<int>, std::vector<int>>> pairs, int subdomain, std::vector<int> dofs)
    {
        std::vector<std::pair<std::vector<int>, std::vector<int>>> result(2 * pairs.size() + 1);
//#pragma omp taskloop shared(pairs, dofs, result)
        for (size_t i = 0; i < pairs.size(); i++) {
            result[2 * i] = std::make_pair(pairs[i].first, std::vector<int>());
            std::set_difference(
                    pairs[i].second.begin(), pairs[i].second.end(), 
                    dofs.begin(), dofs.end(), std::back_inserter(result[2 * i].second)); // a - b
            if (result[2 * i].second.size() <= result[2 * i].first.size()) {
                for (auto idx : result[2 * i].second) {
                    result.emplace_back(result[2 * i].first, std::vector<int>{idx});
                }
                result[2 * i].second.clear();
            }
            result[2 * i + 1] = std::make_pair(pairs[i].first, std::vector<int>());
            result[2 * i + 1].first.emplace_back(subdomain);
            std::set_intersection(
                    pairs[i].second.begin(), pairs[i].second.end(), 
                    dofs.begin(), dofs.end(), std::back_inserter(result[2 * i + 1].second)); // a & b
            if (result[2 * i + 1].second.size() <= result[2 * i + 1].first.size()) {
                for (auto idx : result[2 * i + 1].second) {
                    result.emplace_back(result[2 * i + 1].first, std::vector<int>{idx});
                }
                result[2 * i + 1].second.clear();
            }
        }
        result[2 * pairs.size()].first = {subdomain};
        result[2 * pairs.size()].second = dofs;
        for (size_t i = 0; i < pairs.size(); i++) {
            result[2 * pairs.size()].second.erase(std::remove_if(result[2 * pairs.size()].second.begin(), result[2 * pairs.size()].second.end(), [&](auto a) {
                return std::find(pairs[i].second.begin(), pairs[i].second.end(), a) != pairs[i].second.end();
            }), result[2 * pairs.size()].second.end());
        }
        result.erase(std::remove_if(result.begin(), result.end(), [](auto a) {
            return a.second.size() == 0;
        }), result.end());
        return result;
    }

    block_matrix(std::vector<gko::matrix_data<double, int>> local_data, std::vector<std::vector<int>> inner_idxs, 
                 std::vector<std::vector<int>> boundary_idxs, std::shared_ptr<gko::Executor> exec) : inner_idxs{inner_idxs}
    {
        std::cout << "Creating block matrix" << std::endl;
        size_ = local_data[0].size;
        auto N = local_data.size();
        local_mtxs_.resize(N);
        inner_mtxs_.resize(N);
        bndry_mtxs_.resize(N);
        R_.resize(N);
        RIT_.resize(N);
        RGT_.resize(N);
        buf1.resize(N);
        buf2.resize(N);
        one = gko::initialize<vec>({1.0}, exec);
        neg_one = gko::initialize<vec>({-1.0}, exec);
        
        std::vector<std::vector<int>> bndry_to_subdomains(size_[0], std::vector<int>());
        std::set<std::vector<int>> unique_rank_sets;
        bndry_idxs.resize(N);
        // Set up mapping from dofs to sharing subdomains
        for (size_t subdomain = 0; subdomain < N; subdomain++) {
//#pragma omp task shared (bndry_to_subdomains, unique_rank_sets, boundary_idxs)
            {
                std::sort(boundary_idxs[subdomain].begin(), boundary_idxs[subdomain].end());
                auto local_bndry_idxs = boundary_idxs[subdomain];
                for (auto bndry_idx : local_bndry_idxs) {
//#pragma omp critical
                    bndry_to_subdomains[bndry_idx].emplace_back(subdomain);
                }
            }
        }
//#pragma omp taskwait
        for (size_t i = 0; i < size_[0]; i++) {
            if (bndry_to_subdomains[i].size() > 0) {
                std::sort(bndry_to_subdomains[i].begin(), bndry_to_subdomains[i].end());
                unique_rank_sets.insert(bndry_to_subdomains[i]);
            }
        }

        std::cout << "Unique rank sets: " << unique_rank_sets.size() << std::endl;
        std::vector<std::vector<std::vector<int>>> sets_to_interfaces(unique_rank_sets.size());
        std::vector<std::vector<int>> unique_rank_vecs(unique_rank_sets.begin(), unique_rank_sets.end());
        int idx = 0;
        // Set up interfaces between subdomains
//#pragma omp taskloop shared (sets_to_interfaces, idx, unique_rank_vecs)
        for (int i = 0; i < unique_rank_vecs.size(); i++) {
            auto set = unique_rank_vecs[i];
            std::vector<int> dofs;
            auto local_bndry_idxs = boundary_idxs[set[0]];
            for (size_t i = 0; i < local_bndry_idxs.size(); i++) {
                auto dof = local_bndry_idxs[i];
                if (bndry_to_subdomains[dof].size() == set.size()) {
                    if (bndry_to_subdomains[dof] == set) {
                        dofs.emplace_back(dof);
                    }
                }
            }
            if (dofs.size() <= set.size()) {
                for (auto dof : dofs) {
                    sets_to_interfaces[i].emplace_back(std::vector<int>{dof});
                }
            } else {
                sets_to_interfaces[i].emplace_back(dofs);
            }
        }
        for (size_t i = 0; i < sets_to_interfaces.size(); i++) {
            for (size_t j = 0; j < sets_to_interfaces[i].size(); j++) {
                interfaces.emplace_back(unique_rank_vecs[i], sets_to_interfaces[i][j]);
            }
        }
        /* interfaces.emplace_back(std::vector<int>{0}, boundary_idxs[0]); */
        /* for (int subdomain = 1; subdomain < N; subdomain++) { */
        /*     std::cout << interfaces.size() << " " << boundary_idxs[subdomain].size() << std::endl; */
        /*     interfaces = intersect(interfaces, subdomain, boundary_idxs[subdomain]); */
        /*     std::cout << "--------------------------------------------------------------" << std::endl; */
        /*     for (int i = 0; i < interfaces.size(); i++) { */
        /*         std::cout << "Interface " << i << ": "; */
        /*         for (auto idx : interfaces[i].first) { */
        /*             std::cout << idx << " "; */
        /*         } */
        /*         std::cout << " | "; */
        /*         for (auto idx : interfaces[i].second) { */
        /*             std::cout << idx << " "; */
        /*         } */
        /*         std::cout << std::endl; */
        /*     } */
        /* } */
        std::cout << "Interfaces: " << interfaces.size() << std::endl;
        // sort interfaces by dof count
        std::sort(interfaces.begin(), interfaces.end(), [](auto a, auto b) {
            return a.second.size() > b.second.size();
        });
        local_interfaces.resize(N);
        for (size_t i = 0; i < N; i++) {
//#pragma omp task shared(local_interfaces, bndry_idxs)
            {
                for (size_t j = 0; j < interfaces.size(); j++) {
                    auto interf = interfaces[j];
                    if (std::find(interf.first.begin(), interf.first.end(), i) != interf.first.end()) {
                        for (auto idx : interf.second) {
                            bndry_idxs[i].emplace_back(idx);
                        }
                        local_interfaces[i].emplace_back(j);
                    }
                }
            }
        }
//#pragma omp taskwait
        std::cout << "Identified local interfaces" << std::endl;
        for (int i = 0; i < N; i++) {
//#pragma omp task shared(inner_idxs, bndry_idxs, local_mtxs_, inner_mtxs_, bndry_mtxs_, R_, RIT_, RGT_, buf1, buf2)
            {
                std::map<int, int> global_to_local{};
                std::vector<int> local_to_global;
                int idx = 0;
                for (auto inner_idx : inner_idxs[i]) {
                    global_to_local[inner_idx] = idx;
                    local_to_global.push_back(inner_idx);
                    idx++;
                }
                for (auto boundary_idx : bndry_idxs[i]) {
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
                    R_data.nonzeros.emplace_back(j, inner_idxs[i][j], 1.0);
                    RIT_data.nonzeros.emplace_back(inner_idxs[i][j], j, 1.0);
                }
                for (int j = 0; j < bndry_idxs[i].size(); j++) {
                    R_data.nonzeros.emplace_back(j + inner_idxs[i].size(), bndry_idxs[i][j], 1.0);
                    RGT_data.nonzeros.emplace_back(bndry_idxs[i][j], j + inner_idxs[i].size(), 1.0);
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
                inner_mtxs_[i] = gko::share(local_mtxs_[i]->create_submatrix(gko::span{0, inner_idxs[i].size()}, gko::span{0, local_mtxs_[i]->get_size()[1]}));
                bndry_mtxs_[i] = gko::share(local_mtxs_[i]->create_submatrix(gko::span{inner_idxs[i].size(), local_mtxs_[i]->get_size()[1]}, gko::span{0, local_mtxs_[i]->get_size()[1]}));
            }
        }
//#pragma omp taskwait
        std::cout << "Block matrix created" << std::endl;
    }

    void apply_bndry(std::shared_ptr<overlapping_vector> x, std::shared_ptr<overlapping_vector> y)
    {
        for (int i = 0; i < local_mtxs_.size(); i++) {
#pragma omp task depend (in: x->inner_data[i], x->bndry_data[i], y->bndry_data[i]) depend(out: y->bndry_data[i])
            bndry_mtxs_[i]->apply(x->data[i], y->bndry_data[i]);
        }
        y->make_consistent();
    }

    void apply(std::shared_ptr<overlapping_vector> x, std::shared_ptr<overlapping_vector> y)
    {
        apply_bndry(x, y);
        for (int i = 0; i < local_mtxs_.size(); i++) {
#pragma omp task depend (in: x->inner_data[i], x->bndry_data[i], y->inner_data[i]) depend(out: y->inner_data[i])
            inner_mtxs_[i]->apply(x->data[i], y->inner_data[i]);
        }
    }

    void apply(std::shared_ptr<vec> alpha, std::shared_ptr<overlapping_vector> x, std::shared_ptr<vec> beta, std::shared_ptr<overlapping_vector> y)
    {
        auto y_clone = y->clone();
        this->apply(x, y_clone);
        y->scale(beta);
        y->add_scaled(alpha, y_clone);
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
    std::vector<std::vector<int>> inner_idxs;
    std::vector<std::vector<int>> bndry_idxs;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> interfaces;
    std::vector<std::vector<int>> local_interfaces;
};
