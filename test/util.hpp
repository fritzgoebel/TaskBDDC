namespace util {
    using mat_data = gko::matrix_data<double, int>;

    block_matrix create_tridiag_matrix(int n_parts, int local_size) {
        std::vector<mat_data> blocks;
        std::vector<std::vector<int>> inner_idxs;
        std::vector<std::vector<int>> boundary_idxs;
        for (int i = 0; i < n_parts; i++) {
            mat_data local_data{gko::dim<2>{n_parts * (local_size - 1) + 1, n_parts * (local_size - 1) + 1}};
            std::vector<int> inner;
            std::vector<int> boundary;
            for (int j = i * (local_size - 1); j < (i + 1) * (local_size - 1) + 1; j++) {
                if (j > i * (local_size - 1)) {
                    local_data.nonzeros.emplace_back(j, j, 1.0);
                    if (j != 1 && j != n_parts * (local_size - 1)) {
                        local_data.nonzeros.emplace_back(j, j - 1, -1.0);
                    }
                }
                if (j < (i + 1) * (local_size - 1)) {
                    local_data.nonzeros.emplace_back(j, j, 1.0);
                    if (j != 0 && j + 1 != n_parts * (local_size - 1)) {
                        local_data.nonzeros.emplace_back(j, j + 1, -1.0);
                    }
                }
                if (j < (i + 1) * (local_size - 1) && j > i * (local_size - 1)) {
                    inner.push_back(j);
                } else if (j == 0 || j == n_parts * (local_size - 1)) {
                    inner.push_back(j);
                } else {
                    boundary.push_back(j);
                }
            }
            local_data.sum_duplicates();
            local_data.sort_row_major();
            blocks.emplace_back(local_data);
            inner_idxs.push_back(inner);
            boundary_idxs.push_back(boundary);
        }
        return block_matrix{blocks, inner_idxs, boundary_idxs};
    }

    void assume_correct_matrix(int n_parts, int local_size, block_matrix& matrix) {
        ASSERT_EQ(matrix.size_[0], n_parts * (local_size - 1) + 1);
        ASSERT_EQ(matrix.size_[1], n_parts * (local_size - 1) + 1);
        {
            auto row_ptrs = matrix.local_mtxs_[0]->get_const_row_ptrs();
            auto col_idxs = matrix.local_mtxs_[0]->get_const_col_idxs();
            auto values = matrix.local_mtxs_[0]->get_const_values();
            ASSERT_EQ(row_ptrs[0], 0);
            ASSERT_EQ(col_idxs[0], 0);
            ASSERT_EQ(values[0], 1.0);
            ASSERT_EQ(row_ptrs[1], 1);
            ASSERT_EQ(col_idxs[1], 1);
            ASSERT_EQ(values[1], 2.0);
            ASSERT_EQ(col_idxs[2], 2);
            ASSERT_EQ(values[2], -1.0);
            for (int j = 2; j < local_size - 1 && j < n_parts * (local_size - 1) - 1; j++) {
                ASSERT_EQ(row_ptrs[j], 3 * (j - 1));
                ASSERT_EQ(col_idxs[3 * (j - 1)], j - 1);
                ASSERT_EQ(values[3 * (j - 1)], -1.0);
                ASSERT_EQ(col_idxs[3 * (j - 1) + 1], j);
                ASSERT_EQ(values[3 * (j - 1) + 1], 2.0);
                ASSERT_EQ(col_idxs[3 * (j - 1) + 2], j + 1);
                ASSERT_EQ(values[3 * (j - 1) + 2], -1.0);
            }
            if (n_parts > 1) {
                ASSERT_EQ(row_ptrs[local_size - 1], 3 * (local_size - 2));
                ASSERT_EQ(col_idxs[3 * (local_size - 2)], local_size - 2);
                ASSERT_EQ(values[3 * (local_size - 2)], -1.0);
                ASSERT_EQ(col_idxs[3 * (local_size - 2) + 1], local_size - 1);
                ASSERT_EQ(values[3 * (local_size - 2) + 1], 1.0);
                ASSERT_EQ(row_ptrs[local_size], 3 * (local_size - 2) + 2);
            }
        }
        for (int i = 1; i < n_parts - 1; i++) {
            auto row_ptrs = matrix.local_mtxs_[i]->get_const_row_ptrs();
            auto col_idxs = matrix.local_mtxs_[i]->get_const_col_idxs();
            auto values = matrix.local_mtxs_[i]->get_const_values();
            ASSERT_EQ(row_ptrs[0], 0);
            ASSERT_EQ(col_idxs[0], 0);
            ASSERT_EQ(values[0], 2.0);
            ASSERT_EQ(col_idxs[1], 1);
            ASSERT_EQ(values[1], -1.0);
            ASSERT_EQ(col_idxs[2], local_size - 2);
            ASSERT_EQ(values[2], -1.0);
            for (int j = 1; j < local_size - 3; j++) {
                ASSERT_EQ(row_ptrs[j], 3 * j);
                ASSERT_EQ(col_idxs[3 * j], j - 1);
                ASSERT_EQ(values[3 * j], -1.0);
                ASSERT_EQ(col_idxs[3 * j + 1], j);
                ASSERT_EQ(values[3 * j + 1], 2.0);
                ASSERT_EQ(col_idxs[3 * j + 2], j + 1);
                ASSERT_EQ(values[3 * j + 2], -1.0);
            }
            ASSERT_EQ(row_ptrs[local_size - 3], 3 * (local_size - 3));
            ASSERT_EQ(col_idxs[3 * (local_size - 3)], local_size - 4);
            ASSERT_EQ(values[3 * (local_size - 3)], -1.0);
            ASSERT_EQ(col_idxs[3 * (local_size - 3) + 1], local_size - 3);
            ASSERT_EQ(values[3 * (local_size - 3) + 1], 2.0);
            ASSERT_EQ(col_idxs[3 * (local_size - 3) + 2], local_size - 1);
            ASSERT_EQ(values[3 * (local_size - 3) + 2], -1.0);
            ASSERT_EQ(row_ptrs[local_size - 2], 3 * (local_size - 2));
            ASSERT_EQ(col_idxs[3 * (local_size - 2)], 0);
            ASSERT_EQ(values[3 * (local_size - 2)], -1.0);
            ASSERT_EQ(col_idxs[3 * (local_size - 2) + 1], local_size - 2);
            ASSERT_EQ(values[3 * (local_size - 2) + 1], 1.0);
            ASSERT_EQ(row_ptrs[local_size - 1], 3 * (local_size - 2) + 2);
            ASSERT_EQ(col_idxs[3 * (local_size - 2) + 2], local_size - 3);
            ASSERT_EQ(values[3 * (local_size - 2) + 2], -1.0);
            ASSERT_EQ(col_idxs[3 * (local_size - 2) + 3], local_size - 1);
            ASSERT_EQ(row_ptrs[local_size], 4 + 3 * (local_size - 2));
        }
        {
            auto row_ptrs = matrix.local_mtxs_[n_parts - 1]->get_const_row_ptrs();
            auto col_idxs = matrix.local_mtxs_[n_parts - 1]->get_const_col_idxs();
            auto values = matrix.local_mtxs_[n_parts - 1]->get_const_values();
            if (n_parts > 1) {
                ASSERT_EQ(row_ptrs[0], 0);
                ASSERT_EQ(col_idxs[0], 0);
                ASSERT_EQ(values[0], 2.0);
                ASSERT_EQ(col_idxs[1], 1);
                ASSERT_EQ(values[1], -1.0);
                ASSERT_EQ(col_idxs[2], local_size - 1);
                ASSERT_EQ(values[2], -1.0);
                for (int j = 1; j < local_size - 3; j++) {
                    ASSERT_EQ(row_ptrs[j], 3 * j);
                    ASSERT_EQ(col_idxs[3 * j], j - 1);
                    ASSERT_EQ(values[3 * j], -1.0);
                    ASSERT_EQ(col_idxs[3 * j + 1], j);
                    ASSERT_EQ(values[3 * j + 1], 2.0);
                    ASSERT_EQ(col_idxs[3 * j + 2], j + 1);
                    ASSERT_EQ(values[3 * j + 2], -1.0);
                }
                ASSERT_EQ(row_ptrs[local_size - 3], 3 * (local_size - 3));
                ASSERT_EQ(col_idxs[3 * (local_size - 3)], local_size - 4);
                ASSERT_EQ(values[3 * (local_size - 3)], -1.0);
                ASSERT_EQ(col_idxs[3 * (local_size - 3) + 1], local_size - 3);
                ASSERT_EQ(values[3 * (local_size - 3) + 1], 2.0);
                ASSERT_EQ(row_ptrs[local_size - 2], 3 * (local_size - 3) + 2);
                ASSERT_EQ(col_idxs[3 * (local_size - 3) + 2], local_size - 2);
                ASSERT_EQ(values[3 * (local_size - 3) + 2], 1.0);
                ASSERT_EQ(row_ptrs[local_size - 1], 3 * (local_size - 3) + 3);
                ASSERT_EQ(col_idxs[3 * (local_size - 3) + 3], 0);
                ASSERT_EQ(values[3 * (local_size - 3) + 3], -1.0);
                ASSERT_EQ(col_idxs[3 * (local_size - 3) + 4], local_size - 1);
                ASSERT_EQ(values[3 * (local_size - 3) + 4], 1.0);
                ASSERT_EQ(row_ptrs[local_size], 3 * (local_size - 3) + 5);
            } else {
                ASSERT_EQ(row_ptrs[local_size - 2], 3 * (local_size - 3));
                ASSERT_EQ(col_idxs[3 * (local_size - 3)], local_size - 3);
                ASSERT_EQ(values[3 * (local_size - 3)], -1.0);
                ASSERT_EQ(col_idxs[3 * (local_size - 3) + 1], local_size - 2);
                ASSERT_EQ(values[3 * (local_size - 3) + 1], 2.0);
                ASSERT_EQ(row_ptrs[local_size - 1], 3 * (local_size - 3) + 2);
                ASSERT_EQ(col_idxs[3 * (local_size - 3) + 2], local_size - 1);
                ASSERT_EQ(values[3 * (local_size - 3) + 2], 1.0);
                ASSERT_EQ(row_ptrs[local_size], 3 * (local_size - 3) + 3);
            }
        }       
    }

    void assume_vectors_near(std::shared_ptr<gko::matrix::Dense<double>> vec1, std::shared_ptr<gko::matrix::Dense<double>> vec2, double tol = 1e-14) {
        ASSERT_EQ(vec1->get_size(), vec2->get_size());
        auto diff = vec1->clone();
        auto one = gko::initialize<gko::matrix::Dense<double>>({1.0}, vec1->get_executor());
        diff->sub_scaled(one, vec2);
        auto norm = gko::initialize<gko::matrix::Dense<double>>({0.0}, vec1->get_executor());
        diff->compute_norm2(norm);
        ASSERT_LE(norm->at(0, 0), tol);
    }
}

