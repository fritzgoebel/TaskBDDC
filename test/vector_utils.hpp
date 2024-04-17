overlapping_vector setup_three_part_vector() {
    using part = gko::experimental::distributed::Partition<int, int>;
    std::vector<std::vector<int>> inner_idxs{{0, 1, 2, 3, 4, 5, 6}, {10, 11, 12, 13, 14, 15, 16}, {20, 21, 22, 23, 24, 25, 26}};
    std::vector<std::vector<int>> bndry_idxs{{7, 8, 9, 17, 19, 27, 29}, {7, 9, 17, 18, 19, 28, 29}, {8, 9, 18, 19, 27, 28, 29}};
    auto exec = gko::ReferenceExecutor::create();
    auto partition = gko::share(part::build_from_global_size_uniform(exec, 3, 30));

    auto vec = overlapping_vector(inner_idxs, bndry_idxs, partition, 30);
    return vec;
}

