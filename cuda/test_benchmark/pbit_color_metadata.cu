// CPU-only metadata tool: use exactly the simulator's coloring implementation.
#include <cstdio>
#include <cstdlib>
#include "../pbit_coloring.cuh"

int main(int argc, char** argv) {
    int n = atoi(argv[2]);
    std::vector<float> J(size_t(n) * n);
    FILE* file = fopen(argv[1], "rb");
    fread(J.data(), sizeof(float), J.size(), file);
    fclose(file);
    auto groups = largest_degree_first_coloring(J.data(), n);
    printf("{\"num_pbits\":%d,\"num_colors\":%zu,\"colors\":[", n, groups.size());
    for (int c = 0; c < int(groups.size()); c++) {
        if (c) printf(",");
        printf("{\"color_index\":%d,\"nodes_in_color\":%zu}", c, groups[c].size());
    }
    printf("]}\n");
    return 0;
}
