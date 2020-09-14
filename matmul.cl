__kernel void init_id(__global int* matrix, uint dim) {
    uint row = get_global_id(0);
    uint col = get_global_id(1);

    if (row >= dim || col >= dim) {
        return;
    }

    matrix[row * dim + col] = row == col ? 1 : 0;
}

__kernel void multiply(__global int* left, uint rows_l, uint cols_l,
                       __global int* right, uint rows_r, uint cols_r,
                       __global int* result) {
    __local int tile_l[BLOCK_SIZE][BLOCK_SIZE];
    __local int tile_r[BLOCK_SIZE][BLOCK_SIZE];

    int tile_row = get_local_id(0);
    int tile_col = get_local_id(1);
    int row = get_global_id(0);
    int col = get_global_id(1);

    int cell = 0;
    for (int t = 0; t < cols_l; t += BLOCK_SIZE) {
        if (min(row, t + tile_row) < rows_l) {
            tile_l[tile_row][tile_col] = left[row * cols_l + t + tile_col];
        }
        if (min(col, t + tile_col) < cols_r) {
            tile_r[tile_row][tile_col] = right[(t + tile_row) * cols_r + col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < rows_l && col < cols_r) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                cell |= tile_l[tile_row][k] * tile_r[k][tile_col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    if (row < rows_l && col < cols_r) {
        result[row * cols_r + col] = cell;
    }
}