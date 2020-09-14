#ifndef MATMUL_MATMUL_H
#define MATMUL_MATMUL_H

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <chrono>

struct GpuMatrix {
    cl::Buffer data;
    size_t rows;
    size_t cols;

    GpuMatrix(size_t rows, size_t cols, const cl::Buffer& data): rows(rows), cols(cols), data(data) { }
};

struct Matrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<int> data;

    Matrix(size_t rows, size_t cols);
    static Matrix read(std::istream& input);
    static Matrix read_square(std::istream& input);
    static Matrix generate(size_t rows, size_t cols);
    void print(std::ostream& output);
    void print_square(std::ostream& output);
    const int* data_ptr() const;
    int* data_ptr();
    int at(size_t row, size_t col) const;
    int& at(size_t row, size_t col);

 private:
    void print_data(std::ostream &output) const;
};

class MatrixMultiplier {
public:
    explicit MatrixMultiplier(size_t block_size = 12);
    GpuMatrix write_matrix(const Matrix& matrix) const;
    Matrix read_matrix(const GpuMatrix& gpu_matrix) const;
    GpuMatrix create_matrix(size_t rows, size_t cols) const;
    GpuMatrix create_id_matrix(size_t dim) const;
    void multiply(const GpuMatrix& left, const GpuMatrix& right, const GpuMatrix& result) const;

private:
    cl::Context context_;
    cl::CommandQueue command_queue_;
    cl::Kernel multiply_kernel_;
    cl::Kernel init_id_kernel_;
    size_t block_size_;
};

Matrix multiply_two_matrices(const Matrix& left, const Matrix& right, const MatrixMultiplier& multiplier);
Matrix transitive_closure(const Matrix& graph);

#endif //MATMUL_MATMUL_H
