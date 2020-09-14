#include "test.h"
#include "matmul.h"
#include <exception>
#include <tuple>

template<typename T>
static void assert_equal(T expected, T actual, const std::string& message) {
    if (expected != actual) {
        throw std::runtime_error(message);
    }
}

static Matrix multiply_two_matrices_cpu(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows, right.cols);
    for (size_t row = 0; row < result.rows; ++row) {
        for (size_t col = 0; col < result.cols; ++col) {
            for (size_t k = 0; k < left.cols; ++k) {
                result.at(row, col) |= left.at(row, k) * right.at(k, col);
            }
        }
    }
    return result;
}

void correctness_test_multiply(size_t a, size_t b, size_t c) {
    Matrix left = Matrix::generate(a, b);
    Matrix right = Matrix::generate(b, c);
    Matrix expected = multiply_two_matrices_cpu(left, right);
    MatrixMultiplier multiplier;
    Matrix actual = multiply_two_matrices(left, right, multiplier);

    std::cout << "multiply (" << a << " x " << b << ") * (" << b << " x " << c << ") ";

    try {
        assert_equal(std::tie(expected.rows, expected.cols), std::tie(actual.rows, actual.cols), "incorrect product size");
        for (size_t i = 0; i < expected.rows * expected.cols; ++i) {
            assert_equal(expected.data[i], actual.data[i], "incorrect product value");
        }
        std::cout << "OK" << std::endl;
    } catch (std::runtime_error& e) {
        std::cout << "FAIL: " << e.what() << std::endl;
    }
}

static Matrix transitive_closure_cpu(const Matrix& graph) {
    Matrix result = graph;
    for (size_t k = 0; k < result.cols; ++k) {
        for (size_t row = 0; row < result.rows; ++row) {
            for (size_t col = 0; col < result.cols; ++col) {
                result.at(row, col) |= result.at(row, k) * result.at(k, col);
            }
        }
    }
    return result;
}

void correctness_test_closure(size_t size) {
    Matrix graph = Matrix::generate(size, size);
    Matrix expected = transitive_closure_cpu(graph);
    Matrix actual = transitive_closure(graph);

    std::cout << "closure " << size << " x " << size << " ";

    try {
        for (size_t i = 0; i < expected.rows * expected.cols; ++i) {
            assert_equal(expected.data[i], actual.data[i], "incorrect closure");
        }
        std::cout << "OK" << std::endl;
    } catch (std::runtime_error& e) {
        std::cout << "FAIL: " << e.what() << std::endl;
    }
}

void performance_test(size_t a, size_t b, size_t c) {
    Matrix left = Matrix::generate(a, b);
    Matrix right = Matrix::generate(b, c);
    std::cout << a << " x " << b << " x " << c << std::endl;
    for (int i = 4; i <= 16; i += 2) {
        MatrixMultiplier multiplier(i);
        auto start_time = std::chrono::high_resolution_clock::now();

        multiply_two_matrices(left, right, multiplier);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = (end_time - start_time) / std::chrono::milliseconds(1);
        std::cout << "  block_size = " << i << ": " << duration << " ms" << std::endl;
    }
}
