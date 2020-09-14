#ifndef MATMUL_TEST_H
#define MATMUL_TEST_H

#include <cstdint>

void performance_test(std::size_t a, std::size_t b, std::size_t c);
void correctness_test_multiply(std::size_t a, std::size_t b, std::size_t c);
void correctness_test_closure(std::size_t size);

#endif //MATMUL_TEST_H
