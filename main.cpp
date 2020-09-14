#include "matmul.h"
#include "test.h"

static void print_usage(const char* name) {
    std::cerr << "usage:\n"
              << name << " -t: run correctness tests\n"
              << name << " -p: run performance tests\n"
              << name << " -m: read two matrices from stdin and write their product to stdout\n"
              << name << " -c: read a graph from stdin and write its transitive closure to stdout\n";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 0;
    }
    std::string arg = std::string(argv[1]);
    if (arg == "-t") {
        try {
            for (size_t a = 100; a <= 300; a += 100) {
                for (size_t b = 100; b <= 300; b += 100) {
                    for (size_t c = 100; c <= 300; c += 100) {
                        correctness_test_multiply(a, b, c);
                    }
                }
            }
            for (size_t size = 100; size <= 200; size += 10) {
                correctness_test_closure(size);
            }
        } catch (cl::Error &e) {
            std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }
    } else if (arg == "-p") {
        for (size_t size = 128; size <= 2048; size *= 2) {
            try {
                performance_test(size, size, size);
            } catch (cl::Error &e) {
                std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
            }
        }
    } else if (arg == "-m") {
        Matrix left = Matrix::read(std::cin);
        Matrix right = Matrix::read(std::cin);
        try {
            MatrixMultiplier multiplier;
            Matrix result = multiply_two_matrices(left, right, multiplier);
            result.print(std::cout);
        } catch (cl::Error &e) {
            std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }
    } else if (arg == "-c") {
        Matrix graph = Matrix::read_square(std::cin);
        try {
            Matrix result = transitive_closure(graph);
            result.print_square(std::cout);
        } catch (cl::Error &e) {
            std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }
    }
    return 0;
}