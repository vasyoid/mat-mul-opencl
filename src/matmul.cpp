#include "matmul.h"

Matrix::Matrix(size_t rows, size_t cols): rows(rows), cols(cols) {
    data = std::vector<int>(rows * cols);
}

Matrix Matrix::read(std::istream& input) {
    size_t rows;
    size_t cols;
    input >> rows >> cols;
    Matrix matrix(rows, cols);
    for (int& x : matrix.data) {
        input >> x;
    }
    return matrix;
}

Matrix Matrix::read_square(std::istream& input) {
    size_t dim;
    input >> dim;
    Matrix matrix(dim, dim);
    for (int& x : matrix.data) {
        input >> x;
    }
    return matrix;
}

Matrix Matrix::generate(size_t rows, size_t cols) {
    Matrix matrix(rows, cols);
    std::mt19937 generator;
    std::bernoulli_distribution distribution(0.1);
    for (int& x : matrix.data) {
        x = distribution(generator);
    }
    return matrix;
}

void Matrix::print(std::ostream& output) {
    output << rows << " " << cols << std::endl;
    print_data(output);
}

void Matrix::print_square(std::ostream &output) {
    output << rows << std::endl;
    print_data(output);
}

void Matrix::print_data(std::ostream &output) const {
    for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                    output << data[row * cols + col] << " ";
                }
            output << std::endl;
        }
}

const int* Matrix::data_ptr() const {
    return &data[0];
}

int* Matrix::data_ptr() {
    return &data[0];
}

int Matrix::at(size_t row, size_t col) const {
    return data[row * cols + col];
}

int& Matrix::at(size_t row, size_t col) {
    return data[row * cols + col];
}

MatrixMultiplier::MatrixMultiplier(size_t block_size): block_size_(block_size) {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    context_ = cl::Context(devices);

    std::ifstream cl_file("matmul.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));
    cl::Program program(context_, source);
    std::string options = "-D BLOCK_SIZE=" + std::to_string(block_size_);
    program.build(devices, options.c_str());
    multiply_kernel_ = cl::Kernel(program, "multiply");
    init_id_kernel_ = cl::Kernel(program, "init_id");

    command_queue_ = cl::CommandQueue(context_, devices[0], CL_QUEUE_PROFILING_ENABLE);
}

GpuMatrix MatrixMultiplier::write_matrix(const Matrix& matrix) const {
    size_t buffer_size = matrix.rows * matrix.cols;
    cl::Buffer buffer(context_, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
    command_queue_.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(int) * buffer_size, matrix.data_ptr());
    return GpuMatrix(matrix.rows, matrix.cols, buffer);
}

Matrix MatrixMultiplier::read_matrix(const GpuMatrix& gpu_matrix) const {
    size_t buffer_size = gpu_matrix.rows * gpu_matrix.cols;
    Matrix matrix(gpu_matrix.rows, gpu_matrix.cols);
    command_queue_.enqueueReadBuffer(gpu_matrix.data, CL_TRUE, 0, sizeof(int) * buffer_size, matrix.data_ptr());
    return matrix;
}

GpuMatrix MatrixMultiplier::create_matrix(size_t rows, size_t cols) const {
    return GpuMatrix(rows, cols, cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(int) * rows * cols));
}

GpuMatrix MatrixMultiplier::create_id_matrix(size_t dim) const {
    cl::Buffer data = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(int) * dim * dim);
    cl::KernelFunctor functor(init_id_kernel_,
                            command_queue_,
                            cl::NullRange,
                            cl::NDRange(dim, dim),
                            cl::NullRange);
    functor(data, dim);
    return GpuMatrix(dim, dim, data);
}

void MatrixMultiplier::multiply(const GpuMatrix& left, const GpuMatrix& right, const GpuMatrix& result) const {
    cl::NDRange global_work_size(((left.rows + block_size_ - 1) / block_size_) * block_size_,
                               ((right.cols + block_size_ - 1) / block_size_) * block_size_);
    cl::KernelFunctor functor(multiply_kernel_,
                            command_queue_,
                            cl::NullRange,
                            global_work_size,
                            cl::NDRange(block_size_, block_size_));
    functor(left.data, left.rows, left.cols, right.data, right.rows, right.cols, result.data);
}

Matrix multiply_two_matrices(const Matrix& left, const Matrix& right, const MatrixMultiplier& multiplier) {
    GpuMatrix gpu_left = multiplier.write_matrix(left);
    GpuMatrix gpu_right = multiplier.write_matrix(right);
    GpuMatrix gpu_result = multiplier.create_matrix(left.rows, right.cols);
    multiplier.multiply(gpu_left, gpu_right, gpu_result);
    return multiplier.read_matrix(gpu_result);
}

static Matrix power(const Matrix& base, size_t pow, const MatrixMultiplier& multiplier) {
    GpuMatrix gpu_base = multiplier.write_matrix(base);
    GpuMatrix gpu_result = multiplier.create_id_matrix(base.rows);
    GpuMatrix gpu_tmp = multiplier.create_matrix(base.rows, base.rows);
    while (pow > 0) {
        if (pow & 1u) {
            multiplier.multiply(gpu_result, gpu_base, gpu_tmp);
            std::swap(gpu_result, gpu_tmp);
        }
        pow >>= 1u;
        if (pow > 0) {
            multiplier.multiply(gpu_base, gpu_base, gpu_tmp);
            std::swap(gpu_base, gpu_tmp);
        }
    }
    return multiplier.read_matrix(gpu_result);
}

Matrix transitive_closure(const Matrix& graph) {
    MatrixMultiplier multiplier;
    return power(graph, graph.rows - 1, multiplier);
}
