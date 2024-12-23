#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            std::cerr << "CUDA error at " << __FILE__       \
                      << ":" << __LINE__ << ": "            \
                      << cudaGetErrorString(err) << "\n";   \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// store number of occurrences of values in a at their value as index in 
// out.
// assume out has size of max(a)
__global__
void count_occurrences(int *out, int *a, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        atomicAdd(&out[a[i]], 1);
}

// multiply every index in out by the occurrences in a and b,
// return list of length n containing all products
__global__
void multiply_weighted_occurences(int *out, int *a, int *b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        out[i] = i * a[i] * b[i];
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    // Get input size
    int n_lines = 0;
    std::string line;
    std::ifstream inputFile("input.txt");
    while (std::getline(inputFile, line))
        ++n_lines;
    int N = n_lines;
    inputFile.close();
    inputFile.clear();

    int *a, *b, *out_a, *out_b, *out_reduce;
    CUDA_CHECK(cudaMallocManaged(&a, N*sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&b, N*sizeof(int)));
    
    int i = 0;
    inputFile.open("input.txt");
    while (std::getline(inputFile, line))
    {
        std::stringstream ss(line);
        int x, y;
        // no error handling for malformed input
        ss >> x >> y;
        a[i] = x;
        b[i] = y;
        ++i;
    }
    inputFile.close();

    std::sort(a, a + N);
    std::sort(b, b + N);
    int M = std::max(a[N-1], b[N-1]);
    
    // initialize output
    CUDA_CHECK(cudaMallocManaged(&out_a, M*sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&out_b, M*sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&out_reduce, M*sizeof(int)));
    cudaMemset(out_a, 0, M*sizeof(int));
    cudaMemset(out_b, 0, M*sizeof(int));
    cudaMemset(out_reduce, 0, M*sizeof(int));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    count_occurrences<<<numBlocks, blockSize>>>(out_a, a, N);
    count_occurrences<<<numBlocks, blockSize>>>(out_b, b, N);
    cudaDeviceSynchronize();

    multiply_weighted_occurences<<<numBlocks, blockSize>>>(out_reduce, out_a, out_b, M);
    cudaDeviceSynchronize();

    std::cout << "Answer: " 
        << std::accumulate(out_reduce, out_reduce + M, 0) 
        << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(out_a);
    cudaFree(out_b);
    cudaFree(out_reduce);

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Took: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "μs" << std::endl;
}
