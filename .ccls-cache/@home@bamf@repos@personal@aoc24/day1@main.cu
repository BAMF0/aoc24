#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

__global__
void subtract(int n, int *x, int *y, int *z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        z[i] = (x[i] < y[i]) * (y[i] - x[i]) + 
               (y[i] < x[i]) * (x[i] - y[i]);
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

    int *a, *b, *c;
    cudaMallocManaged(&a, N*sizeof(int));
    cudaMallocManaged(&b, N*sizeof(int));
    cudaMallocManaged(&c, N*sizeof(int));
    
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

    // Sort arrays
    std::sort(a, a + N);
    std::sort(b, b + N);
        
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    subtract<<<numBlocks, blockSize>>>(N, a, b, c);

    cudaDeviceSynchronize();

    std::cout << "Answer: " << std::accumulate(c, c + N, 0) << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Took: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "μs" << std::endl;
}
