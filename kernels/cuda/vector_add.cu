#include<cuda_runtime.h>

#include<cmath>
#include<cstdlib>
#include<iostream>
#include<stdexcept>
#include<vector>
#include<algorithm>


#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)


__global__ void fill_kernel(float* data, float value, std::size_t size) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = value;
    }
}


__global__ void vector_add_kernel(const float* x, const float* y, float* output, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = x[index] + y[index];
    }
}


void verify_vector_add(int block_size) {
    constexpr std::size_t n = 1'000'003;

    constexpr float x_value = 1.25f;
    constexpr float y_value = 2.50f;
    constexpr float expected = 3.75f;

    float* x = nullptr;
    float* y = nullptr;
    float* output = nullptr;

    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(float)));

    const int blocks = static_cast<int>((n + block_size - 1) / block_size);

    fill_kernel<<<blocks, block_size>>>(x, x_value, n);
    fill_kernel<<<blocks, block_size>>>(y, y_value, n);
    vector_add_kernel<<<blocks, block_size>>>(x, y, output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_output(n);

    CUDA_CHECK(cudaMemcpy(host_output.data(), output, n * sizeof(float), cudaMemcpyDeviceToHost));

    for(std::size_t i = 0; i < n; i++) {
        if (std::fabs(host_output[i] - expected) > 1e-6) {
            throw std::runtime_error("VectorAdd correctness failed.");
        }
    }

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(output));
}


float benchmark_vector_add(std::size_t n, int block_size, int warmups, int iterations) {
    float* x = nullptr;
    float* y = nullptr;
    float* output = nullptr;

    const std::size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));

    const int blocks = static_cast<int>((n + block_size - 1) / block_size);

    fill_kernel<<<blocks, block_size>>>(x, 1.0f, n);
    fill_kernel<<<blocks, block_size>>>(y, 2.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmups; i++) {
        vector_add_kernel<<<blocks, block_size>>>(x, y, output, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cudaEvent_t> starts(iterations);
    std::vector<cudaEvent_t> stops(iterations);

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(starts[i]));
        vector_add_kernel<<<blocks, block_size>>>(x, y, output, n);
        CUDA_CHECK(cudaEventRecord(stops[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, starts[i], stops[i]));
        samples.push_back(elapsed_ms);
        CUDA_CHECK(cudaEventDestroy(starts[i]));
        CUDA_CHECK(cudaEventDestroy(stops[i]));
    }

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(output));

    std::sort(samples.begin(), samples.end());

    return samples[samples.size() / 2];
}


void print_build_and_device_info(int block_size) {
    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    int active_blocks_per_sm = 0;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, vector_add_kernel, block_size, 0));

    const int warps_per_block = (block_size + properties.warpSize - 1) / properties.warpSize;
    const int active_warps_per_sm = active_blocks_per_sm * warps_per_block;
    const int max_warps_per_sm = properties.maxThreadsPerMultiProcessor / properties.warpSize;
    
    const double occupancy = 100.0 * static_cast<double>(active_warps_per_sm) / max_warps_per_sm;

    int cuda_runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));

    std::cout << "device_name=" << properties.name << '\n';
    std::cout << "compute_capability=" << properties.major << '.' << properties.minor << '\n';
    std::cout << "sm_count=" << properties.multiProcessorCount << '\n';
    std::cout << "warp_size=" << properties.warpSize << '\n';
    std::cout << "max_threads_per_block=" << properties.maxThreadsPerBlock << '\n';
    std::cout << "max_threads_per_sm=" << properties.maxThreadsPerMultiProcessor << '\n';
    std::cout << "warps_per_block=" << warps_per_block << '\n';
    std::cout << "active_blocks_per_sm=" << active_blocks_per_sm << '\n';
    std::cout << "active_warps_per_sm=" << active_warps_per_sm << '\n';
    std::cout << "max_warps_per_sm=" << max_warps_per_sm << '\n';
    std::cout << "theoretical_occupancy_pct=" << occupancy << '\n';
    std::cout << "l2_cache_bytes=" << properties.l2CacheSize << '\n';
    std::cout << "nvcc_version=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '\n';
    
    #ifdef __GNUC__
    std::cout << "host_gcc_version=" << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__ << '\n';
    #endif

    std::cout << "compiled_cuda_arch=" << LLMFORGE_CUDA_ARCH << '\n';
    std::cout << "cuda_runtime_version_raw=" << cuda_runtime_version << '\n';
}



int main(int argc, char** argv) {
    std::size_t elements = static_cast<std::size_t>(1) << 27;

    int block_size = 256;
    int warmups = 20;
    int iterations = 50;

    if (argc >= 2) {
        elements = std::stoull(argv[1]);
    }

    if (argc >= 3) {
        block_size = std::stoi(argv[2]);
    }

    print_build_and_device_info(block_size);

    verify_vector_add(block_size);

    const float median_ms = benchmark_vector_add(elements, block_size, warmups, iterations);
    const double bytes = static_cast<double>(elements) * sizeof(float) * 3.0;
    const double bandwidth_gbps = bytes / (median_ms * 1e-3) / 1e9;

    std::cout << "elements=" << elements << '\n';
    std::cout << "block_size=" << block_size << '\n';
    std::cout << "median_ms=" << median_ms << '\n';
    std::cout << "bandwidth_gbps=" << bandwidth_gbps << '\n';
    std::cout << "correct=true" << '\n';

    return 0;
}