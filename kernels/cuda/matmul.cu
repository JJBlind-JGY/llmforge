#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                << " at " << __FILE__ << ":" << __LINE__ << '\n';       \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)


__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}


template<int TILE>
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    const int tile_count = (n + TILE - 1) / TILE;

    for (int tile = 0; tile < tile_count; tile++) {
        const int a_col = tile * TILE + tx;
        const int b_row = tile * TILE + ty;

        tile_a[ty][tx] = row < n && a_col < n ? a[row * n + a_col] : 0.0f;
        tile_b[ty][tx] = b_row < n && col < n ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

void launch_matmul(const std::string& variant, const float* a, const float* b, float* c, int n) {
    if (variant == "naive") {
        constexpr int tile = 16;
        const dim3 block(tile, tile);
        const dim3 grid((n + tile - 1) / tile, (n + tile - 1) / tile);
        matmul_naive_kernel<<<grid, block>>>(a, b, c, n);
        return;
    }

    if (variant == "tiled16") {
        constexpr int tile = 16;
        const dim3 block(tile, tile);
        const dim3 grid((n + tile - 1) / tile, (n + tile - 1) / tile);
        matmul_tiled_kernel<tile><<<grid, block>>>(a, b, c, n);
        return;
    }

    if (variant == "tiled32") {
        constexpr int tile = 32;
        const dim3 block(tile, tile);
        const dim3 grid((n + tile - 1) / tile, (n + tile - 1) / tile);
        matmul_tiled_kernel<tile><<<grid, block>>>(a, b, c, n);
        return;
    }

    throw std::invalid_argument("Unknown MatMul variant.");
}


void verify_matmul(const std::string& variant) {
    constexpr int n = 127;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> host_a(elements);
    std::vector<float> host_b(elements);
    std::vector<float> reference(elements);
    std::vector<float> actual(elements);

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            host_a[row * n + col] = static_cast<float>((row * 3 + col * 5) % 17 - 8) * 0.05f;
            host_b[row * n + col] = static_cast<float>((row * 7 + col * 2) % 13 - 6) * 0.04f;
        }
    }

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += static_cast<double>(host_a[row * n + k]) * static_cast<double>(host_b[k * n + col]);
            }

            reference[row * n + col] = static_cast<float>(sum);
        }
    }

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    CUDA_CHECK(cudaMalloc(&device_a, bytes));
    CUDA_CHECK(cudaMalloc(&device_b, bytes));
    CUDA_CHECK(cudaMalloc(&device_c, bytes));
    CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    launch_matmul(variant, device_a, device_b, device_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < elements; i++) {
        const float error = std::fabs(actual[i] - reference[i]);
        const float tolerance = 1e-3f + 1e-3f * std::fabs(reference[i]);
        if (error > tolerance) {
            std::cerr << "index=" << i << "\n";
            std::cerr << "expected=" << reference[i] << "\n";
            std::cerr << "actual=" << actual[i] << "\n";
            throw std::runtime_error("MatMul correctness failed.");
        }
    }

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));
}


__global__ void fill_matrix_kernel(float* data, std::size_t elements, float value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        data[index] = value;
    }
}


float benchmark_matmul(const std::string& variant, int n, int warmups, int iterations) {
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elements * sizeof(float);

    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemset(a, 0.0f, bytes));
    CUDA_CHECK(cudaMemset(b, 0.0f, bytes));

    const int fill_threads = 256;
    const int fill_blocks = static_cast<int>((elements + fill_threads - 1) / fill_threads);
    fill_matrix_kernel<<<fill_blocks, fill_threads>>>(a, elements, 0.5f);
    fill_matrix_kernel<<<fill_blocks, fill_threads>>>(b, elements, 0.25f);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmups; i++) {
        launch_matmul(variant, a, b, c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(c, 0.0f, bytes));
        CUDA_CHECK(cudaEventRecord(start));
        launch_matmul(variant, a, b, c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void print_matmul_kernel_info(const std::string& variant) {
    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    cudaFuncAttributes attributes{};

    int threads_per_block = 0;
    int active_blocks_per_sm = 0;

    if (variant == "naive") {
        threads_per_block = 16 * 16;
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, matmul_naive_kernel));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, matmul_naive_kernel, threads_per_block, 0));
    } else if (variant == "tiled16") {
        threads_per_block = 16 * 16;
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, matmul_tiled_kernel<16>));

        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm,
                matmul_tiled_kernel<16>,
                threads_per_block,
                0
            ));
    } else if (variant == "tiled32") {
        threads_per_block = 32 * 32;

        CUDA_CHECK(
            cudaFuncGetAttributes(
                &attributes,
                matmul_tiled_kernel<32>
            )
        );

        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm,
                matmul_tiled_kernel<32>,
                threads_per_block,
                0
            ));
    } else {
        throw std::invalid_argument(
            "Unknown MatMul variant."
        );
    }

    const int warps_per_block =
        (
            threads_per_block
            + properties.warpSize
            - 1
        )
        / properties.warpSize;

    const int active_warps_per_sm =
        active_blocks_per_sm
        * warps_per_block;

    const int max_warps_per_sm =
        properties.maxThreadsPerMultiProcessor
        / properties.warpSize;

    const double theoretical_occupancy =
        100.0
        * static_cast<double>(
            active_warps_per_sm
        )
        / max_warps_per_sm;

    std::cout
        << "kernel_threads_per_block="
        << threads_per_block
        << '\n';

    std::cout
        << "kernel_registers_per_thread="
        << attributes.numRegs
        << '\n';

    std::cout
        << "kernel_static_shared_bytes="
        << attributes.sharedSizeBytes
        << '\n';

    std::cout
        << "kernel_active_blocks_per_sm="
        << active_blocks_per_sm
        << '\n';

    std::cout
        << "kernel_active_warps_per_sm="
        << active_warps_per_sm
        << '\n';

    std::cout
        << "kernel_theoretical_occupancy_pct="
        << theoretical_occupancy
        << '\n';
}

int main(int argc, char** argv) {
    int n = 1024;                  
    std::string variant = "naive";  // naive / tiled16 / tiled32
    int warmups = 5;
    int iterations = 10;

    if (argc >= 2) {
        n = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        variant = argv[2];
    }
    if (argc >= 4) {
        warmups = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        iterations = std::stoi(argv[4]);
    }

    if (n <= 0) {
        throw std::invalid_argument("n must be positive.");
    }
    if (variant != "naive" && variant != "tiled16" && variant != "tiled32") {
        throw std::invalid_argument("variant must be naive, tiled16, or tiled32.");
    }

    print_matmul_kernel_info(variant);

    std::cout << "\n--- Verifying correctness ---\n";
    verify_matmul(variant);
    std::cout << "Correctness: PASSED\n";

    std::cout << "\n--- Running benchmark ---\n";
    const float median_ms = benchmark_matmul(variant, n, warmups, iterations);

    const double flops = 2.0 * static_cast<double>(n) * n * n;
    const double seconds = median_ms / 1000.0;
    const double effective_tflops = flops / seconds / 1e12;

    std::cout << "\n--- Results ---\n";
    std::cout << "variant=" << variant << '\n';
    std::cout << "n=" << n << '\n';
    std::cout << "warmups=" << warmups << '\n';
    std::cout << "iterations=" << iterations << '\n';
    std::cout << "median_ms=" << median_ms << '\n';
    std::cout << "effective_tflops=" << effective_tflops << '\n';
    std::cout << "correct=true\n";

    return 0;
}