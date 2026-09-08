#include<cuda_runtime.h>

#include<algorithm>
#include<cmath>
#include<cstddef>
#include<cstdlib>
#include<iostream>
#include<stdexcept>
#include<string>
#include<vector>


#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                << " at " << __FILE__ << ":" << __LINE__ << '\n';       \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)


__global__ void fill_kernel(float* data, float value, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        data[index] = value;
    }
}

__global__ void reduce_atomic_kernel(const float* input, float* output, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        atomicAdd(output, input[index]);
    }
}

__global__ void reduce_shared_kernel(const float* input, float* output, std::size_t n) {
    extern __shared__ float shared[];
    const unsigned int tid = threadIdx.x;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    shared[tid] = index < n ? input[index] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}


// 效率较低，出现Bank Conflict
__global__ void reduce_shared_interleaved_kernel(const float* input, float* output, std::size_t n) {
    extern __shared__ float shared[];
    const unsigned int tid = threadIdx.x;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;

    shared[tid] = index < n ? input[index] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        const unsigned int shared_index = 2 * stride * tid;
        if (shared_index < blockDim.x) {
            shared[shared_index] += shared[shared_index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}


__device__ __forceinline__
float warp_reduce_sum(float value) {
    constexpr unsigned int full_mask = 0xffffffffu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(full_mask, value, offset);
    }
    return value;
}


__global__ void reduce_warp_kernel(const float* input, float* output, std::size_t n) {
    __shared__ float warp_sums[32];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid >> 5;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    float value = index < n ? input[index] : 0.0f;

    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}



void launch_reduction(const std::string& variant, const float* input, float* output, std::size_t n, int block_size) {
    const int blocks = static_cast<int>((n + block_size - 1) / block_size);
    
    if (variant == "atomic") {
        reduce_atomic_kernel<<<blocks, block_size>>>(input, output, n);
        return;
    }

    if (variant == "shared_interleaved") {
        const std::size_t shared_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
        reduce_shared_interleaved_kernel<<<blocks, block_size, shared_bytes>>>(input, output, n);
        return;
    }

    if (variant == "shared" || variant == "shared_sequential") {
        const std::size_t shared_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
        reduce_shared_kernel<<<blocks, block_size, shared_bytes>>>(input, output, n);
        return;
    }

    if (variant == "warp") {
        reduce_warp_kernel<<<blocks, block_size>>>(input, output, n);
        return;
    }

    throw std::invalid_argument("Unknown reduction variant.");
}


void verify_reduction(const std::string& variant, int block_size) {
    constexpr std::size_t n = 1'000'003;
    
    float* input = nullptr;
    float* output = nullptr;

    CUDA_CHECK(cudaMalloc(&input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, sizeof(float)));

    const int blocks = static_cast<int>((n + block_size - 1) / block_size);
    fill_kernel<<<blocks, block_size>>>(input, 1.0f, n);
    CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));

    launch_reduction(variant, input, output, n, block_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float actual = 0.0f;
    CUDA_CHECK(cudaMemcpy(&actual, output, sizeof(float), cudaMemcpyDeviceToHost));

    const float expected = static_cast<float>(n);
    if (std::fabs(actual - expected) > 0.5f) {
        std::cerr << "expected=" << expected << "\n";
        std::cerr << "actual=" << actual << "\n";
        throw std::runtime_error("Reduction correctness failed.");
    }

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
}


float benchmark_reduction(const std::string& variant, std::size_t n, int block_size, int warmups, int iterations) {
    float* input = nullptr;
    float* output = nullptr;

    const std::size_t input_bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&input, input_bytes));
    CUDA_CHECK(cudaMalloc(&output, sizeof(float)));

    const int blocks = static_cast<int>((n + block_size - 1) / block_size);
    fill_kernel<<<blocks, block_size>>>(input, 1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmups; ++i) {
        CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
        launch_reduction(variant, input, output, n, block_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        launch_reduction(variant, input, output, n, block_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    
    std::sort(samples.begin(), samples.end());

    return samples[samples.size() / 2];
}


bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}


// ============================================================
// 【新增】与 vector_add.cu 完全一致的设备信息打印函数
// ============================================================
void print_build_and_device_info(int block_size) {
    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, reduce_shared_kernel, block_size, 0));

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
    std::size_t elements = static_cast<std::size_t>(1) << 24;
    int block_size = 256;
    std::string variant = "atomic";
    int warmups = 3;
    int iterations = 10;
    
    if (argc >= 2) {
        elements = std::stoull(argv[1]);
    }

    if (argc >= 3) {
        block_size = std::stoi(argv[2]);
    }

    if (argc >= 4) {
        variant = argv[3];
    }

    if (argc >= 5) {
        warmups = std::stoi(argv[4]);
    }

    if (argc >= 6) {
        iterations = std::stoi(argv[5]);
    }

    if (!is_power_of_two(block_size) || block_size > 1024 || block_size < 32) {
        throw std::invalid_argument("block_size must be a power of two between 1024 and at least 32.");
    }

    // ============================================================
    // 【关键修改】与 vector_add.cu 完全对齐：先打印设备信息（强制初始化上下文）
    // ============================================================
    print_build_and_device_info(block_size);

    // 现在上下文已建立，后续所有 CUDA 调用都能正常执行
    verify_reduction(variant, block_size);
    const float median_ms = benchmark_reduction(variant, elements, block_size, warmups, iterations);
    const double seconds = median_ms / 1000.0;
    const double useful_bytes = static_cast<double>(elements) * sizeof(float);
    const double useful_bandwidth = useful_bytes / seconds / 1e9;
    const double reduction_flops = static_cast<double>(elements - 1);
    const double effective_gflops = reduction_flops / seconds / 1e9;

    const std::size_t blocks = (elements + block_size - 1) / block_size;
    const std::size_t atomic_updates = variant == "atomic" ? elements : blocks;

    std::cout << "variant=" << variant << "\n";
    std::cout << "elements=" << elements << "\n";
    std::cout << "block_size=" << block_size << "\n";
    std::cout << "median_ms=" << median_ms << "\n";
    std::cout << "useful_input_bandwidth_gbps=" << useful_bandwidth << "\n";
    std::cout << "effective_gflops=" << effective_gflops << "\n";
    std::cout << "global_atomic_updates=" << atomic_updates << "\n";
    std::cout << "correct=true\n";
    return 0;
}