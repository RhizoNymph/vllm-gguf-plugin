// Minimal ggml runtime shim for the vendored llama.cpp MMQ kernels.
//
// `llamacpp/mmq.cuh` is vendored verbatim (see llamacpp/LLAMA_CPP_VERSION.txt) so
// that re-vendoring stays a copy rather than a merge. It compiles standalone, but
// references a few functions that normally live in ggml-cuda.cu -- the whole ggml
// backend, which we do not want. They are tiny and are reimplemented here against
// the CUDA runtime directly.
//
// Deliberately NOT provided: ggml_backend_cuda_context. `launch_mul_mat_q` only
// dereferences it inside the stream-k branch (for the fixup scratch buffer), so
// the launcher in mmq_mma.cu passes use_stream_k=false and never touches it.

#include "llamacpp/common.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// GGML_ASSERT/GGML_ABORT funnel here. ggml's version prints a backtrace; a plain
// abort is enough for us and keeps libggml out of the link.
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "vllm-gguf-plugin: fatal error at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

// Backs the CUDA_CHECK macro in common.cuh. Declared there, defined in
// ggml-cuda.cu, which we do not vendor -- omitting it still LINKS (shared
// objects tolerate undefined symbols) and then fails at dlopen, at which point
// the plugin quietly falls back to its Triton path. That failure mode is silent
// enough that a benchmark can measure Triton and look plausible, so anything
// added to the vendored set needs an import check, not just a successful build.
[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file,
                     int line, const char * msg) {
    fprintf(stderr, "vllm-gguf-plugin: CUDA error: %s\n", msg);
    fprintf(stderr, "  %s:%d in %s\n", file, line, func);
    fprintf(stderr, "  %s\n", stmt);
    abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static ggml_cuda_device_info ggml_cuda_init_device_info() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "vllm-gguf-plugin: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        info.device_count = 0;
        return info;
    }

    for (int id = 0; id < info.device_count && id < GGML_CUDA_MAX_DEVICES; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.devices[id].nsm       = prop.multiProcessorCount;
        info.devices[id].smpb      = prop.sharedMemPerBlock;
        info.devices[id].smpbo     = prop.sharedMemPerBlockOptin;
        info.devices[id].warp_size = prop.warpSize;
        info.devices[id].integrated = prop.integrated != 0;
        info.devices[id].total_vram = prop.totalGlobalMem;
        // The MMQ path never allocates through ggml's VMM pool.
        info.devices[id].vmm             = false;
        info.devices[id].vmm_granularity = 0;
        info.devices[id].supports_cooperative_launch = prop.cooperativeLaunch != 0;
        // ggml encodes NVIDIA capability as major*100 + minor*10 with no vendor
        // offset (AMD/MTHREADS get one, which is how GGML_CUDA_CC_IS_NVIDIA
        // distinguishes them). Getting this right is what makes
        // turing_mma_available() select the MMA path: sm_86 -> 860 >= 750.
        info.devices[id].cc = prop.major * 100 + prop.minor * 10;
    }
    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init_device_info();
    return info;
}

// From ggml.c. Only the block sizes for types we instantiate MMQ kernels for;
// anything else is rejected by gguf_mma::supported() before reaching here.
extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return QK4_0;
        case GGML_TYPE_Q4_1: return QK4_1;
        case GGML_TYPE_Q5_0: return QK5_0;
        case GGML_TYPE_Q5_1: return QK5_1;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return QK_K;
        default:
            GGML_ABORT("ggml_blck_size: unsupported type %d", (int) type);
    }
}

// Stream-k needs a scratch buffer for its cross-block fixup. It asks for the
// same size every call for a given shape, so a one-slot cache is enough and
// keeps cudaMalloc off the hot path. Not thread-safe by design: the plugin
// serialises on the CUDA stream it was handed.
namespace {
struct single_slot_pool : ggml_cuda_pool {
    void * ptr = nullptr;
    size_t capacity = 0;
    int device = 0;

    explicit single_slot_pool(int device) : device(device) {}

    ~single_slot_pool() override {
        if (ptr != nullptr) {
            (void) cudaFree(ptr);
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (size > capacity) {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
                ptr = nullptr;
            }
            CUDA_CHECK(cudaMalloc(&ptr, size));
            capacity = size;
        }
        *actual_size = capacity;
        return ptr;
    }

    // Retained rather than released, so the next call of the same shape reuses
    // it. Freed for real when the pool is destroyed.
    void free(void * p, size_t size) override {
        (void) p;
        (void) size;
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(
        int device, int stream_no) {
    (void) stream_no;
    return std::make_unique<single_slot_pool>(device);
}
