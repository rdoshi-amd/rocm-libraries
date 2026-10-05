// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime_api.h>
#ifdef PROBE_BLAS
#include <hipblaslt/hipblaslt.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstring>

static const auto start = std::chrono::steady_clock::now();

static void mark(const char* phase, const char* operation, int status = 0) {
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%.6f %s %s status=%d\n", seconds, phase, operation, status);
}

#define CALL(expression)                         \
    do {                                         \
        mark("before", #expression);             \
        const auto status = (expression);        \
        mark("after", #expression, int(status)); \
        if (int(status) != 0) return 1;          \
    } while (false)

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
    std::setbuf(stderr, nullptr);
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Normally linked HIP startup probe; no arguments run the GPU workload.");
        return 0;
    }
    if (argc != 1) return 2;
    mark("before", "HIP initialization");
    CALL(hipInit(0));
    int count = 0;
    CALL(hipGetDeviceCount(&count));
    std::printf("device_count=%d\n", count);
    if (count == 0) return 1;
    int device = -1;
    CALL(hipGetDevice(&device));
    char pci[64] = {};
    CALL(hipDeviceGetPCIBusId(pci, sizeof(pci), device));
    std::printf("selected_device=%d pci_bus_id=%s\n", device, pci);
#ifdef PROBE_BLAS
    hipblasLtHandle_t handle = nullptr;
    CALL(hipblasLtCreate(&handle));
    CALL(hipblasLtDestroy(handle));
#else
    void* memory = nullptr;
    constexpr size_t bytes = 25 * 1024 * 1024;
    CALL(hipMalloc(&memory, bytes));
    CALL(hipMemset(memory, 0, bytes));
    CALL(hipDeviceSynchronize());
    CALL(hipFree(memory));
#endif
    std::puts("PROBE PASSED");
    return 0;
}
