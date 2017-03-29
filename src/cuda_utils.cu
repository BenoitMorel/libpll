#define PLLCUDA
 # include "cuda_runtime.h"
#include <stdio.h>
#include "pll.h"


static unsigned int cuda_check(cudaError_t error_code, const char *msg)
{
  if (cudaSuccess != error_code) 
  {
    fprintf(stderr, "[libpll cuda error] [%s] [%s]\n", msg, cudaGetErrorString(error_code));
    return 0;
  }
  return 1;
}

PLL_EXPORT void pll_print_cuda_info()
{
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  fprintf(stderr, "Devices: %d\n", nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    fprintf(stderr, "Major revision number:         %d\n",  devProp.major);
    fprintf(stderr, "Minor revision number:         %d\n",  devProp.minor);
    fprintf(stderr, "Name:                          %s\n",  devProp.name);
    fprintf(stderr, "Total global memory:           %u\n",  devProp.totalGlobalMem);
    fprintf(stderr, "Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    fprintf(stderr, "Total registers per block:     %d\n",  devProp.regsPerBlock);
    fprintf(stderr, "Warp size:                     %d\n",  devProp.warpSize);
    fprintf(stderr, "Maximum memory pitch:          %u\n",  devProp.memPitch);
    fprintf(stderr, "Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
      fprintf(stderr, "Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
      fprintf(stderr, "Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    fprintf(stderr, "Clock rate:                    %d\n",  devProp.clockRate);
    fprintf(stderr, "Total constant memory:         %u\n",  devProp.totalConstMem);
    fprintf(stderr, "Texture alignment:             %u\n",  devProp.textureAlignment);
    fprintf(stderr, "Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    fprintf(stderr, "Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    fprintf(stderr, "Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
  }
}

PLL_EXPORT void * pll_cuda_malloc(size_t size)
{
  void *res;
  if (!cuda_check(cudaMalloc(&res, size), "pll_cuda_malloc"))
    return 0;
  return res;
}

void * pll_cuda_calloc(size_t size)
{
  void *res = pll_cuda_malloc(size);
  if (!res)
  {
    return 0;
  }
  if (!cuda_check(cudaMemset(res, 0, size), "pll_cuda_malloc"))
  {
    cudaFree(res);
    res = 0;
  }
  return 0;
}

int pll_cuda_memcpy_to_gpu(void *dest, const void *src, size_t n)
{
  return cuda_check(cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice), "pll_cuda_memcpy_to_gpu");
}

int pll_cuda_memcpy_to_cpu(void *dest, const void *src, size_t n)
{
  return cuda_check(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost), "pll_cuda_memcpy_to_cpu");
}

