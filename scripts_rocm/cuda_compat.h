#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <string.h>

static inline hipError_t cudaMemcpyToSymbol_compat(const void* symbol,
                                                   const void* src,
                                                   size_t size,
                                                   size_t offset,
                                                   hipMemcpyKind kind) {
  if (kind == hipMemcpyDeviceToDevice) {
    hipPointerAttribute_t attr;
    hipError_t a = hipPointerGetAttributes(&attr, src);
    if (a != hipSuccess || attr.type != hipMemoryTypeDevice) {
      return hipMemcpyToSymbol(symbol, src, size, offset, hipMemcpyHostToDevice);
    }
    void* tmp = malloc(size);
    if (!tmp) return hipErrorOutOfMemory;
    hipError_t e = hipMemcpy(tmp, src, size, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
      free(tmp);
#ifdef HIP_CHECK_EACH_CTE
      fprintf(stderr, "cte_d2h failed: err=%d\n", (int)e);
#endif
      /* Fallback: try treating src as host pointer */
      return hipMemcpyToSymbol(symbol, src, size, offset, hipMemcpyHostToDevice);
    }
    e = hipMemcpyToSymbol(symbol, tmp, size, offset, hipMemcpyHostToDevice);
    free(tmp);
    return e;
  }
  return hipMemcpyToSymbol(symbol, src, size, offset, kind);
}

#ifdef HIP_CHECK_EACH_CTE
  #include <stdio.h>
  static inline void hip_debug_ptr(const void* src) {
    hipPointerAttribute_t attr;
    hipError_t a = hipPointerGetAttributes(&attr, src);
    if (a != hipSuccess) {
      fprintf(stderr, "cte_ptr: attr_err=%d\n", (int)a);
      return;
    }
    fprintf(stderr, "cte_ptr: type=%d dev=%d devptr=%p hostptr=%p managed=%d\n",
            (int)attr.type, attr.device, attr.devicePointer, attr.hostPointer, attr.isManaged);
  }
  #define cudaMemcpyToSymbol(symbol, src, size, offset, kind) do { \
    fprintf(stderr, "cte_cpy try: %s size=%zu src=%p kind=%d\\n", #symbol, (size_t)(size), (const void*)(src), (int)(kind)); \
    if ((kind) == hipMemcpyDeviceToDevice) hip_debug_ptr((src)); \
    hipError_t _e = cudaMemcpyToSymbol_compat(HIP_SYMBOL(symbol), (src), (size), (offset), (kind)); \
    if (_e != hipSuccess) { \
      fprintf(stderr, "cte_cpy failed: %s size=%zu err=%d\\n", #symbol, (size_t)(size), (int)_e); \
    } \
  } while(0)
#else
  #define cudaMemcpyToSymbol(symbol, src, size, offset, kind) \
    cudaMemcpyToSymbol_compat(HIP_SYMBOL(symbol), (src), (size), (offset), (kind))
#endif
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#define cudaFuncSetCacheConfig(func, cfg) hipFuncSetCacheConfig((const void*)(func), (cfg))

#endif
