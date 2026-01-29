#pragma once
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>
#include "cuda.h"

#ifndef __CUDA_PITCHED_PTR_DEFINED__
#define __CUDA_PITCHED_PTR_DEFINED__
typedef struct cudaPitchedPtr {
  void*  ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
} cudaPitchedPtr;

static inline cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) {
  cudaPitchedPtr r;
  r.ptr   = d;
  r.pitch = p;
  r.xsize = xsz;
  r.ysize = ysz;
  return r;
}
#endif

#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif

#ifndef cudaMallocPitch
#define cudaMallocPitch hipMallocPitch
#endif

#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif

#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif

#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif

#ifndef cudaMemcpy2D
#define cudaMemcpy2D hipMemcpy2D
#endif

#if !defined(__CUDA_MEMCPY3D_PARMS_DEFINED__) || !defined(__CUDA_MEMCPY3D_DEFINED__)
/* cuda.h provides cudaMemcpy3DParms/cudaMemcpy3D */
#endif

#if !defined(__CUDA_POS_DEFINED__) || !defined(__CUDA_EXTENT_DEFINED__)
/* cuda.h provides cudaPos/cudaExtent and make_cudaPos/make_cudaExtent */
#endif
