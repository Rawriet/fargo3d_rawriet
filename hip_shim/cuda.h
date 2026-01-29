#pragma once
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>
#include <string.h>

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

#ifndef cudaMemcpyHostToHost
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#endif

#ifndef cudaMemcpy2D
#define cudaMemcpy2D hipMemcpy2D
#endif

#ifndef __CUDA_POS_DEFINED__
#define __CUDA_POS_DEFINED__
typedef struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
} cudaPos;

static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) {
  cudaPos p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}
#endif

#ifndef __CUDA_EXTENT_DEFINED__
#define __CUDA_EXTENT_DEFINED__
typedef struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
} cudaExtent;

static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
  cudaExtent e;
  e.width  = w;
  e.height = h;
  e.depth  = d;
  return e;
}
#endif

#ifndef __CUDA_MEMCPY3D_PARMS_DEFINED__
#define __CUDA_MEMCPY3D_PARMS_DEFINED__
typedef struct cudaMemcpy3DParms {
  const void* srcArray;
  const void* dstArray;
  cudaPos srcPos;
  cudaPos dstPos;
  cudaPitchedPtr srcPtr;
  cudaPitchedPtr dstPtr;
  cudaExtent extent;
  hipMemcpyKind kind;
} cudaMemcpy3DParms;
#endif

#ifndef __CUDA_MEMCPY3D_DEFINED__
#define __CUDA_MEMCPY3D_DEFINED__
static inline hipError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) {
  hipMemcpy3DParms hp;
  hp.srcArray = (hipArray_t)p->srcArray;
  hp.dstArray = (hipArray_t)p->dstArray;
  hp.srcPos.x = p->srcPos.x;
  hp.srcPos.y = p->srcPos.y;
  hp.srcPos.z = p->srcPos.z;
  hp.dstPos.x = p->dstPos.x;
  hp.dstPos.y = p->dstPos.y;
  hp.dstPos.z = p->dstPos.z;
  hp.srcPtr.ptr   = p->srcPtr.ptr;
  hp.srcPtr.pitch = p->srcPtr.pitch;
  hp.srcPtr.xsize = p->srcPtr.xsize;
  hp.srcPtr.ysize = p->srcPtr.ysize;
  hp.dstPtr.ptr   = p->dstPtr.ptr;
  hp.dstPtr.pitch = p->dstPtr.pitch;
  hp.dstPtr.xsize = p->dstPtr.xsize;
  hp.dstPtr.ysize = p->dstPtr.ysize;
  hp.extent.width  = p->extent.width;
  hp.extent.height = p->extent.height;
  hp.extent.depth  = p->extent.depth;
  hp.kind     = p->kind;
  return hipMemcpy3D(&hp);
}
#endif

#ifndef __CUDA_DEVICE_PROP_DEFINED__
#define __CUDA_DEVICE_PROP_DEFINED__
typedef struct cudaDeviceProp {
  char   name[256];
  size_t totalGlobalMem;
  int    major;
  int    minor;
} cudaDeviceProp;
#endif

#ifndef __CUDA_DEVICE_FUNCS_DEFINED__
#define __CUDA_DEVICE_FUNCS_DEFINED__
typedef hipError_t cudaError_t;

static inline hipError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  hipInit(0);
  hipDeviceProp_t hp;
  hipError_t err = hipGetDeviceProperties(&hp, device);
  if (err != hipSuccess) return err;
  strncpy(prop->name, hp.name, sizeof(prop->name) - 1);
  prop->name[sizeof(prop->name) - 1] = '\0';
  prop->totalGlobalMem = hp.totalGlobalMem;
  prop->major = hp.major;
  prop->minor = hp.minor;
  return hipSuccess;
}

static inline hipError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop) {
  hipInit(0);
  int count = 0;
  hipError_t ecount = hipGetDeviceCount(&count);
  if (ecount != hipSuccess || count <= 0) {
    return ecount != hipSuccess ? ecount : hipErrorNoDevice;
  }
  if (device) *device = 0;
  return hipSuccess;
  hipDeviceProp_t hp;
  memset(&hp, 0, sizeof(hp));
  hp.major = prop ? prop->major : 0;
  hp.minor = prop ? prop->minor : 0;
  return hipChooseDevice(device, &hp);
}

static inline hipError_t cudaSetDevice(int device) {
  return hipSetDevice(device);
}

static inline hipError_t cudaGetDevice(int* device) {
  return hipGetDevice(device);
}

static inline hipError_t cudaDeviceReset(void) {
  return hipDeviceReset();
}

static inline hipError_t cudaDeviceSynchronize(void) {
  return hipDeviceSynchronize();
}

static inline cudaError_t cudaGetLastError(void) {
  return hipGetLastError();
}

static inline const char* cudaGetErrorString(cudaError_t err) {
  return hipGetErrorString(err);
}

static inline hipError_t cudaMalloc(void** p, size_t sz) {
  return hipMalloc(p, sz);
}

static inline hipError_t cudaMemcpy(void* dst, const void* src, size_t sz, hipMemcpyKind kind) {
  return hipMemcpy(dst, src, sz, kind);
}
#endif

#ifndef __CUDA_MALLOC3D_DEFINED__
#define __CUDA_MALLOC3D_DEFINED__
static inline hipError_t cudaMalloc3D(cudaPitchedPtr* p, cudaExtent e) {
  hipPitchedPtr hp;
  hipExtent he;
  he.width  = e.width;
  he.height = e.height;
  he.depth  = e.depth;
  hipError_t err = hipMalloc3D(&hp, he);
  if (err != hipSuccess) return err;
  p->ptr   = hp.ptr;
  p->pitch = hp.pitch;
  p->xsize = hp.xsize;
  p->ysize = hp.ysize;
  return err;
}
#endif
