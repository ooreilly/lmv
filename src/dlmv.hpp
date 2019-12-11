#ifndef DLMV_HPP
#define DLMV_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#ifndef CHECK_BOUNDS
#define CHECK_BOUNDS 1
#endif

static const int use_bounds_check = CHECK_BOUNDS;

#define inbounds(x, y) inbounds_(x, #x, y, #y,  __FILE__, __LINE__)
__device__ int inbounds_(int x, const char *msgx, int y, const char *msgy,
                         const char *file, int line) {
        if (use_bounds_check && !x) {
                printf("%s:%d %s is out of bounds, %s = %d\n", file, line, msgx,
                                msgy, y);
                return 0;
        }
        return 1;
}


#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define cusparseErrCheck(stat) { cusparseErrCheck_((stat), __FILE__, __LINE__); }
void cusparseErrCheck_(cusparseStatus_t stat, const char *file, int line) {
   if (stat != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr, "cuSPARSE Error: %d %s %d\n", stat, file, line);
   }
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if( stat != cudaSuccess) {                                                   
        fprintf(stderr, "CUDA error in %s:%i %s.\n",                          
          file, line, cudaGetErrorString(stat) );            
        fflush(stderr);                                                             
  }
}

#endif
