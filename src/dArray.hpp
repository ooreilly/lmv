#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <hlmv.hpp>
#include <dlmv.hpp>

template <typename T>
class hArray;

template <typename T>
class dArray
{
public:
        size_t size;
        T *x;

        dArray() {size = 0; x = NULL;}
        dArray(size_t in_size) {
                size = in_size;
                cudaErrCheck(cudaMalloc((void**)&x, sizeof x * in_size));
                cudaErrCheck(cudaMemset(x, 0, in_size));
        }
        ~dArray() { if (x != NULL) cudaErrCheck(cudaFree(x)); }

};


template<typename T>
dArray<T> htod(hArray<T>& src)
{
        dArray<T> dest(src.size);
        cudaErrCheck(cudaMemcpy(dest.x, src.x, sizeof(T) * src.size,
                                cudaMemcpyHostToDevice));
        return dest;
}

template<typename T>
hArray<T> dtoh(dArray<T>& src)
{
        hArray<T> dest(src.size);
        cudaErrCheck(cudaMemcpy(dest.x, src.x, sizeof(T) * src.size,
                                cudaMemcpyDeviceToHost));
        return dest;
}


void axpy(cublasHandle_t cublasH, dArray<double>& y, dArray<double>& x, double alpha=1.0, int incx=1, int incy=1)
{
        cublasErrCheck(cublasDaxpy(cublasH, y.size, &alpha, x.x, incx, y.x, incy));
}

void axpy(cublasHandle_t cublasH, dArray<float>& y, dArray<float>& x, float alpha=1.0f, int incx=1, int incy=1)
{
        cublasErrCheck(cublasSaxpy(cublasH, y.size, &alpha, x.x, incx, y.x, incy));
}

template <typename T>
__global__ void d_insert(T* out, const T* u, const int n, const int incfac) {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n) return;
        out[i + incfac * n] = u[i];
}

template <typename T>
void insert(dArray<T>& y, dArray<T>& x, int incfac=0) {
        assert(y.size % x.size == 0);
        assert(x.size * incfac <= y.size);
        dim3 threads (32, 1, 1);
        dim3 blocks ((x.size-1)/threads.x + 1, 1, 1);
        d_insert<<<threads, blocks>>>(y.x, x.x, x.size, incfac);
}

template <typename Tv, typename Ti>
__global__ void d_insert_sparse_left(Tv* out, const Tv* u, const Ti* idx,
                                const int nu, const int nx, const int incfac) {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= nx) return;
        if (inbounds(idx[i] < nu, idx[i])) out[idx[i] + incfac * nu] = u[i];
}

template <typename Tv, typename Ti>
void insert(dArray<Tv>& y, dArray<Ti>& idx, dArray<Tv>& x, int incfac=0) {
        assert(y.size % idx.size == 0);
        assert(idx.size * incfac <= y.size);
        dim3 threads (32, 1, 1);
        dim3 blocks ((x.size-1)/threads.x + 1, 1, 1);
        d_insert_sparse_left<<<threads, blocks>>>(y.x, x.x, idx.x, x.size, idx.size, incfac);
}

template <typename Tv, typename Ti>
__global__ void d_insert_sparse_right(Tv* out, const Tv* u, const Ti* idx,
                                const int nu, const int nx, const int incfac) {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= nx) return;
        if (inbounds(idx[i] < nu, idx[i])) out[i + incfac * nu] = u[idx[i]];
}


template <typename Tv, typename Ti>
void insert(dArray<Tv>& y, dArray<Tv>& x, dArray<Ti>& idx, int incfac=0) {
        assert(y.size % idx.size == 0);
        assert(idx.size * incfac <= y.size);
        dim3 threads (32, 1, 1);
        dim3 blocks ((x.size-1)/threads.x + 1, 1, 1);
        d_insert_sparse_right<<<threads, blocks>>>(y.x, x.x, idx.x, x.size, idx.size, incfac);
}


#endif
