#ifndef DCSR_HPP
#define DCSR_HPP

template<typename Tv, typename Ti>
class hCSR;

template<typename Tv, typename Ti>
class dCSR {

  public:

    size_t m;     // Number of rows 
    size_t n;     // Number of columns
    size_t nnz;   // Number of non-zeros
    Tv *val;
    Ti *row;
    Ti *col;

    cusparseMatDescr_t descrA;

    dCSR(){m = 0; n = 0; nnz = 0; val = NULL; row = NULL; col = NULL;}

    dCSR(size_t m_, size_t n_, size_t nnz_) { init(m_, n_, nnz_); }

    ~dCSR(){
            if( val != NULL) cudaErrCheck(cudaFree(val));
            if( row != NULL) cudaErrCheck(cudaFree(row));
            if( col != NULL) cudaErrCheck(cudaFree(col));
        }

  private:
    void init(size_t m_, size_t n_, size_t nnz_) {
        m = m_;
        n = n_;
        nnz = nnz_;
        cudaErrCheck(cudaMalloc((void**)&val, sizeof(Tv) * nnz));
        cudaErrCheck(cudaMalloc((void**)&row, sizeof(Ti) * (m + 1)));
        cudaErrCheck(cudaMalloc((void**)&col, sizeof(Ti) * nnz));
        
        cusparseErrCheck(cusparseCreateMatDescr(&descrA));
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
        }
};

template <typename Tv, typename Ti>
inline dCSR<Tv, Ti> htod(const hCSR<Tv, Ti>& src)
{
    dCSR<Tv, Ti> dest(src.m, src.n, src.nnz);    
    cudaErrCheck(cudaMemcpy(dest.val, src.val, 
                 sizeof(Tv) * (src.nnz), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(dest.row, src.row, 
                 sizeof(Ti) * (src.m+1), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(dest.col, src.col, 
                 sizeof(Ti) * (src.nnz), cudaMemcpyHostToDevice));
    return dest; 
}

template <typename Tv, typename Ti>
inline hCSR<Tv, Ti> dtoh(const dCSR<Tv, Ti>& src) {
    hCSR<Tv, Ti> dest(src.m, src.n, src.nnz);
    cudaErrCheck(cudaMemcpy(dest.val, src.val, 
                 sizeof(Tv) * (src.nnz), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(dest.row, src.row, 
                 sizeof(Ti) * (src.m+1), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(dest.col, src.col, 
                 sizeof(Ti) * (src.nnz), cudaMemcpyDeviceToHost));
    return dest;
}

void mv(cusparseHandle_t cusparseH, dArray<double>& y,
        const dCSR<double, int>& mat, dArray<double>& x, const double a = 1.0,
        const double b = 0.0) {
        cusparseErrCheck(cusparseDcsrmv_mp(
            cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, mat.m, mat.n, mat.nnz,
            &a, mat.descrA, mat.val, mat.row, mat.col, x.x, &b, y.x));
}

void mv(cusparseHandle_t cusparseH, dArray<float>& y,
        const dCSR<float, int>& mat, dArray<float>& x, const float a = 1.0,
        const float b = 0.0) {
        cusparseErrCheck(cusparseScsrmv_mp(
            cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, mat.m, mat.n, mat.nnz,
            &a, mat.descrA, mat.val, mat.row, mat.col, x.x, &b, y.x));
}

#endif
