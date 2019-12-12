#ifndef HCSR_HPP
#define HCSR_HPP
#include <hlmv.hpp>
template<typename Tv, typename Ti>
class hCSR {
  
  public:

    size_t m;     // Number of rows 
    size_t n;     // Number of columns
    size_t nnz;   // Number of non-zeros
    Tv *val;
    Ti *row;
    Ti *col;

    hCSR(size_t m_, size_t n_, size_t nnz_) {
            assert(m_ > 0);
            assert(n_ > 0);
            assert(nnz_ > 0);
        m = m_;
        n = n_;
        nnz = nnz_;
        val = (Tv*)calloc(sizeof val, nnz_);
        row = (Ti*)calloc(sizeof row, (m + 1));
        col = (Ti*)calloc(sizeof col, nnz);
    }

    ~hCSR() {
            if (val != NULL) free(val);
            if (row != NULL) free(row);
            if (col != NULL) free(col);
    }

};


template<typename Tv, typename Ti>
read::operator hCSR<Tv, Ti>() {
                FILE* fh = fopen(filename, "rb");

                if (!fh)
                  throw std::runtime_error("Could not open " + std::string(filename)); 
                
                size_t m = 0; size_t n = 0; size_t nnz = 0;
                fread(&m, 1, sizeof(size_t), fh);
                fread(&n, 1, sizeof(size_t), fh);
                fread(&nnz, 1, sizeof(size_t), fh);

                assert(m > 0);
                assert(n > 0);
                assert(nnz > 0);
                
                hCSR<Tv, Ti> A(m, n, nnz);
 
                fread(A.row, 1, (m+1)*sizeof(Ti), fh);
                fread(A.col, 1, nnz*sizeof(Ti), fh);
                fread(A.val, 1, nnz*sizeof(Tv), fh);
                fclose(fh);

                if (verbose) {
                  std::cout << "Read sparse matrix <" << m << "," << n;
                  std::cout << "> with " << nnz << " entries: ";
                  std::cout << filename << std::endl;
                }
                return A;
}

template<typename Tv, typename Ti>
void write(const char *filename, hCSR<Tv, Ti>& A, bool verbose = 0) {
                FILE* fh = fopen(filename, "wb");

                if (!fh)
                  throw std::runtime_error("Could not open " +
                                  std::string(filename)); 
                
                fwrite(&A.m, 1, sizeof(size_t), fh);
                fwrite(&A.n, 1, sizeof(size_t), fh);
                fwrite(&A.nnz, 1, sizeof(size_t), fh);
 
                fwrite(&A.row[0], 1, (A.m+1)*sizeof(Ti), fh);
                fwrite(&A.col[0], 1, A.nnz*sizeof(Ti), fh);
                fwrite(&A.val[0], 1, A.nnz*sizeof(Tv), fh);
                fclose(fh);

                if (verbose) {
                  std::cout << "Wrote sparse matrix <" << A.m << "," << A.n;
                  std::cout << "> with " << A.nnz << " entries: ";
                  std::cout << filename << std::endl;
                }
}

#endif
