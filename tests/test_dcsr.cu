#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <assert.h>
#include <map>

#include <test.hpp>
#include <hlmv.hpp>
#include <dlmv.hpp>
#include <hArray.hpp>
#include <dArray.hpp>
#include <hcsr.hpp>
#include <dcsr.hpp>

int main()
{
        {
                Test t("Init");
                size_t m = 10;
                size_t n = 10;
                size_t nnz = 20;
                hCSR<double, int> A2(m, n, nnz);
                dCSR<double, int> dA2(m, n, nnz);
                test(t, A2.m == m);
                test(t, A2.n == n);
                test(t, A2.nnz == nnz);
                test(t, dA2.m == m);
                test(t, dA2.n == n);
                test(t, dA2.nnz == nnz);
        }



        {
                Test t("Transfer to host/device");
                size_t m = 10;
                size_t n = 10;
                size_t nnz = 20;
                hCSR<double, int> A1(m, n, nnz);
                A1.row[0] = 1;
                A1.col[0] = 1;
                A1.val[0] = 1.0;

                dCSR<double, int> dA1 = htod(A1);

                hCSR<double, int> A2 = dtoh(dA1);
                test(t, A1.row[0] == A2.row[0]);
                test(t, A1.col[0] == A2.col[0]);
                test(t, A1.val[0] == A2.val[0]);
        }

        {
                Test t("Matrix vector multiplication (double)");

                cusparseHandle_t cusparseH;
                cusparseErrCheck(cusparseCreate(&cusparseH));
                hCSR<double, int> A1(1, 10, 1);
                A1.row[0] = 0;
                A1.row[1] = 1;
                A1.col[0] = 0;
                A1.val[0] = 1.0;
                dCSR<double, int> dA = htod(A1);
                hArray<double> x(A1.n);
                x[0] = -1.0;
                hArray<double> y(A1.m);
                dArray<double> dx = htod(x);
                dArray<double> dy = htod(y);

                mv(cusparseH, dy, dA, dx);

                hArray<double> y2 = dtoh(dy);
                test(t, y2[0] == x[0]);
                cusparseErrCheck(cusparseDestroy(cusparseH));
        }

        {
                Test t("Matrix vector multiplication (float)");

                cusparseHandle_t cusparseH;
                cusparseErrCheck(cusparseCreate(&cusparseH));
                hCSR<float, int> A1(1, 10, 1);
                A1.row[0] = 0;
                A1.row[1] = 1;
                A1.col[0] = 0;
                A1.val[0] = 1.0;
                dCSR<float, int> dA = htod(A1);
                hArray<float> x(A1.n);
                x[0] = -1.0;
                hArray<float> y(A1.m);
                dArray<float> dx = htod(x);
                dArray<float> dy = htod(y);

                mv(cusparseH, dy, dA, dx);

                hArray<float> y2 = dtoh(dy);
                test(t, y2[0] == x[0]);
                cusparseErrCheck(cusparseDestroy(cusparseH));
        }

        return test_status();
}
