#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <assert.h>
#include <map>

#include <test.hpp>
#include <hlmv.hpp>
#include <hcsr.hpp>

int main()
{
        {
                Test t("Init");
                size_t m = 10;
                size_t n = 10;
                size_t nnz = 20;
                hCSR<double, int> A2(m, n, nnz);
                test(t, A2.m == m);
                test(t, A2.n == n);
                test(t, A2.nnz == nnz);
        }

        {
                Test t("Read/write");
                hCSR<double, int> A1(10, 10, 10);
                A1.row[0] = 1;
                A1.col[0] = 1;
                A1.val[0] = 1.0;

                write("csr.bin", A1);
                write("csr.bin", A1, 1);
                
                hCSR<double, int> A2 = read("csr.bin");
                hCSR<double, int> A3 = read("csr.bin", 1);

                test(t, A1.row[0] == A2.row[0]);
                test(t, A1.col[0] == A2.col[0]);
                test(t, A1.val[0] == A2.val[0]);

                remove("csr.bin");
        }

        return test_status();
}
