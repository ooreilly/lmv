#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <assert.h>
#include <map>

#include <dlmv.hpp>
#include <hArray.hpp>
#include <dArray.hpp>
#include <test.hpp>

int main()
{
        {
                Test t = Test("Transfer host/device");
                hArray<double> a(10);
                dArray<double> da = htod(a);
                hArray<double> b = dtoh(da);
                test(t, a[0] == b[0]);

                dArray<double> db(10);
                hArray<double> dc = dtoh(db);
                test(t, dc[0] == 0.0);
        }


        {
                Test t = Test("axpy (double)");
                cublasHandle_t cublasH = NULL;
                cublasErrCheck(cublasCreate(&cublasH));
                hArray<double> a(10);
                hArray<double> b(10);
                a[1] = 1.0;
                b[2] = 1.0;
                dArray<double> da = htod(a);
                dArray<double> db = htod(b);

                // db = db + 2.0 * da
                axpy(cublasH, db, da, 2.0);
                hArray<double> c = dtoh(db);
                test(t, c[1] == 2.0 * a[1] + b[0]);
                test(t, c[2] == b[2]);
                cublasErrCheck(cublasDestroy(cublasH));
        }

        {
                Test t = Test("axpy (float)");
                cublasHandle_t cublasH = NULL;
                cublasErrCheck(cublasCreate(&cublasH));
                hArray<float> a(10);
                hArray<float> b(10);
                a[1] = 1.0;
                b[2] = 1.0;
                dArray<float> da = htod(a);
                dArray<float> db = htod(b);

                // db = db + 2.0 * da
                axpy(cublasH, db, da, 2.0);
                hArray<float> c = dtoh(db);
                test(t, c[1] == 2.0 * a[1] + b[0]);
                test(t, c[2] == b[2]);
                cublasErrCheck(cublasDestroy(cublasH));
        }

        {
                Test t = Test("insert (double)");
                hArray<double> a(100);
                hArray<double> b(10);
                b[0] = 1.0;
                dArray<double> da = htod(a);
                dArray<double> db = htod(b);
                insert(da, db, 0);
                insert(da, db, 1);
                hArray<double> c = dtoh(da);
                test(t, c[0] == 1.0);
                test(t, c[10] == 1.0);
        }

        {
                Test t = Test("insert (float)");
                hArray<float> a(100);
                hArray<float> b(10);
                b[0] = 1.0;
                dArray<float> da = htod(a);
                dArray<float> db = htod(b);
                insert(da, db, 0);
                insert(da, db, 1);
                hArray<float> c = dtoh(da);
                test(t, c[0] == 1.0);
                test(t, c[10] == 1.0);
        }

        {
                Test t = Test("insert a[idx] = b (double, int)");
                hArray<double> a(100);
                hArray<double> b(10);
                hArray<int> idx(1);
                b[0] = 1.0;
                idx[0] = 2;
                dArray<double> da = htod(a);
                dArray<double> db = htod(b);
                dArray<int> didx = htod(idx);
                insert(da, didx, db, 0);
                insert(da, didx, db, 1);
                hArray<double> c = dtoh(da);
                test(t, c[2] == 1.0);
                test(t, c[12] == 1.0);
        }

        {
                Test t = Test("insert a = b[idx] (double, int)");
                hArray<double> a(100);
                hArray<double> b(10);
                hArray<int> idx(1);
                b[1] = 1.0;
                idx[0] = 1;
                dArray<double> da = htod(a);
                dArray<double> db = htod(b);
                dArray<int> didx = htod(idx);
                insert(da, db, didx, 0);
                insert(da, db, didx, 1);
                hArray<double> c = dtoh(da);
                test(t, c[0] == 1.0);
                test(t, c[10] == 1.0);
        }

        return test_status();
}

