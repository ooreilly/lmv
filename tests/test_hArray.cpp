#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <assert.h>
#include <map>

#include <hArray.hpp>
#include <hlmv.hpp>
#include <test.hpp>

int main()
{
        {
                Test t = Test("Init");
                hArray<double> a(10);
                a[0] = 1.0;
                test(t, a[0] == 1.0);
                test(t, a[1] == 0.0);
        }

        {
                Test t = Test("Read/write");
                hArray<double> a(1);
                a[0] = 1.0;
                write("a.bin", a);
                hArray<double> b = read("a.bin");
                test(t, a[0] == b[0]);
                remove("a.bin");
        }

        {
                Test t = Test("Dump array");
                hArray<double> a(10);
                dump(a);
                hArray<float> af(10);
                dump(af);
        }

        return test_status();
}

