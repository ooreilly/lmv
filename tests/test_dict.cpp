#include <hlmv.hpp>
#include <assert.h>
#include <test.hpp>
#include <dict.hpp>
#include <string>
#include <stdio.h>

void op_test() { }

int main(int arc, char **argv)
{

         std::map<std::string, std::string> m;
         m["i"] = "1";
         m["f"] = "1.0";
         m["d"] = "1.0";

        {
                Test t = Test("Parse");
                Dict dict(m);
                int i = 1;
                int di = dict["i"];
                float f = 1.0;
                float df = dict["f"];
                float d = 1.0;
                float dd = dict["d"];
                test(t, di == i);
                test(t, df == f);
                test(t, dd == d);
        }

        {
                Test t = Test("Read/Write");
                std::string filename = "test.txt";
                Dict d1(m);
                write(filename, d1);
                Dict d2 = read(filename);
                int i1 = d1["i"];
                float f1 = d1["f"];
                double g1 = d1["d"];

                int i2 = d2["i"];
                float f2 = d2["f"];
                double g2 = d2["d"];
                test(t, i1 == i2);
                test(t, f1 == f2);
                test(t, g1 == g2);
                remove(filename.c_str());
        }

        return test_status();
}
