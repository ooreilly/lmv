#ifndef TEST_HPP
#define TEST_HPP

#include <string>

static int TEST_ERR = 0;
#define test(obj, cond) obj.test_(cond, #cond, __FILE__, __LINE__) 

class Test
{

        public:

        Test(const char* name_)
        {
                pass = 0;
                total = 0;
                name = std::string(name_);

        }

        void test_(bool cond, const char *cond_str, const char *file, int line)
        {
                total++;
                if (cond) {
                        pass++;
                        return;
                } else {
                        TEST_ERR = -1;
                        printf("%s:%d %s failed\n", file, line, cond_str);
                }
        }

        ~Test() {
                int len = name.length();
                std::string s;
                for (int i = 0; i < len; ++i) s = s + " ";
                printf("%s | Pass  Total \n", name.c_str());
                printf("%s   %3d   %3d \n", s.c_str(), pass, total);
        }

        private:
        std::string name;
        int pass;
        int total;

};

int test_status() {return TEST_ERR;}

#endif
