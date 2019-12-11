#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "read_config.hpp"

void test_new_lines(void);
void test_parse(void);

int main(int argc, char **argv)
{
        printf("Testing test_read_config.c\n");
        test_new_lines();
        test_parse();
        return 0;
}

void test_new_lines(void)
{
        char input[] = "file1=test.txt\nfile2=test.txt\n";
        int num_lines = count_lines(input);
        assert(num_lines == 2);
}

void test_parse(void)
{
        char input[] = "file1=test.txt\nfile2=test.txt\n";
        int err = 0;
        char **keys;
        char **values;
        keys = malloc(sizeof(keys)*100);
        for (int i = 0; i < 100; ++i) {
                keys[i] = malloc(sizeof(keys[i])*2048);
        }
        err = read_config(keys, values, input);
        assert(err == 0);
        printf("lala %s \n", keys[0]);
        //assert(strcmp(keys[0], "file1")==0);

}
