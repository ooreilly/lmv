#ifndef HLMV_HPP
#define HLMV_HPP
#define VERSION 1.0.0b

#include <stdio.h>
#include <stdlib.h>



static const int LMV_STATUS_SUCCESS = 0;
static const int LMV_MALLOC_FAILURE = 100;
static const int LMV_PARSE_FAILURE = 101;

inline const char *lmvGetErrorString(int stat)
{
        switch (stat) {
        case LMV_STATUS_SUCCESS:
                break;
        case LMV_PARSE_FAILURE:
                return "Failed to parse config";
                break;
        case LMV_MALLOC_FAILURE:
                return "Failed to allocate memory";
                break;
        default:
                return "Unknown error";
                break;
        }
        return "";

}

#define lmvErrCheck(stat) { lmvErrCheck_((stat), __FILE__, __LINE__); }
inline void lmvErrCheck_(int stat, const char *file, int line) {
   if (stat != LMV_STATUS_SUCCESS) {
      fprintf(stderr, "lmv error in %s:%i %s.\n",                          
          file, line, lmvGetErrorString(stat) );            
      exit(1);
   }
}

template<typename T>
class hArray;

template<typename Tv, typename Ti>
class hCSR;

class Dict;


class read
{
public:
 read(const char *filename_, bool verbose_ = false)
 {
        filename = filename_;
        verbose = verbose_;
 }
template<typename T>
operator hArray<T>();

template<typename Tv, typename Ti>
operator hCSR<Tv, Ti>();

operator Dict();

private:
        const char *filename;
        bool verbose;
};

#define dump(x) dump_(x, #x)


#endif
