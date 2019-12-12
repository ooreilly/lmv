#ifndef HARRAY_HPP
#define HARRAY_HPP

#include <hlmv.hpp>
#include <fstream>
#include <iostream>

template <typename T>
class hArray
{
public:
        size_t size;
        T *x;

        hArray() { size = 0; x = NULL; }
        hArray(size_t in_size) {
                assert(in_size > 0);
                size = in_size;
                x = (T*)calloc(sizeof(T*), in_size);
                if (x == NULL) lmvErrCheck(LMV_MALLOC_FAILURE);
        }
        ~hArray() { if (x != NULL) free(x); }
        T& operator[](size_t elem){ assert(elem < size); return x[elem]; }


};

template<typename T>
void write(const char *filename, hArray<T>& arr, bool verbose = false) {

  FILE* fh = fopen(filename, "wb");

  if (!fh)
    throw std::runtime_error("Could not open " + std::string(filename)); 
  
  assert(arr.size > 0);
  size_t type_size = sizeof(T);
  fwrite(&type_size, 1, sizeof(size_t), fh);
  fwrite(&arr.size, 1, sizeof(size_t), fh);
  fwrite(arr.x, 1, arr.size * sizeof(T), fh);
  fclose(fh);

  if (verbose) { 
    std::cout << "Write array <" << arr.size << "> : ";
    std::cout << filename << std::endl;
  }

}

template<typename T>
read::operator hArray<T>() {

  FILE* fh = fopen(filename.c_str(), "rb");

  if (!fh)
    throw std::runtime_error("Could not open " + std::string(filename)); 

  size_t type_size, size;
  fread(&type_size, 1, sizeof(size_t), fh);
  assert(type_size == sizeof(T));
  fread(&size, 1, sizeof(size_t), fh);
  assert(size > 0);
  hArray<T> arr(size);
  fread(arr.x, 1, size * sizeof(T), fh);
  fclose(fh);

  if (verbose) { 
    std::cout << "Read array <" << arr.size << "> : ";
    std::cout << filename << std::endl;
  }

  return arr;

}

template<typename T>
void dump_(hArray<T>& x, const char *name, FILE *stream=NULL)
{
        if (!stream) stream = stdout;
        fprintf(stream, "%s = [", name);
        for (int i = 0; i < x.size; ++i)
                fprintf(stream, " %g", x[i]);
        fprintf(stream, "]\n");
}


#endif
