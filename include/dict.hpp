#ifndef DICT_CPP
#define DICT_CPP

#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>

#include <hlmv.hpp>


template <typename T>
class hArray;

class parse {

        std::string val;

        public:
        parse(std::string val_) { val = val_; }

        operator int() { return std::stoi(val); } 
        operator float() { return std::stof(val); } 
        operator double() { return std::stod(val); } 
        operator std::string() { return val; }
        operator const char*() { return val.c_str(); }
        template <typename T>
        operator hArray<T>() { 

                hArray<T> a(2);
                a[0] = parse("1");
                return a;
        } 
};

class Dict
{
public:
        std::map<std::string, std::string> map;

        Dict(std::map<std::string, std::string>& m_ ) {
                map = m_;
        }

        parse operator [](const std::string& key){ 
                return parse(map[key]);
        }

};

int write(const std::string& filename, Dict& d, int verbose = 0)
{
        FILE *fh = fopen(filename.c_str(), "w");

        if (!fh) return LMV_WRITE_FAILURE; 

        for (auto it = d.map.begin(); it != d.map.end(); ++it) {
                fprintf(fh, "%s=%s\n", it->first.c_str(), it->second.c_str());
        }
        fclose(fh);

        return LMV_STATUS_SUCCESS;
}

read::operator Dict() {
        std::map<std::string, std::string> map;
        std::ifstream cFile (filename);
        if (cFile.is_open()) {
                std::string line;
                while (getline(cFile, line)) {
                        line.erase(
                            std::remove_if(line.begin(), line.end(), isspace),
                            line.end());
                        if (line[0] == '#' || line.empty()) continue;
                        auto delimiterPos = line.find("=");
                        auto key = line.substr(0, delimiterPos);
                        auto value = line.substr(delimiterPos + 1);
                        map[key] = value;
                }
        } else {
                fprintf(stderr, "Failed to open: %s.\n", filename.c_str());
        }
        Dict d(map);
        return d;
}

void dump_(Dict& x, const char *name, FILE *stream=NULL) {
        if (!stream) stream = stdout;
        fprintf(stream, "%s = {", name);
        for (auto it = x.map.begin(); it != x.map.end(); ++it) {
                fprintf(stream, " %s = %s", it->first.c_str(), it->second.c_str());
        }
        fprintf(stream, "}\n");
}

#endif

