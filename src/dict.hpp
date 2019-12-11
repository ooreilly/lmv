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

int write(const char *filename, Dict& d, int verbose = 0)
{
        FILE *fh = fopen(filename, "w");

        for (auto it = d.map.begin(); it != d.map.end(); ++it) {
                fprintf(fh, "%s=%s\n", it->first.c_str(), it->second.c_str());
        }
        fclose(fh);
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
                fprintf(stderr, "Failed to open: %s.\n", filename);
        }
        Dict d(map);
        return d;
}

#endif

