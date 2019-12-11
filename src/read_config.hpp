#ifndef READ_CONFIG_HPP
#define READ_CONFIG_HPP

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>

int read_config(const char *filename, std::map<std::string, std::string> &config)
{
        std::ifstream cFile (filename);
        if (cFile.is_open())
        {
            std::string line;
            while(getline(cFile, line)){
                line.erase(std::remove_if(line.begin(), line.end(), isspace),
                                     line.end());
                if(line[0] == '#' || line.empty())
                    continue;
                auto delimiterPos = line.find("=");
                auto key = line.substr(0, delimiterPos);
                auto value = line.substr(delimiterPos + 1);
                config.insert(std::pair<std::string, std::string>(key, value));
        }
        }
        else {
            std::cerr << "Couldn't open config file for reading.\n";
            return DCSRMV_PARSE_FAILURE;
        }
        return DCSRMV_STATUS_SUCCESS;
}


#endif
