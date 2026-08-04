#include "io.h"
#include <fstream>
#include <stdexcept>

void save_binary(const std::string& path, const std::vector<double>& data)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for writing: " + path);
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}

std::vector<double> load_binary(const std::string& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Could not open file for reading: " + path);
    }

    std::streamsize size = in.tellg();
    if (size % sizeof(double) != 0) {
        throw std::runtime_error("File size is not a multiple of sizeof(double): " + path);
    }

    in.seekg(0, std::ios::beg);
    std::vector<double> data(size / sizeof(double));
    in.read(reinterpret_cast<char*>(data.data()), size);

    return data;
}
