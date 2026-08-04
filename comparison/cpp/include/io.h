#pragma once

#include <vector>
#include <string>

// Writes a flat N*N double array as raw binary (no header) so Python's
// np.fromfile(path, dtype=np.float64) can read it back for visualization.
void save_binary(const std::string& path, const std::vector<double>& data);

// Reads a flat N*N double array back from raw binary (the counterpart to
// save_binary / Python's ndarray.tofile()). Throws if the file can't be
// opened or its size isn't an exact multiple of sizeof(double).
std::vector<double> load_binary(const std::string& path);
