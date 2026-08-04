#pragma once

#include <vector>

// Unfiltered/naive back-projection. sinogram is flat (num_angles * N),
// angles in degrees, same order as sinogram rows. Returns an N x N image.
std::vector<double> back_project(
    const std::vector<double>& sinogram,
    const std::vector<double>& angles,
    int N
);
