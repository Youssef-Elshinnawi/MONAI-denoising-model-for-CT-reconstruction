#pragma once

#include <vector>

// Returns evenly spaced angles in degrees, [0, 180), length num_angles.
std::vector<double> linspace_angles(int num_angles);

// sinogram is a flat (num_angles * N) array, row-major: sinogram[i*N + s]
std::vector<double> forward_project(
    const std::vector<double>& image,
    int N,
    const std::vector<double>& angles
);
