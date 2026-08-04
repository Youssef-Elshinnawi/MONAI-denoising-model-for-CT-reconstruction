#pragma once

#include <vector>

// Rotates an N x N image by theta_degrees using inverse mapping +
// bilinear interpolation. Out-of-bounds samples are treated as 0.
std::vector<double> rotate_image(
    const std::vector<double>& image,
    int N,
    double theta_degrees
);
