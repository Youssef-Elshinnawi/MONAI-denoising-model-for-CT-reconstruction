#pragma once

#include <vector>

std::vector<double> filtered_back_projection(
    const std::vector<double>& sinogram,
    const std::vector<double>& angles,
    int N
);
