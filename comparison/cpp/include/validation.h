#pragma once

#include <vector>

// Returns NRMSE between reconstruction and phantom (both flat N*N arrays).
// Lower is better. Rescales reconstruction to phantom's value range first.
double compute_nrmse(
    const std::vector<double>& reconstruction,
    const std::vector<double>& phantom,
    int N
);
