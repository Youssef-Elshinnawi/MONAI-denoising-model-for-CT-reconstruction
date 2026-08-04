#pragma once

#include <vector>

// One explicit steepest-descent step on the isotropic TV functional,
// applied in place: image -= alpha * grad_TV(image).
// epsilon avoids division by zero in flat (zero-gradient) regions.
void tv_denoise_step(
    std::vector<double>& image,
    int N,
    double alpha,
    double epsilon
);
