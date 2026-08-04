#pragma once

#include <vector>

// One SART sweep over every angle, updating `image` in place.
// `sinogram` is flat (num_angles * N), same row order as `angles`.
// `relaxation` is the SART step size (lambda); values close to 1.0 can
// diverge since forward/back-projection here are only an approximate
// adjoint pair -- 0.1-0.3 is typically stable.
void sart_iteration(
    std::vector<double>& image,
    int N,
    const std::vector<double>& sinogram,
    const std::vector<double>& angles,
    double relaxation
);
