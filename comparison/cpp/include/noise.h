#pragma once

#include <vector>

// Adds i.i.d. Gaussian noise with standard deviation `sigma`.
std::vector<double> add_gaussian_noise(
    const std::vector<double>& sinogram,
    double sigma
);

// Low-dose CT noise model: Beer-Lambert attenuation + Poisson photon
// statistics. Lower I0 (incident photon count) means lower dose and
// more noise.
std::vector<double> add_poisson_noise(
    const std::vector<double>& sinogram,
    double I0,
    double mu_scale
);
