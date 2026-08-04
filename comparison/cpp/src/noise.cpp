#include "noise.h"
#include <random>
#include <algorithm>
#include <cmath>

std::vector<double> add_gaussian_noise(
    const std::vector<double>& sinogram,
    double sigma
)
{
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, sigma);

    std::vector<double> noisy(sinogram.size());
    for (size_t i = 0; i < sinogram.size(); i++) {
        noisy[i] = sinogram[i] + dist(rng);
    }

    return noisy;
}

std::vector<double> add_poisson_noise(
    const std::vector<double>& sinogram,
    double I0,
    double mu_scale
)
{
    static std::mt19937 rng(std::random_device{}());

    std::vector<double> noisy(sinogram.size());
    for (size_t i = 0; i < sinogram.size(); i++) {
        double I_clean = I0 * std::exp(-mu_scale * sinogram[i]);
        std::poisson_distribution<long> dist(I_clean);
        long I_noisy = std::max(dist(rng), 1L);

        double p_noisy_scaled = -std::log(static_cast<double>(I_noisy) / I0);
        noisy[i] = p_noisy_scaled / mu_scale;
    }

    return noisy;
}
