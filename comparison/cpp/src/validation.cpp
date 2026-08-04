#include "validation.h"
#include <algorithm>
#include <cmath>

double compute_nrmse(
    const std::vector<double>& reconstruction,
    const std::vector<double>& phantom,
    int N
)
{
    double recon_min = *std::min_element(reconstruction.begin(), reconstruction.end());
    double recon_max = *std::max_element(reconstruction.begin(), reconstruction.end());
    double phantom_min = *std::min_element(phantom.begin(), phantom.end());
    double phantom_max = *std::max_element(phantom.begin(), phantom.end());

    // Back-projection's normalization doesn't guarantee the reconstruction
    // lands on the phantom's original scale, so rescale before comparing.
    std::vector<double> recon_scaled(N * N);
    for (int i = 0; i < N*N; i++) {
        recon_scaled[i] = (reconstruction[i] - recon_min) / (recon_max - recon_min) * (phantom_max - phantom_min) + phantom_min;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < N * N; i++) {
        double d = recon_scaled[i] - phantom[i];
        sum_sq += d * d;
    }
    double mse = sum_sq / (N * N);
    double rmse = std::sqrt(mse);
    double nrmse = rmse / (phantom_max - phantom_min);

    return nrmse;
}
