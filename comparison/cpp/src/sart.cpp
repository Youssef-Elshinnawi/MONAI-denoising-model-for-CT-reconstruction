#include "sart.h"
#include "forward_projection.h"
#include "back_projection.h"

// Ray-length normalization for a single angle: forward-projecting an
// all-ones image gives, per ray, how much image mass it passes through.
static std::vector<double> compute_row_sum(int N, double theta)
{
    std::vector<double> ones_image(N * N, 1.0);
    std::vector<double> theta_vec = {theta};
    return forward_project(ones_image, N, theta_vec);
}

// Per-pixel normalization for a single angle: back-projecting an
// all-ones sinogram row gives, per pixel, how much this angle's rays
// touch it. back_project divides by num_angles internally, a no-op here
// since theta_vec has length 1.
static std::vector<double> compute_col_sum(int N, double theta)
{
    std::vector<double> ones_row(N, 1.0);
    std::vector<double> theta_vec = {theta};
    return back_project(ones_row, theta_vec, N);
}

void sart_iteration(
    std::vector<double>& image,
    int N,
    const std::vector<double>& sinogram,
    const std::vector<double>& angles,
    double relaxation
)
{
    int num_angles = angles.size();

    for (int i = 0; i < num_angles; i++) {
        double theta = angles[i];
        std::vector<double> theta_vec = {theta};

        std::vector<double> p_est = forward_project(image, N, theta_vec);
        std::vector<double> measured(sinogram.begin() + i * N, sinogram.begin() + i * N + N);
        std::vector<double> row_sum = compute_row_sum(N, theta);

        // Data-consistency residual per ray, normalized by ray length.
        // A zero row_sum means the ray misses the image entirely.
        std::vector<double> residual(N);
        for (int s = 0; s < N; s++) {
            if (row_sum[s] == 0) {
                residual[s] = 0;
            } else {
                residual[s] = (measured[s] - p_est[s]) / row_sum[s];
            }
        }

        std::vector<double> correction = back_project(residual, theta_vec, N);
        std::vector<double> col_sum = compute_col_sum(N, theta);

        for (int j = 0; j < N * N; j++) {
            if (col_sum[j] == 0) {
                continue;
            }
            image[j] += relaxation * correction[j] / col_sum[j];
        }
    }
}
