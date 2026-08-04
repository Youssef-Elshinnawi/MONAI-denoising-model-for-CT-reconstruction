#include "back_projection.h"
#include "rotate.h"

std::vector<double> back_project(
    const std::vector<double>& sinogram,
    const std::vector<double>& angles,
    int N
)
{
    int num_angles = angles.size();

    std::vector<double> reconstruction(N*N, 0.0);

    for (int i = 0; i < num_angles; i++) {
        double theta = angles[i];

        // Smear row i back across the image: every row of the strip is
        // the same projection, constant along the ray direction.
        std::vector<double> smeared(N * N);
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                smeared[row*N + col] = sinogram[i*N + col];
            }
        }

        // Undo the forward-projection rotation and accumulate.
        std::vector<double> rotated_back = rotate_image(smeared, N, -theta);
        for (int j = 0; j < N*N; j++) {
            reconstruction[j] += rotated_back[j];
        }
    }

    for (int j = 0; j < N*N; j++) {
        reconstruction[j] /= num_angles;
    }

    return reconstruction;
}
