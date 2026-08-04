#include "forward_projection.h"
#include "rotate.h"

std::vector<double> linspace_angles(int num_angles)
{
    std::vector<double> angles(num_angles);
    for (int i = 0; i < num_angles; i++) {
        angles[i] = i * (180.0 / num_angles);
    }

    return angles;
}

std::vector<double> forward_project(
    const std::vector<double>& image,
    int N,
    const std::vector<double>& angles
)
{
    int num_angles = angles.size();

    std::vector<double> sinogram(num_angles * N, 0.0);

    for (int i = 0; i < num_angles; i++) {
        double theta = angles[i];
        std::vector<double> rotated = rotate_image(image, N, theta);

        // Column sums give one projection value per detector position.
        for (int col = 0; col < N; col++) {
            double sum = 0.0;
            for (int row = 0; row < N; row++) {
                sum += rotated[row * N + col];
            }

            sinogram[i * N + col] = sum;
        }
    }

    return sinogram;
}
