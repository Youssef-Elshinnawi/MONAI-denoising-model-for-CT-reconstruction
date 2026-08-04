#include "filtered_backprojection.h"
#include "ramp_filter.h"
#include "back_projection.h"

std::vector<double> filtered_back_projection(
    const std::vector<double>& sinogram,
    const std::vector<double>& angles,
    int N
)
{
    int num_angles = angles.size();

    std::vector<double> filtered = ramp_filter(sinogram, num_angles, N);
    std::vector<double> reconstruction = back_project(filtered, angles, N);

    return reconstruction;
}
