#include "rotate.h"
#include <cmath>

std::vector<double> rotate_image(
    const std::vector<double>& image,
    int N,
    double theta_degrees
)
{
    std::vector<double> output(N * N, 0.0);

    const double PI = std::acos(-1.0);
    // Row index increasing downward is a flipped y-axis relative to
    // standard math convention (y increases upward), so a textbook CCW
    // rotation matrix here would run opposite to scipy's convention unless
    // we negate the angle to compensate.
    double theta = -theta_degrees * PI / 180.0;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {

            double x = -1.0 + 2.0*col/(N-1);
            double y = -1.0 + 2.0*row/(N-1);

            // Inverse mapping: apply the rotation to the output coordinate
            // to find where it came from in the source image.
            double x_src = x*std::cos(theta) + y*std::sin(theta);
            double y_src = -x*std::sin(theta) + y*std::cos(theta);

            double col_src = (x_src + 1.0) * (N-1) / 2.0;
            double row_src = (y_src + 1.0) * (N-1) / 2.0;

            // Whole-pixel bounds check (matches scipy's 'constant' mode):
            // if the source coordinate itself falls outside [0, N-1], the
            // entire output pixel is the fill value (0) -- no partial
            // blending using whichever individual corners happen to be
            // in-bounds. This only differs from per-corner clamping when
            // the signal has real nonzero content right at the image edge.
            if (row_src < 0.0 || row_src > N - 1 || col_src < 0.0 || col_src > N - 1) {
                output[row * N + col] = 0.0;
                continue;
            }

            int col0 = std::floor(col_src);
            int row0 = std::floor(row_src);
            int col1 = col0 + 1;
            int row1 = row0 + 1;
            double fx = col_src - col0;
            double fy = row_src - row0;

            // Safe lookup: any corner outside [0, N-1] contributes 0.
            auto pixel = [&](int r, int c) -> double {
                if (r < 0 || r >= N || c < 0 || c >= N) return 0.0;
                return image[r * N + c];
            };

            double value =
                (1 - fx) * (1 - fy) * pixel(row0, col0) +
                fx * (1 - fy)       * pixel(row0, col1) +
                (1 - fx) * fy       * pixel(row1, col0) +
                fx * fy             * pixel(row1, col1);

            output[row * N + col] = value;
        }
    }

    return output;
}
