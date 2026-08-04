#include "phantom.h"
#include <cmath>

void add_ellipse(
    std::vector<double>& image,
    int N,
    double x0,
    double y0,
    double a,
    double b,
    double alpha,
    double v
)
{
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {

            int index = row * N + col;

            double x = -1.0 + 2.0 * col / (N - 1);
            double y = -1.0 + 2.0 * row / (N - 1);

            // Shift so the ellipse center is the origin, then rotate by
            // -alpha into the ellipse's own axis-aligned frame.
            double dx = x - x0;
            double dy = y - y0;
            double x_rot = dx * std::cos(alpha) + dy * std::sin(alpha);
            double y_rot = -dx * std::sin(alpha) + dy * std::cos(alpha);

            double ellipse_value =
                (x_rot * x_rot) / (a * a) +
                (y_rot * y_rot) / (b * b);

            if (ellipse_value <= 1.0) {
                image[index] += v;
            }
        }
    }
}

std::vector<double> build_phantom(
    int N,
    const std::vector<Ellipse>& ellipses
)
{
    std::vector<double> image(N * N, 0.0);
    const double PI = std::acos(-1.0);

    for (const Ellipse& e : ellipses) {
        double alpha = e.angle_deg * PI / 180.0;

        add_ellipse(
            image,
            N,
            e.x0,
            e.y0,
            e.a,
            e.b,
            alpha,
            e.value
        );
    }

    return image;
}

// Same table as Python's SHEPP_LOGAN_ELLIPSES (shepp_logan.py)
const std::vector<Ellipse> SHEPP_LOGAN_ELLIPSES = {
    {1.0,   .69,   .92,   0,      0,      0},
    {-.8,   .6624, .8740, 0,     -.0184,  0},
    {-.2,   .1100, .3100, .22,    0,    -18},
    {-.2,   .1600, .4100, -.22,   0,     18},
    {.1,    .2100, .2500, 0,      .35,    0},
    {.1,    .0460, .0460, 0,      .1,     0},
    {.1,    .0460, .0460, 0,     -.1,     0},
    {.1,    .0460, .0230, -.08,  -.605,   0},
    {.1,    .0230, .0230, 0,     -.606,   0},
    {.1,    .0230, .0460, .06,   -.605,   0},
};
