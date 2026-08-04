#pragma once

#include <vector>

struct Ellipse {
    double value;
    double a;
    double b;
    double x0;
    double y0;
    double angle_deg;
};

void add_ellipse(
    std::vector<double>& image,
    int N,
    double x0,
    double y0,
    double a,
    double b,
    double alpha,
    double v
);

std::vector<double> build_phantom(
    int N,
    const std::vector<Ellipse>& ellipses
);

extern const std::vector<Ellipse> SHEPP_LOGAN_ELLIPSES;
