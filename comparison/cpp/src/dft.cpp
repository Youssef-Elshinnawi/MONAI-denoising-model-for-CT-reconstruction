#include "dft.h"
#include <cmath>

// Direct O(N^2) DFT, used as a correctness reference for the FFT.
std::vector<std::complex<double>> dft(const std::vector<double>& x)
{
    int N = x.size();
    std::vector<std::complex<double>> X(N);

    const double PI = std::acos(-1.0);

    for (int k = 0; k < N; k++) {
        std::complex<double> sum(0.0, 0.0);

        for (int n = 0; n < N; n++) {
            double angle = -2*PI*k*n/N;
            sum += x[n] * std::complex<double>(std::cos(angle), std::sin(angle));
        }

        X[k] = sum;
    }

    return X;
}
