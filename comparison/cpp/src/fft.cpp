#include "fft.h"
#include <cmath>

// Radix-2 Cooley-Tukey FFT. Requires x.size() to be a power of 2.
std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& x)
{
    int N = x.size();

    if (N == 1) {
        return x;
    }

    std::vector<std::complex<double>> even, odd;
    for (int i = 0; i < N/2; i++) {
        even.push_back(x[2*i]);
        odd.push_back(x[2*i+1]);
    }

    std::vector<std::complex<double>> E = fft(even);
    std::vector<std::complex<double>> O = fft(odd);
    std::vector<std::complex<double>> X(N);

    const double PI = std::acos(-1.0);

    // Butterfly: one twiddle multiply produces both X[k] and X[k+N/2].
    for (int k = 0; k < N / 2; k++) {
        std::complex<double> W(std::cos(-2.0 * PI * k / N), std::sin(-2.0 * PI * k / N));
        X[k] = E[k] + W * O[k];
        X[k + N/2] = E[k] - W * O[k];
    }

    return X;
}

// Inverse FFT via the conjugate trick, so it can reuse fft() directly:
// ifft(X) = conj(fft(conj(X))) / N.
std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& X)
{
    int N = X.size();

    std::vector<std::complex<double>> conj_X(N);
    for (int i = 0; i < N; i++) {
        conj_X[i] = std::conj(X[i]);
    }

    std::vector<std::complex<double>> Y = fft(conj_X);

    std::vector<std::complex<double>> x(N);
    for (int i = 0; i < N; i++) {
        x[i] = std::conj(Y[i]) / double(N);
    }

    return x;
}
