#pragma once

#include <vector>
#include <complex>

// Direct O(N^2) Discrete Fourier Transform. Ground-truth reference for
// validating the Cooley-Tukey FFT (Stage 6).
std::vector<std::complex<double>> dft(const std::vector<double>& x);
