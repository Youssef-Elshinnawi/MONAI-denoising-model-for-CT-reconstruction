#pragma once

#include <vector>
#include <complex>

// Radix-2 Cooley-Tukey FFT. Requires x.size() to be a power of 2.
std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& x);

// Inverse FFT, implemented by reusing fft() via the conjugate trick.
std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& X);
