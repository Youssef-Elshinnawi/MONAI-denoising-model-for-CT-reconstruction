#pragma once

#include <vector>

// FFT-bin frequency axis matching np.fft.fftfreq(N) ordering.
std::vector<double> fft_freqs(int N);

// Ramp-filters a flat (num_angles * N) sinogram row-by-row (each row
// independent). Returns a filtered sinogram of the same shape.
std::vector<double> ramp_filter(
    const std::vector<double>& sinogram,
    int num_angles,
    int N
);
