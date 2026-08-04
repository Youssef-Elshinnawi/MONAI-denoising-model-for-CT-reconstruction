#include "ramp_filter.h"
#include "fft.h"
#include <complex>

// FFT-bin frequency axis matching np.fft.fftfreq's ordering: 0, positive
// frequencies, then negative frequencies.
std::vector<double> fft_freqs(int N)
{
    std::vector<double> freqs(N);

    for (int k = 0; k < N; k++) {
        if (k < N/2) {
            freqs[k] = k / double(N);
        } else {
            freqs[k] = (k-N) / double(N);
        }
    }

    return freqs;
}

std::vector<double> ramp_filter(
    const std::vector<double>& sinogram,
    int num_angles,
    int N
)
{
    std::vector<double> freqs = fft_freqs(N);
    std::vector<double> filtered(num_angles * N);

    for (int i = 0; i < num_angles; i++) {
        std::vector<std::complex<double>> row(N);
        for (int n = 0; n < N; n++) {
            row[n] = std::complex<double>(sinogram[i*N + n], 0);
        }

        std::vector<std::complex<double>> spectrum = fft(row);
        for (int k = 0; k < N; k++) {
            spectrum[k] *= std::fabs(freqs[k]);
        }

        std::vector<std::complex<double>> filtered_row = ifft(spectrum);
        for (int n = 0; n < N; n++) {
            filtered[i*N + n] = filtered_row[n].real();
        }
    }

    return filtered;
}
