#include <iostream>
#include <fstream>
#include <vector>

#include "forward_projection.h"
#include "filtered_backprojection.h"
#include "noise.h"
#include "sart.h"
#include "tv.h"
#include "validation.h"
#include "io.h"

// Same pipeline as main.cpp (forward-project -> simulated low-dose noise ->
// FBP / SART+TV), but the phantom is a real CT slice loaded from disk
// instead of a synthetic Shepp-Logan image, so SART+TV can be benchmarked
// against the MONAI denoiser (see comparison/scripts/compare_methods.py) on
// the same ground truth. Note this is a controlled comparison under a
// shared synthetic noise model, not a reproduction of real scanner noise
// statistics -- see README for the caveat.
int main()
{
    const int N = 256;
    const int num_angles = 180;
    const double relaxation = 0.2;
    double nrmse = 0.0;

    std::vector<double> phantom = load_binary("../data/ct_slice_256_for_project2.bin");
    std::vector<double> angles = linspace_angles(num_angles);
    std::vector<double> sinogram_clean = forward_project(phantom, N, angles);

    // Same noise parameters as main.cpp's synthetic-phantom run.
    std::vector<double> sinogram_noisy = add_poisson_noise(sinogram_clean, /*I0=*/1e3, /*mu_scale=*/0.06);

    // --- FBP baseline ---
    std::vector<double> fbp_recon = filtered_back_projection(sinogram_noisy, angles, N);
    double fbp_nrmse = compute_nrmse(fbp_recon, phantom, N);
    std::cout << "FBP NRMSE (noisy): " << fbp_nrmse << std::endl;

    // --- SART + TV ---
    std::vector<double> sart_tv_recon(N * N, 0);

    const double tv_alpha = 0.02;
    const double tv_epsilon = 1e-8;
    const int tv_steps_per_sweep = 3;
    const int num_sweeps = 15;

    std::vector<double> sart_tv_recon_best;
    int sart_tv_best_sweep = -1;
    double sart_tv_best_nrmse = 1e300;

    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        sart_iteration(sart_tv_recon, N, sinogram_noisy, angles, relaxation);

        for (int tv_step = 0; tv_step < tv_steps_per_sweep; tv_step++) {
            tv_denoise_step(sart_tv_recon, N, tv_alpha, tv_epsilon);
        }

        nrmse = compute_nrmse(sart_tv_recon, phantom, N);
        std::cout << "sweep " << sweep << ": SART+TV NRMSE = " << nrmse << std::endl;
        if (nrmse < sart_tv_best_nrmse) {
            sart_tv_best_nrmse = nrmse;
            sart_tv_best_sweep = sweep;
            sart_tv_recon_best = sart_tv_recon;
        }
    }
    std::cout << "best SART+TV sweep: " << sart_tv_best_sweep
               << ", NRMSE: " << sart_tv_best_nrmse << std::endl;

    // --- Results ---
    save_binary("../figures/ct_compare/phantom.bin", phantom);
    save_binary("../figures/ct_compare/fbp_recon.bin", fbp_recon);
    save_binary("../figures/ct_compare/sart_tv_recon_best.bin", sart_tv_recon_best);

    return 0;
}
