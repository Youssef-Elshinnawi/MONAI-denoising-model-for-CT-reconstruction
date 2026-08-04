#include <iostream>
#include <fstream>
#include <vector>

#include "phantom.h"
#include "forward_projection.h"
#include "filtered_backprojection.h"
#include "noise.h"
#include "sart.h"
#include "tv.h"
#include "validation.h"
#include "io.h"

int main()
{
    const int N = 128;
    const int num_angles = 180;
    const double relaxation = 0.2;
    double nrmse = 0.0;

    std::vector<double> phantom = build_phantom(N, SHEPP_LOGAN_ELLIPSES);
    std::vector<double> angles = linspace_angles(num_angles);

    std::vector<double> sinogram_clean = forward_project(phantom, N, angles);

    // Simulated low-dose acquisition.
    std::vector<double> sinogram_noisy = add_poisson_noise(sinogram_clean, /*I0=*/1e3, /*mu_scale=*/0.06);

    // --- FBP baseline ---
    std::vector<double> fbp_recon = filtered_back_projection(sinogram_noisy, angles, N);
    double fbp_nrmse = compute_nrmse(fbp_recon, phantom, N);
    std::cout << "FBP NRMSE (noisy): " << fbp_nrmse << std::endl;

    // --- Plain SART ---
    std::vector<double> sart_recon(N * N, 0);

    const int num_sweeps = 15;
    std::vector<double> nrmse_sart_history;
    std::vector<double> sart_recon_best;
    int sart_best_sweep = -1;
    double sart_best_nrmse = 1e300;
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        sart_iteration(sart_recon, N, sinogram_noisy, angles, relaxation);
        nrmse = compute_nrmse(sart_recon, phantom, N);
        std::cout << "sweep " << sweep << ": NRMSE = " << nrmse << std::endl;
        nrmse_sart_history.push_back(nrmse);
        if (nrmse < sart_best_nrmse) {
            sart_best_nrmse = nrmse;
            sart_best_sweep = sweep;
            sart_recon_best = sart_recon;
        }
    }

    // --- SART + TV ---
    std::vector<double> sart_tv_recon(N * N, 0);

    // tv_alpha is applied tv_steps_per_sweep times per sweep on top of
    // SART's own update, so it needs to be much smaller than relaxation.
    const double tv_alpha = 0.02;
    const double tv_epsilon = 1e-8;
    const int tv_steps_per_sweep = 3;
    double nrmse_TV_SART;

    std::vector<double> nrmse_sart_tv_history;
    std::vector<double> sart_tv_recon_best;
    int sart_tv_best_sweep = -1;
    double sart_tv_best_nrmse = 1e300;
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        sart_iteration(sart_tv_recon, N, sinogram_noisy, angles, relaxation);

        for (int tv_step = 0; tv_step < tv_steps_per_sweep; tv_step++) {
            tv_denoise_step(sart_tv_recon, N, tv_alpha, tv_epsilon);
        }

        nrmse_TV_SART = compute_nrmse(sart_tv_recon, phantom, N);
        std::cout << "sweep " << sweep << ": SART+TV NRMSE = " << nrmse_TV_SART << std::endl;
        nrmse_sart_tv_history.push_back(nrmse_TV_SART);
        if (nrmse_TV_SART < sart_tv_best_nrmse) {
            sart_tv_best_nrmse = nrmse_TV_SART;
            sart_tv_best_sweep = sweep;
            sart_tv_recon_best = sart_tv_recon;
        }
    }

    // --- Results ---
    save_binary("../figures/phantom.bin", phantom);
    save_binary("../figures/fbp_recon.bin", fbp_recon);
    save_binary("../figures/sart_recon.bin", sart_recon);
    save_binary("../figures/sart_tv_recon.bin", sart_tv_recon);
    save_binary("../figures/sart_recon_best.bin", sart_recon_best);
    save_binary("../figures/sart_tv_recon_best.bin", sart_tv_recon_best);

    std::ofstream nrmse_file("../figures/nrmse_history.csv");
    nrmse_file << "fbp_nrmse," << fbp_nrmse << "\n";
    nrmse_file << "sart_best_sweep,sart_best_nrmse," << sart_best_sweep << "," << sart_best_nrmse << "\n";
    nrmse_file << "sart_tv_best_sweep,sart_tv_best_nrmse," << sart_tv_best_sweep << "," << sart_tv_best_nrmse << "\n";
    nrmse_file << "sweep,sart_nrmse,sart_tv_nrmse\n";
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        nrmse_file << sweep << "," << nrmse_sart_history[sweep] << "," << nrmse_sart_tv_history[sweep] << "\n";
    }

    return 0;
}
