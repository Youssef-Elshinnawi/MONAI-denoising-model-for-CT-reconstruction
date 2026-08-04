#include "tv.h"
#include <cmath>

// index helper: row i, col j -> flat index
static inline int idx(int i, int j, int N) { return i * N + j; }

void tv_denoise_step(
    std::vector<double>& image,
    int N,
    double alpha,
    double epsilon
)
{
    // dx(i,j) = x(i,j+1) - x(i,j), 0 at the last column (no right neighbor)
    // dy(i,j) = x(i+1,j) - x(i,j), 0 at the last row (no neighbor below)
    std::vector<double> dx(N * N, 0.0);
    std::vector<double> dy(N * N, 0.0);
    std::vector<double> G(N * N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j < N - 1) {
                dx[idx(i, j, N)] = image[idx(i, j + 1, N)] - image[idx(i, j, N)];
            }
            if (i < N - 1) {
                dy[idx(i, j, N)] = image[idx(i + 1, j, N)] - image[idx(i, j, N)];
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dxij = dx[idx(i, j, N)];
            double dyij = dy[idx(i, j, N)];
            G[idx(i, j, N)] = std::sqrt(dxij * dxij + dyij * dyij + epsilon);
        }
    }

    // grad(i,j) = [-dx(i,j) - dy(i,j)]/G(i,j) + dx(i,j-1)/G(i,j-1) + dy(i-1,j)/G(i-1,j).
    // Computed into a separate array since every term reads neighbors'
    // dx/dy/G, which must stay fixed at their pre-update values.
    std::vector<double> grad(N * N, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double term_self = (-dx[idx(i, j, N)] - dy[idx(i, j, N)]) / G[idx(i, j, N)];

            double term_left = 0.0;
            if (j > 0) {
                term_left = (dx[idx(i, j-1, N)] / G[idx(i, j-1, N)]);
            }

            double term_above = 0.0;
            if (i > 0) {
                term_above = (dy[idx(i-1,j,N)] / G[idx(i-1, j, N)]);
            }

            grad[idx(i, j, N)] = term_self + term_left + term_above;
        }
    }

    // Apply the descent step.
    for (int k = 0; k < N * N; k++) {
        image[k] -= alpha * grad[k];
    }
}
