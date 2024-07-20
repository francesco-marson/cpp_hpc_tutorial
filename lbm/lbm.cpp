#include <iostream>
#include <vector>
#include <cmath>
#include <mdspan>
#include <ranges>
#include <execution>
#include <numeric>
#include <array>
#include <cartesian_product.hpp> // Brings C++23 std::views::cartesian_product to C++20

// Data structure for the grid
struct Grid {
  int nx, ny;
  double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
  std::array<int, 9> cx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  std::array<int, 9> cy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  std::vector<double> f, feq, rho, u, v;

  Grid(int nx, int ny) : nx(nx), ny(ny),
                         f(nx * ny * 9, 1.0),
                         feq(nx * ny * 9, 1.0),
                         rho(nx * ny, 1.0),
                         u(nx * ny, 0.0),
                         v(nx * ny, 0.0) {}

  auto index(int x, int y, int i) const { return (y * nx + x) * 9 + i; }
};

// Bounce-back boundary conditions
void apply_boundary_conditions(Grid& g) {
  // Implement bounce-back boundary conditions
  for (int y = 0; y < g.ny; ++y) {
    for (int x = 0; x < g.nx; ++x) {
      for (int i = 0; i < 9; ++i) {
        // Logic for bounce-back reflected indices goes here
        // This is just a placeholder; proper implementation needed
      }
    }
  }
}

// Stream and Collision step
void stream_collision_step(Grid& g) {
  double tau = 0.6;
  double omega = 1.0 / tau;
  double u_max = 0.1;

  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto is = std::views::iota(0, 9);
//  std::ranges::for_each(xs, [](int i)
//                        {
//                          std::cout << i << ' ';
//                        });
//  std::cout << std::endl;
//  abort();
  auto ids = std::views::cartesian_product(is,xs, ys);

  std::for_each(std::execution::par, ids.begin(), ids.end(), [&g, omega, u_max](auto idx) {
    auto [i, x, y] = idx;

    // Compute macroscopic variables
    double rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      rho += g.f[g.index(i, x, y)];
      ux += g.f[g.index(i, x, y)] * g.cx[i];
      uy += g.f[g.index(i, x, y)] * g.cy[i];
    }

    ux /= rho;
    uy /= rho;

    // Update equilibrium distributions
//    for (int i = 0; i < 9; ++i) {
      double cu = 3.0 * (g.cx[i] * ux + g.cy[i] * uy);
      double u_sq = 1.5 * (ux * ux + uy * uy);
      g.feq[g.index(i, x, y)] = g.w[i] * rho * (1 + cu + 0.5 * cu * cu - u_sq);
//    }

    // Collision step
//    for (int i = 0; i < 9; ++i) {
      g.f[g.index(i, x, y)] += omega * (g.feq[g.index(i, x, y)] - g.f[g.index(i,x, y)]);
//    }
  });

  // Apply boundary conditions
  apply_boundary_conditions(g);
}

int main() {
  // Grid parameters
  int nx = 100;
  int ny = 100;

  // Setup grid and initial conditions
  Grid g(nx, ny);

  // Time-stepping loop
  int num_steps = 1000;
  for (int t = 0; t < num_steps; ++t) {
    stream_collision_step(g);
  }

  std::cout << "Simulation complete.\n";
  return 0;
}
