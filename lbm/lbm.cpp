#include <iostream>
#include <vector>
#include <cmath>
#include <ranges>
#include <execution>
#include <numeric>
#include <array>
#include <span>
#include <cartesian_product.hpp>  // Brings C++23 std::views::cartesian_product to C++20
#include <fstream>
#include <string>
#include <memory> // For std::unique_ptr and std::make_unique
#include <chrono>
#include <experimental/mdspan> // Include standard mdspan header
#include "statistics.h"

// Define a type alias for convenience
using T = float;
namespace exper = std::experimental;
using layout = exper::layout_right;


// Function to write VTK file for 2D data
template<typename T1, typename T2>
void writeVTK2D(const std::string &filename, const T1 &grid_coordinates, const T2 &md2, int NX, int NY) {
  std::ofstream out(filename); // Open the file

  if (!out.is_open()) {
    throw std::ios_base::failure("Failed to open file");
  }

  int N = NX * NY;

  out << "# vtk DataFile Version 3.0\n";
  out << "2D Test file\n";
  out << "ASCII\n";
  out << "DATASET STRUCTURED_GRID\n";
  out << "DIMENSIONS " << NX << ' ' << NY << ' ' << 1 << '\n'; // Z dimension is 1 for 2D data
  out << "POINTS " << N << " double\n";

  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      out << grid_coordinates[(ix * NY + iy) * 3 + 0] << " " << grid_coordinates[(ix * NY + iy) * 3 + 1] << " " << grid_coordinates[(ix * NY + iy) * 3 + 2] << '\n';
    }
  }

  // Writing scalar or vector field data
  out << "POINT_DATA " << N << '\n';
  out << "VECTORS velocity double\n";
  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int ic1 = 0; ic1 < 3; ++ic1) { // Assuming 3 components for velocity vector
        out << md2[(ix * NY + iy) * 3 + ic1] << ' ';
      }
      out << '\n';
    }
  }
  out.close(); // Close the file
}

// Data structure for the grid
struct Grid {
  int nx, ny;
  T w[9] = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
  std::array<int, 9> cx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  std::array<int, 9> cy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  T cslb = 1. / sqrt(3.);
  T cslb2 = cslb * cslb;
  T invCslb = 1. / cslb;
  T invCslb2 = invCslb * invCslb;

  std::vector<T> buffer;
  exper::mdspan<T, exper::dextents<int,3>,layout> f_data, f_data_2;
  exper::mdspan<T, exper::dextents<int,2>,layout> u_data, v_data, rho_data;

  Grid(int nx, int ny) : nx(nx), ny(ny),
                         buffer(nx * ny * 9 * 2 + nx * ny * 3, 1.0) // Adjust buffer size accordingly
  {
    int total_cells = nx * ny;
    int f_data_size = total_cells * 9;

    f_data = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data(), nx, ny, 9);
    f_data_2 = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data_size, nx, ny, 9);
    u_data = exper::mdspan<T, exper::dextents<int,2>,layout>(buffer.data() + f_data_size * 2, nx, ny);
    v_data = exper::mdspan<T, exper::dextents<int,2>,layout>(buffer.data() + f_data_size * 2 + total_cells, nx, ny);
    rho_data = exper::mdspan<T, exper::dextents<int,2>,layout>(buffer.data() + f_data_size * 2 + total_cells * 2, nx, ny);

    initialize();
  }

  void swap() {
    std::swap(f_data, f_data_2);
  }

  int oppositeIndex(int i) const {
    const std::array<int, 9> opposite = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return opposite[i];
  }

  void initialize() {
    for (int x = 0; x < nx; ++x) {
      for (int y = 0; y < ny; ++y) {
        T rho_val = 1.0;
        T ux = 0.0;
        T uy = 0.0;
        rho_data(x, y) = rho_val;
        u_data(x, y) = ux;
        v_data(x, y) = uy;
        for (int i = 0; i < 9; ++i) {
          T cu = 3.0 * (cx[i] * ux + cy[i] * uy);
          T u_sq = 1.5 * (ux * ux + uy * uy);
          f_data(x, y, i) = w[i] * rho_val * (1 + cu + 0.5 * cu * cu - u_sq);
          f_data_2(x, y, i) = f_data(x, y, i); // Initialize f_2 the same way as f
        }
      }
    }
  }
};

// Compute the moments of populations to populate density and velocity vectors in Grid
void computeMoments(Grid &g, bool even, bool before_cs) {
  assert(before_cs);
  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto coords = std::views::cartesian_product(xs, ys);

  // Parallel loop to compute moments
  std::for_each(std::execution::par_unseq, coords.begin(), coords.end(), [&g, before_cs, even](auto coord) {
    auto [x, y] = coord;
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      int iPop = even ? i : g.oppositeIndex(i);
      rho += g.f_data(x, y, iPop);
      ux += g.f_data(x, y, iPop) * g.cx[i];
      uy += g.f_data(x, y, iPop) * g.cy[i];
    }
    g.rho_data(x, y) = rho;
    g.u_data(x, y) = ux / rho;
    g.v_data(x, y) = uy / rho;
  });
}

template<int truncation_level>
void collide_stream_two_populations(Grid &g, T ulb, T tau) {
  T omega = 1.0 / tau;

  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto ids = std::views::cartesian_product(xs, ys);

  // Parallel loop ensuring thread safety
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&g, omega, ulb](auto idx) {
    auto [x, y] = idx;
    std::array<T, 9> tmpf{0};
    std::array<T, 9> feq{0};
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      tmpf[i] = g.f_data(x, y, i);
      rho += tmpf[i];
      ux += tmpf[i] * g.cx[i];
      uy += tmpf[i] * g.cy[i];
    }

    // Compute macroscopic density and velocities
    ux /= rho;
    uy /= rho;

    // Compute equilibrium distributions
    for (int i = 0; i < 9; ++i) {
      T cu = g.cx[i] * ux + g.cy[i] * uy;
      T u_sq = ux * ux + uy * uy;
      T cu_sq = cu * cu;
      feq[i] = g.w[i] * rho * (1.0 + 3. * cu + 4.5 * cu_sq - 1.5 * u_sq);
      tmpf[i] = tmpf[i] * (1. - omega) + feq[i] * omega;
      tmpf[i] = (1.0 - omega) * tmpf[i] + omega * feq[i];
      int x_stream = x + g.cx[i];
      int y_stream = y + g.cy[i];

      // Handle periodic and bounce-back boundary conditions
      if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
        g.f_data_2(x_stream, y_stream, i) = tmpf[i];
      } else {
        g.f_data_2(x, y, g.oppositeIndex(i)) = tmpf[i];

        // Add lid-driven momentum for the moving top wall
        if (y_stream == g.ny) {
          g.f_data_2(x, y, g.oppositeIndex(i)) -= 2.0 * g.invCslb2 * ulb * g.cx[i] * g.w[i];
        }
      }
    }
  });
  g.swap();
}

template<int truncation_level>
void collide_stream_AA(Grid &g, bool even, T ulb, T tau) {
  T omega = 1.0 / tau;

  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto ids = std::views::cartesian_product(xs, ys);

  std::for_each(std::execution::seq, ids.begin(), ids.end(), [&g, omega, ulb, even](auto idx) {
    auto [x, y] = idx;

    std::array<T, 9> tmpf{0};
    std::array<T, 9> feq{0};
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      int x_stream = x - g.cx[i];
      int y_stream = y - g.cy[i];
      if (even) {// pull stream
        if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
          tmpf[i] = g.f_data(x_stream, y_stream, i);
        } else {
          tmpf[i] = g.f_data(x, y, g.oppositeIndex(i)) + ((y_stream == g.ny) ? -2. * g.invCslb2 * ulb * g.cx[i] * g.w[i] : 0.0);
        }
      } else { // odd
        tmpf[i] = g.f_data(x, y, g.oppositeIndex(i));// read-swap
      }
      rho += tmpf[i];
      ux += tmpf[i] * g.cx[i];
      uy += tmpf[i] * g.cy[i];
    }

    // Compute macroscopic density and velocities
    ux /= rho;
    uy /= rho;

    for (int i = 0; i < 9; ++i) {
      T cu = g.cx[i] * ux + g.cy[i] * uy;
      T u_sq = ux * ux + uy * uy;
      T cu_sq = cu * cu;
      feq[i] = g.w[i] * rho * (1.0 + 3. * cu + 4.5 * cu_sq - 1.5 * u_sq);
      tmpf[i] = tmpf[i] * (1. - omega) + feq[i] * omega;
      int x_stream = x + g.cx[i];
      int y_stream = y + g.cy[i];
      if (even) {//push-swap
        if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
          g.f_data(x_stream, y_stream, g.oppositeIndex(i)) = tmpf[i];
        } else {
          g.f_data(x, y, i) = tmpf[i] + ((y_stream == g.ny) ? -2. * g.invCslb2 * ulb * g.cx[i] * g.w[i] : 0.0);
        }
      } else { // odd
        g.f_data(x, y, i) = tmpf[i];
      }
    }
  });
}

int main() {
  // Grid parameters
  int nx = 100;
  int ny = 100;

  // Setup grid and initial conditions
  auto g = std::make_unique<Grid>(nx, ny);

  constexpr int truncation_level = 2; // Level of polynomial truncation for equilibrium

  T Re = 1000;
  T Ma = 0.1;

  T ulb = Ma * g->cslb;
  T nu = ulb * g->nx / Re;
  T taubar = nu * g->invCslb2;
  T tau = taubar + 0.5;

  T Tlb = g->nx / ulb;
  // Time-stepping loop parameters
  int num_steps = 10 * Tlb;
  int outputIter = num_steps / 20;

  printf("T_lb = %f\n", Tlb);

  // Start time measurement
  auto start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> output_time;

  // Loop over time steps
  for (int t = 0; t < num_steps; ++t) {

    // Toggle between different streaming steps
    bool even = (t % 2 == 0);

    // Output results every outputIter iterations
    if (t % outputIter == 0) {
      // Compute macroscopic variables using the new function
      computeMoments(*g, true, true); // Dereference the unique_ptr to pass the reference.
      std::vector<T> grid_points((nx * ny) * 3);
      std::vector<T> macroscopic_data((nx * ny) * 3);

      for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
          int idx3d = (x * ny + y) * 3;

          // Coordinates for each grid point (assuming a uniform grid, z=0 for 2D)
          grid_points[idx3d + 0] = x;
          grid_points[idx3d + 1] = y;
          grid_points[idx3d + 2] = 0;

          // Velocity
          macroscopic_data[idx3d + 0] = g->u_data(x, y);
          macroscopic_data[idx3d + 1] = g->v_data(x, y);
          macroscopic_data[idx3d + 2] = 0; // z component of velocity is zero for 2D
        }
      }

      std::string filename = "output_" + std::to_string(t) + ".vtk";
      auto before_out = std::chrono::high_resolution_clock::now();
      writeVTK2D(filename, grid_points, macroscopic_data, nx, ny);
      auto after_out = std::chrono::high_resolution_clock::now();

      output_time += after_out - before_out;
    }
    collide_stream_two_populations<truncation_level>(*g, ulb, tau);
  }

  // End time measurement
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time - output_time;

  // Compute performance in MLUP/s
  long total_lattice_updates = static_cast<long>(nx) * static_cast<long>(ny) * static_cast<long>(num_steps);
  double mlups = total_lattice_updates / (elapsed.count() * 1.0e6);

  // Print performance
  std::cout << "Performance: " << mlups << " MLUP/s" << std::endl;

  return 0;
}
