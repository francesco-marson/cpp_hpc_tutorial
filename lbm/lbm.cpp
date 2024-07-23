#include <iostream>
#include <vector>
#include <cmath>
#include <mdspan>
#include <ranges>
#include <execution>
#include <numeric>
#include <array>
#include <cartesian_product.hpp>  // Brings C++23 std::views::cartesian_product to C++20

#include <fstream>
#include <string>

using T = double;

template<typename T1, typename T2>
void writeVTK2D(const std::string &filename, const T1 &grid_coordinates, const T2 &md2, int NX, int NY) {
  std::ofstream out(filename);  // Open the file

  if (!out.is_open()) {
    throw std::ios_base::failure("Failed to open file");
  }

  int N = NX * NY;

  out << "# vtk DataFile Version 3.0\n";
  out << "2D Test file\n";
  out << "ASCII\n";
  out << "DATASET STRUCTURED_GRID\n";
  out << "DIMENSIONS " << NX << ' ' << NY << ' ' << 1 << '\n';  // Z dimension is 1 for 2D data
  out << "POINTS " << N << " double\n";

  // Writing point coordinates
  for (int ix = 0; ix < NX; ix++) {
    for (int iy = 0; iy < NY; iy++) {
      out << grid_coordinates(ix, iy, 0) << " " << grid_coordinates(ix, iy, 1) << " " << grid_coordinates(ix, iy, 2) << '\n';
    }
  }

  // Writing scalar or vector field data
  out << "POINT_DATA " << N << '\n';
  out << "VECTORS velocity double\n";
  for (int ix = 0; ix < NX; ix++) {
    for (int iy = 0; iy < NY; iy++) {
      for (int ic1 = 0; ic1 < md2.extent(2); ic1++) {  // Assuming md2.extent(2) is correctly returning the third dimension size
        out << md2(ix, iy, ic1) << ' ';  // Considering md2 is 3D, change accordingly if it's 4D or different
      }
      // Ensure only three components per line for vectors
      if (md2.extent(2) < 3) {
        // Fill remaining components with zero if less than 3 components are present
        for (int remaining = md2.extent(2); remaining < 3; ++remaining) {
          out << "0.0 ";
        }
      }
      out << '\n';
    }
  }
  out.close();  // Close the file
}

// Data structure for the grid
struct Grid {
  int nx, ny;
  T w[9] = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
  std::array<int, 9> cx = {0, 1, 0, -1, 0, 1, -1, -1,  1};
  std::array<int, 9> cy = {0, 0, 1,  0, -1, 1,  1, -1, -1};

  std::vector<T> f_data, feq_data, rho_data, u_data, v_data;
  std::experimental::mdspan<T, std::experimental::dextents<int, 3>> f, feq;
  std::experimental::mdspan<T, std::experimental::dextents<int, 2>> rho, u, v;

  Grid(int nx, int ny)
      : nx(nx), ny(ny), f_data(nx * ny * 9, 1.0), feq_data(nx * ny * 9, 1.0), rho_data(nx * ny, 1.0),
        u_data(nx * ny, 0.0), v_data(nx * ny, 0.0),
        f(f_data.data(), nx, ny, 9), feq(feq_data.data(), nx, ny, 9),
        rho(rho_data.data(), nx, ny), u(u_data.data(), nx, ny), v(v_data.data(), nx, ny) {
    initialize();
  }

  int oppositeIndex(int i) const {
    // Opposite index based on D2Q9 model
    static const std::array<int, 9> opposite = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return opposite[i];
  }

  void initialize() {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        T rho_val = 1.0;
        T ux = 0.0;
        T uy = 0.0;
        rho(y, x) = rho_val;
        u(y, x) = ux;
        v(y, x) = uy;
        for (int i = 0; i < 9; ++i) {
          T cu = 3.0 * (cx[i] * ux + cy[i] * uy);
          T u_sq = 1.5 * (ux * ux + uy * uy);
          feq(y, x, i) = w[i] * rho_val * (1 + cu + 0.5 * cu * cu - u_sq);
          f(y, x, i) = feq(y, x, i);
        }
      }
    }
  }
};

// Compute macroscopic variables
void compute_macroscopic(Grid &g) {
  for (int y = 0; y < g.ny; ++y) {
    for (int x = 0; x < g.nx; ++x) {
      T rho = 0.0, ux = 0.0, uy = 0.0;
      for (int i = 0; i < 9; ++i) {
        rho += g.f(y, x, i);
        ux += g.f(y, x, i) * g.cx[i];
        uy += g.f(y, x, i) * g.cy[i];
      }
      g.rho(y, x) = rho;
      g.u(y, x) = ux / rho;
      g.v(y, x) = uy / rho;
    }
  }
}

// Compute the moments of populations to populate density and velocity vectors in Grid
void computeMoments(Grid &g) {
  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto coords = std::views::cartesian_product(xs, ys);

  // Parallel loop to compute moments
  std::for_each(std::execution::par_unseq, coords.begin(), coords.end(), [&g](auto coord) {
    auto [x, y] = coord;
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      rho += g.f(y, x, i);
      ux += g.f(y, x, i) * g.cx[i];
      uy += g.f(y, x, i) * g.cy[i];
    }
    g.rho(y, x) = rho;
    g.u(y, x) = ux / rho;
    g.v(y, x) = uy / rho;
  });
}

void collide_stream_step(Grid &g, bool even) {
  T tau = 2.0;
  T omega = 1.0 / tau;
  T ulb = 0.01;

  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto ids = std::views::cartesian_product(xs, ys);

  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&g, omega, ulb, even](auto idx) {
    auto [x, y] = idx;

    // even: streamPull -> collide -> swapPush
    // odd: staticPull -> collide -> swapPush

    // Stream step (Pull), then collide
    std::array<T, 9> tmpf;
    std::array<T, 9> feq;
    for (int i = 0; i < 9; ++i) {
      int x_stream = x - g.cx[i];
      int y_stream = y - g.cy[i];
      // pull
      if (even) {
        if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
          tmpf[i] = g.f(y_stream, x_stream, i);
        } else if (y == g.ny - 1 && g.cy[i] == 1 && y_stream == g.ny) {
          tmpf[i] = g.f(y, x, g.oppositeIndex(i)) - 2. * 3. * ulb * g.cx[i] * g.w[i];
        } else {
          tmpf[i] = g.f(y, x, g.oppositeIndex(i));
        }
      } else {
        tmpf[i] = g.f(y, x, g.oppositeIndex(i));
      }
    }

    // Compute equilibrium distribution
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      rho += tmpf[i];
      ux += tmpf[i] * g.cx[i];
      uy += tmpf[i] * g.cy[i];
    }
    g.rho(y, x) = rho;
    g.u(y, x) = ux / rho;
    g.v(y, x) = uy / rho;
    for (int i = 0; i < 9; ++i) {
      T cu = (g.cx[i] * ux + g.cy[i] * uy);
      T invCs2 = 3.;
      feq[i] = g.w[i] * rho * (1. + cu * invCs2);

      // Collision step
      tmpf[i] = tmpf[i] * (1. - omega) + feq[i] * omega;
    }

    // Handling boundaries in-place
    // Bounce-back boundary conditions

    // push
    for (int i = 0; i < 9; ++i) {
      int x_stream = x + g.cx[i];
      int y_stream = y + g.cy[i];
      if (even) {
        if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
          g.f(y_stream, x_stream, g.oppositeIndex(i)) = tmpf[i];
        }
        if (y == g.ny - 1 && g.cy[i] == 1 && y_stream == g.ny) {
          g.f(y, x, i) = tmpf[i] - 2. * 3. * ulb * g.cx[i] * g.w[i];
        } else {
          g.f(y, x, i) = tmpf[i];
        }
      } else {  // odd
        g.f(y, x, i) = tmpf[i];
      }
    }
  });
}

int main() {
  // Grid parameters
  int nx = 50;
  int ny = 50;

  // Setup grid and initial conditions
  Grid g(nx, ny);

  // Time-stepping loop
  int num_steps = 100;
  for (int t = 0; t < num_steps; ++t) {
    // Compute macroscopic variables using the new function
    // computeMoments(g);

    // Toggle between different streaming steps
    bool even = (t % 2 == 0);
    collide_stream_step(g, even);
  }

  // Define the mdspan for all_data and md2
  std::vector<double> ptsV(nx * ny * 3);  // Note 3 instead of 2 for coordinates
  std::vector<double> f0V(nx * ny * 3);   // Note 3 instead of 2 for velocity components

  auto pts = std::experimental::mdspan(ptsV.data(), nx, ny, 3);
  auto f0 = std::experimental::mdspan(f0V.data(), nx, ny, 3);

  for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
      pts(x, y, 0) = x;
      pts(x, y, 1) = y;
      pts(x, y, 2) = 0.0;

      f0(x, y, 0) = g.u(y, x);
      f0(x, y, 1) = g.v(y, x);
      f0(x, y, 2) = 0.0;
    }
  }

  // Write to VTK file
  writeVTK2D("output.vtk", pts, f0, nx, ny);

  std::cout << "Simulation complete.\n";
  return 0;
}
