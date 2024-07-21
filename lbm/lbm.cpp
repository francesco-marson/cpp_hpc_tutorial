#include <iostream>
#include <vector>
#include <cmath>
#include <mdspan>
#include <ranges>
#include <execution>
#include <numeric>
#include <array>
#include <cartesian_product.hpp> // Brings C++23 std::views::cartesian_product to C++20

#include <fstream>
#include <string>

using T = double;

template<typename T1, typename T2>
void writeVTK2D(const std::string &filename, const T1 &all_data, const T2 &md2, int NX, int NY) {
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
      out << all_data(ix, iy, 0) << " " << all_data(ix, iy, 1) << " " << 0.0 << '\n';
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
  std::array<int, 9> cx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  std::array<int, 9> cy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  std::vector<T> f, feq, rho, u, v;

  Grid(int nx, int ny)
      : nx(nx), ny(ny), f(nx * ny * 9, 1.0), feq(nx * ny * 9, 1.0), rho(nx * ny, 1.0),
        u(nx * ny, 0.0), v(nx * ny, 0.0) {
    initialize();
  }

  auto index(int x, int y, int i) const { return (y * nx + x) * 9 + i; }

  int oppositeIndex(int i) const {
    // Opposite index based on D2Q9 model
    static const std::array<int, 9> opposite = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return opposite[i];
  }

  void initialize() {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        T rho = 1.0;
        T ux = 0.0;
        T uy = 0.0;
        this->rho[y * nx + x] = rho;
        u[y * nx + x] = ux;
        v[y * nx + x] = uy;
        for (int i = 0; i < 9; ++i) {
          T cu = 3.0 * (cx[i] * ux + cy[i] * uy);
          T u_sq = 1.5 * (ux * ux + uy * uy);
          feq[index(x, y, i)] = w[i] * rho * (1 + cu + 0.5 * cu * cu - u_sq);
          f[index(x, y, i)] = feq[index(x, y, i)];
        }
      }
    }
  }
};

// Compute macroscopic variables
void compute_macroscopic(Grid& g) {
  for (int y = 0; y < g.ny; ++y) {
    for (int x = 0; x < g.nx; ++x) {
      T rho = 0.0, ux = 0.0, uy = 0.0;
      for (int i = 0; i < 9; ++i) {
        rho += g.f[g.index(x, y, i)];
        ux += g.f[g.index(x, y, i)] * g.cx[i];
        uy += g.f[g.index(x, y, i)] * g.cy[i];
      }
      g.rho[y * g.nx + x] = rho;
      g.u[y * g.nx + x] = ux / rho;
      g.v[y * g.nx + x] = uy / rho;
    }
  }
}

// Compute the moments of populations to populate density and velocity vectors in Grid
void computeMoments(Grid& g) {
  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto coords = std::views::cartesian_product(xs, ys);

  // Parallel loop to compute moments
  std::for_each(std::execution::par_unseq, coords.begin(), coords.end(), [&g](auto coord) {
    auto [x, y] = coord;
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      rho += g.f[g.index(x, y, i)];
      ux += g.f[g.index(x, y, i)] * g.cx[i];
      uy += g.f[g.index(x, y, i)] * g.cy[i];
    }
    g.rho[y * g.nx + x] = rho;
    g.u[y * g.nx + x] = ux / rho;
    g.v[y * g.nx + x] = uy / rho;
  });
}

void collide_stream_step(Grid& g, bool even) {
  T tau = 1.0;
  T omega = 1.0 / tau;
  T ulb = 0.1;

  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto is = std::views::iota(0, 9);
  auto ids = std::views::cartesian_product(is, xs, ys);

  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&g, omega, ulb, even](auto idx) {
    auto [ii, x, y] = idx;
    size_t i = ii;
    size_t i_out = g.oppositeIndex(i);

    // even: streamPull -> collide -> swapPush
    // odd:  staticPull -> collide -> swapPush

    // Stream step (Pull), then collide
    int x_stream = x - g.cx[i];
    int y_stream = y - g.cy[i];

    T tmpf[9];
    T feq[9];

    // pull
    if (even) {
      if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
        tmpf[i] = g.f[g.index(x_stream, y_stream, i)];
      } /*else if (x == g.ny - 1 and g.cy[ii] == 1) {
        tmpf[i] = g.f[g.index(x, y, g.oppositeIndex(i))] *//*- 2. * 3. * ulb*g.cx[ii]*g.w[ii]*//*;
      }*/ else {
        tmpf[i] = g.f[g.index(x, y, g.oppositeIndex(i))];
      }
    } else {
      tmpf[i] = g.f[g.index(x, y, i_out)];
    }

    // Compute equilibrium distribution
    T rho = g.rho[y * g.nx + x];
    T ux = g.u[y * g.nx + x];
    T uy = g.v[y * g.nx + x];
    T cu = 3.0 * (g.cx[i] * ux + g.cy[i] * uy);
    T u_sq = 1.5 * (ux * ux + uy * uy);
    feq[i] = g.w[i] * rho * (1 + cu + 0.5 * cu * cu - u_sq);

    // Collision step
    tmpf[i] = tmpf[i] * (1 - omega) + feq[i] * omega;

    // Handling boundaries in-place
    // Bounce-back boundary conditions

    //push
    if(even){
      if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny)
        g.f[g.index(x_stream, y_stream, i)] = tmpf[g.oppositeIndex(i)];
      else
        g.f[g.index(x, y, i)] = tmpf[i];
    } else { // odd
      g.f[g.index(x, y, i)] = tmpf[i];
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
    int num_steps = 1000;
    for (int t = 0; t < num_steps; ++t) {
      // Compute macroscopic variables using the new function
      computeMoments(g);

      // Toggle between different streaming steps
      bool even = (t % 2 == 0);
      collide_stream_step(g, even);
    }

  // Define the mdspan for all_data and md2
  std::vector<double> pts_data(nx * ny * 3);  // Note 3 instead of 2 for coordinates
  std::vector<double> f0_data(nx * ny * 3);   // Note 3 instead of 2 for velocity components

  for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
      pts_data[(y * nx + x) * 3 + 0] = x;
      pts_data[(y * nx + x) * 3 + 1] = y;
      pts_data[(y * nx + x) * 3 + 2] = 0.0;

      f0_data[(y * nx + x) * 3 + 0] = g.u[y * nx + x];
      f0_data[(y * nx + x) * 3 + 1] = g.v[y * nx + x];
      f0_data[(y * nx + x) * 3 + 2] = 0.0;
    }
  }

  auto pts = std::experimental::mdspan(pts_data.data(), nx, ny, 3);
  auto f0 = std::experimental::mdspan(f0_data.data(), nx, ny, 3);

  // Write to VTK file
  writeVTK2D("output.vtk", pts, f0, nx, ny);

  std::cout << "Simulation complete.\n";
  return 0;
}
