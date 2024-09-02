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
using layout = exper::layout_left;


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

enum class Flags{hwbb,inlet,outlet,symmetry};
// Data structure for the D2Q9lattice
struct D2Q9lattice {
  int nx, ny;
  const T llb;
  const T w[9] = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
  const std::array<int, 9> cx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  const std::array<int, 9> cy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  const int d = 2;
  const int q = 9;
  const std::array<int, 9> opposite = {0, 3, 4, 1, 2, 7, 8, 5, 6};
  // Precomputed indices for 90° rotations
  // index =                                      {0, 1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int, 9> clockwise_90 =     {0, 4, 1, 2, 3, 8, 5, 6, 7};
  const std::array<int, 9> anticlockwise_90 = {0, 2, 3, 4, 1, 6, 7, 8, 5};
  T cslb = 1. / sqrt(3.);
  T cslb2 = cslb * cslb;
  T invCslb = 1. / cslb;
  T invCslb2 = invCslb * invCslb;

  std::vector<T> buffer;
  std::vector<Flags> flags_buffer;
  exper::mdspan<T, exper::dextents<int,3>,layout> f_data, f_data_2, dynamic_data;
  exper::mdspan<T, exper::dextents<int,2>,layout> u_data, v_data, rho_data;
  exper::mdspan<Flags, exper::dextents<int,3>,layout> flags_data;

  D2Q9lattice(int nx, int ny, T llb, int number_dynamic_scalars = 1) : nx(nx), ny(ny), llb(llb),
                         buffer(nx * ny * 9 * 2 + nx * ny * 3+ nx * ny * number_dynamic_scalars, 1.0) // Adjust buffer size accordingly
  {
    f_data = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data(), nx, ny, 9);
    f_data_2 = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data.size(), nx, ny, 9);
    u_data = exper::mdspan<T, exper::dextents<int,2>,layout>(buffer.data() + f_data.size()+ f_data_2.size(), nx, ny);
    v_data = exper::mdspan<T, exper::dextents<int,2>,layout>(buffer.data() + f_data.size()+ f_data_2.size() + u_data.size(), nx, ny);
    rho_data = exper::mdspan<T, exper::dextents<int,2>,layout>(buffer.data() + f_data.size()+ f_data_2.size() + u_data.size() + v_data.size(), nx, ny);
    dynamic_data = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data.size()+ f_data_2.size() + u_data.size() + v_data.size()+ rho_data.size(), nx, ny,number_dynamic_scalars);
    flags_data = exper::mdspan<Flags, exper::dextents<int,3>,layout>(flags_buffer.data(), nx, ny,9);

    initialize();
  }

  void swap() {
    std::swap(f_data, f_data_2);
  }

  inline int oppositeIndex(int i) const {
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

// Compute the moments of populations to populate density and velocity vectors in D2Q9lattice
void computeMoments(D2Q9lattice &g, bool even = true, bool before_cs = true) {
  assert(before_cs);
  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto coords = std::views::cartesian_product(xs, ys);

  // Parallel loop to compute moments
  std::for_each(std::execution::par_unseq, coords.begin(), coords.end(), [&g, even](auto coord) {
    auto [x, y] = coord;
    T rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
      int iPop = even ? i : g.opposite[i];
      rho += g.f_data(x, y, iPop);
      ux += g.f_data(x, y, iPop) * g.cx[i];
      uy += g.f_data(x, y, iPop) * g.cy[i];
    }
    g.rho_data(x, y) = rho;
    g.u_data(x, y) = ux / rho;
    g.v_data(x, y) = uy / rho;
  });
}


auto cylinder_flags_initialization(D2Q9lattice& g){
  // indexes
  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto is = std::views::iota(0, g.q);
  auto xis = std::views::cartesian_product(xs, is);
  auto yis = std::views::cartesian_product(ys, is);
  auto xyis = std::views::cartesian_product(xs,ys, is);



  T cx = g.nx/3.;
  T cy = g.ny/2.;
  T radius = g.ny/10.;

  auto getMinimumPositive = [](const std::array<T, 2>& p) -> std::optional<T> {
    std::optional<T> minimumPositive;

    if (p[0] > 0) {
      minimumPositive = p[0];
    }

    if (p[1] > 0) {
      if (minimumPositive) {
        minimumPositive = std::min(minimumPositive.value(), p[1]);
      } else {
        minimumPositive = p[1];
      }
    }

    return minimumPositive;
  };


  auto circle_intersect_segment = [cx, cy, radius](T x1, T y1, T x2, T y2)-> std::array<T, 2>{
    T dx = x2 - x1;
    T dy = y2 - y1;

    T fx = x1 - cx;
    T fy = y1 - cy;

    T a = dx * dx + dy * dy;
    T b = 2 * (fx * dx + fy * dy);
    T c = fx * fx + fy * fy - radius * radius;

    T discriminant = b * b - 4 * a * c;

    if (discriminant < 0)
      return std::array<T, 2>{NAN, NAN};

    discriminant = std::sqrt(discriminant);

    T t1 = (-b - discriminant) / (2 * a);
    T t2 = (-b + discriminant) / (2 * a);

    return std::array<T, 2>{t1, t2};
  };

    auto is_near = [cx,cy,radius](int x, int y){
      if ( std::abs(x - cx) < radius+1.5 and std::abs(y - cy) < radius+1.5)
        return true;
      else
        return false;
    };
  std::for_each(xyis.begin(),xyis.end(),[&g,cx,cy,radius,circle_intersect_segment,getMinimumPositive,is_near](auto xyi){
    auto [x,y,i] = xyi;


    if(x == 0) g.flags_data(x,y,i) = Flags::inlet;
    else if(x == g.nx - 1) g.flags_data(x,y,i) = Flags::outlet;
    else if (is_near(x,y)) {
      auto intersection = getMinimumPositive(circle_intersect_segment(x, y, x + g.cx[i], y + g.cy[i]));
      if(intersection.) // TODO
    } else if (not is_near(x,y)) std::numeric_limits<T>::signaling_NaN();
  });
}

void collide_stream_two_populations(D2Q9lattice &g, T ulb, T tau) {
  T omega = 1.0 / tau;

  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto ids = std::views::cartesian_product(xs, ys);

  T Rsquared = g.llb*g.llb;

  // Parallel loop ensuring thread safety
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&g, omega, ulb,Rsquared](auto idx) {
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
      // Variables to handle periodic boundary conditions
      int x_stream_periodic = (x_stream + g.nx) % g.nx;
      int y_stream_periodic = (y_stream + g.ny) % g.ny;

      // Handle periodic and bounce-back boundary conditions
      if (g.flags_data(x,y,i) == Flags::hwbb) {
        g.f_data_2(x, y, g.opposite[i]) = tmpf[i];
      } else if (g.flags_data(x,y,i) == Flags::inlet) {
        g.f_data_2(x, y, g.opposite[i]) = tmpf[i] - 2.0 * g.invCslb2 * (ulb * g.cx[i]) * g.w[i];
      } else if (g.flags_data(x,y,i) == Flags::outlet) {
        //do nothing
//      } else if (cylinder_configuration(x_stream, y_stream, g.nx, g.ny,Rsquared, symmetry)) {
//        if (g.cx[i] not_eq 0) g.f_data_2(x, y, g.cx[i] < 0 ? g.clockwise_90[i] : g.anticlockwise_90[i]) = tmpf[i];
//        else g.f_data_2(x, y, g.opposite[i]) = tmpf[i];
      } else { // periodic
        g.f_data_2(x_stream_periodic, y_stream_periodic, i) = tmpf[i];
      }
    }
  });
  g.swap();
}

//template<int truncation_level>
//void collide_stream_AA(D2Q9lattice &g, bool even, T ulb, T tau) {
//  T omega = 1.0 / tau;
//
//  auto xs = std::views::iota(0, g.nx);
//  auto ys = std::views::iota(0, g.ny);
//  auto ids = std::views::cartesian_product(xs, ys);
//
//  std::for_each(std::execution::seq, ids.begin(), ids.end(), [&g, omega, ulb, even](auto idx) {
//    auto [x, y] = idx;
//
//    std::array<T, 9> tmpf{0};
//    std::array<T, 9> feq{0};
//    T rho = 0.0, ux = 0.0, uy = 0.0;
//    for (int i = 0; i < 9; ++i) {
//      int x_stream = x - g.cx[i];
//      int y_stream = y - g.cy[i];
//      if (even) {// pull stream
//        if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
//          tmpf[i] = g.f_data(x_stream, y_stream, i);
//        } else {
//          tmpf[i] = g.f_data(x, y, g.opposite[i]) + ((y_stream == g.ny) ? -2. * g.invCslb2 * ulb * g.cx[i] * g.w[i] : 0.0);
//        }
//      } else { // odd
//        tmpf[i] = g.f_data(x, y, g.opposite[i]);// read-swap
//      }
//      rho += tmpf[i];
//      ux += tmpf[i] * g.cx[i];
//      uy += tmpf[i] * g.cy[i];
//    }
//
//    // Compute macroscopic density and velocities
//    ux /= rho;
//    uy /= rho;
//
//    for (int i = 0; i < 9; ++i) {
//      T cu = g.cx[i] * ux + g.cy[i] * uy;
//      T u_sq = ux * ux + uy * uy;
//      T cu_sq = cu * cu;
//      feq[i] = g.w[i] * rho * (1.0 + 3. * cu + 4.5 * cu_sq - 1.5 * u_sq);
//      tmpf[i] = tmpf[i] * (1. - omega) + feq[i] * omega;
//      int x_stream = x + g.cx[i];
//      int y_stream = y + g.cy[i];
//      if (even) {//push-swap
//        if (x_stream >= 0 && x_stream < g.nx && y_stream >= 0 && y_stream < g.ny) {
//          g.f_data(x_stream, y_stream, g.opposite[i]) = tmpf[i];
//        } else {
//          g.f_data(x, y, i) = tmpf[i] + ((y_stream == g.ny) ? -2. * g.invCslb2 * ulb * g.cx[i] * g.w[i] : 0.0);
//        }
//      } else { // odd
//        g.f_data(x, y, i) = tmpf[i];
//      }
//    }
//  });
//}
// Initialize function for double shear layer
// Initialize function for double shear layer
template <typename T>
void initializeDoubleShearLayer(D2Q9lattice &g, T U0, T alpha=80, T delta=0.05) {
  int nx = g.nx;
  int ny = g.ny;
  T L = nx;

  // Initialize density and velocity fields
  for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
      T rho_val = 1.0;

      // Define velocities based on the double shear layer profile
      T ux = U0 * std::tanh(alpha * (0.25 - std::abs((static_cast<T>(y) / static_cast<T>(ny)) - 0.5)));
      T uy = U0 * delta * std::sin(2.0 * M_PI * (static_cast<T>(x) / static_cast<T>(nx) + 0.25));

      g.rho_data(x, y) = rho_val;
      g.u_data(x, y) = ux;
      g.v_data(x, y) = uy;

      // Initialize populations based on the equilibrium distribution
      for (int i = 0; i < 9; ++i) {
        T cu = g.cx[i] * ux + g.cy[i] * uy;
        T u_sq = 1.5 * (ux * ux + uy * uy);
        g.f_data(x, y, i) = g.w[i] * rho_val * (1. + 3. * cu + 4.5 * cu * cu - u_sq);
        g.f_data_2(x, y, i) = g.f_data(x, y, i); // Initialize f_2 the same way as f
      }
    }
  }
}


int main() {
  int warm_up_iter = 1000;

  // numerical resolution
  int nx = 200;
  int ny = 200;
  T llb = ny/11.;

  // Setup D2Q9lattice and initial conditions
  auto g = std::make_unique<D2Q9lattice>(nx, ny,llb);

  // indexes
  auto xs = std::views::iota(0, g->nx);
  auto ys = std::views::iota(0, g->ny);
  auto is = std::views::iota(0, g->q);
  auto xis = std::views::cartesian_product(xs, is);
  auto yis = std::views::cartesian_product(ys, is);
  auto xyis = std::views::cartesian_product(xs,ys, is);

  // nondimentional numbers
  T Re =100;
  T Ma = 0.1;

  // reference dimensions
  T ulb = Ma * g->cslb;
  T nu = ulb * llb / Re;
  T taubar = nu * g->invCslb2;
  T tau = taubar + 0.5;

  T Tlb = g->nx / ulb;
  // Time-stepping loop parameters
  int num_steps = 200;Tlb;
  int outputIter = 1;num_steps / 100;

  printf("T_lb = %f\n", Tlb);
  printf("num_steps = %d\n", num_steps);
  printf("warm_up_iter = %d\n", warm_up_iter);
  printf("u_lb = %f\ntau = %f\n", ulb,tau);

  cylinder_flags_initialization(*g);


  // Initialize the D2Q9lattice with the double shear layer
//  initializeDoubleShearLayer(*g, ulb);

  // Start time measurement
  auto start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> output_time(0.0);

  // Loop over time steps
  for (int t = 0; t < num_steps; ++t) {

    if (t == warm_up_iter)
      start_time = std::chrono::high_resolution_clock::now();

    // Toggle between different streaming steps
    //    bool even = (t % 2 == 0);

    // Output results every outputIter iterations
    if (t % outputIter == 0) {
      // Compute macroscopic variables using the new function
      computeMoments(*g); // Dereference the unique_ptr to pass the reference.
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

    collide_stream_two_populations(*g, ulb, tau);

    std::for_each(std::execution::par_unseq, yis.begin(), yis.end(),
                  [f = g->f_data, nx, ny](auto yi) {
                    auto [y, i] = yi;
                    f(nx - 1, y, i) = f(nx - 2, y, i);
                  });
//    std::for_each(std::execution::par_unseq, xis.begin(), xis.end(),
//                  [f = g->f_data, nx, ny](auto xi) {
//                    auto [x, i] = xi;
//                    f(x, ny - 1, i) = f(x, ny - 2, i);
//                    f(x, 0, i) = f(x, 1, i);
//                  });
  }

  // End time measurement
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time - output_time;

  // Compute performance in MLUP/s
  long total_lattice_updates = static_cast<long>(nx) * static_cast<long>(ny) * static_cast<long>(num_steps-warm_up_iter);
  double mlups = total_lattice_updates / (elapsed.count() * 1.0e6);

  // Print performance
  std::cout << "Performance: " << mlups << " MLUP/s" << std::endl;

  return 0;
}

//      auto lastx = exper::submdspan(g->f_data, nx-1, exper::full_extent, exper::full_extent);
//      auto beforelastx = exper::submdspan(g->f_data, nx-2, exper::full_extent, exper::full_extent);
//      auto lasty = exper::submdspan(g->f_data,exper::full_extent, ny-1, exper::full_extent);
//      auto beforelasty = exper::submdspan(g->f_data, exper::full_extent, ny-2, exper::full_extent);