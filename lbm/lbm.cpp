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


template<typename T, typename LayoutPolicy>
void writeVTK2D(
    const std::string &filename,
    tl::cartesian_product_view<std::ranges::iota_view<int, int>, std::ranges::iota_view<int, int>>grid,
    std::vector<std::pair<std::string, exper::mdspan<T, exper::dextents<int, 3>, LayoutPolicy>>> fields,
    int NX, int NY
) {
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

  // Writing the grid coordinates using cartesian product
  for (const auto& [iy, ix] : grid) {
    out << ix << " " << iy << " " << 0 << '\n';
  }

  // Writing field data
  out << "POINT_DATA " << N << '\n';

  for (const auto& [field_name, field_data] : fields) {
    int num_components = field_data.extent(2);

    if (num_components <= 1) {
      out << "SCALARS " << field_name << " double\n";
//      std::cout << "writing... SCALARS " << field_name << " double" << "; num_components = " << num_components << std::endl;
      out << "LOOKUP_TABLE default\n";
    } else if (num_components < 4){
      out << "VECTORS " << field_name << " double\n";
//      std::cout << "writing... VECTORS " << field_name << " double" << "; num_components = " << num_components << std::endl;
    }else {
      out << "TENSORS " << field_name << " double\n";
//      std::cout << "writing... TENSORS " << field_name << " double" << "; num_components = " << num_components << std::endl;
    }

    for (const auto& [iy, ix] : grid) {
      for (int ic = 0; ic < num_components; ++ic) {
        out << field_data(ix, iy, ic) << ' ';
      }
      if (num_components < 3) out << 0 << ' ';
      out << '\n';
    }
  }
  out.close(); // Close the file
}


// Data structure for the D2Q9lattice
struct D2Q9lattice {
  enum Flags{bulk,hwbb,inlet,outlet,symmetry};
  int nx, ny;
  const T llb;
  const T w[9] = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
  const std::array<int, 9> cx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  const std::array<int, 9> cy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  const std::array<T, 9> cnorm = {0, 1, 1, 1, 1, sqrt((T)2), sqrt((T)2), sqrt((T)2), sqrt((T)2)};
  const int d = 2;
  const int q = 9;
  const int number_dynamic_scalars = q;
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
  exper::mdspan<T, exper::dextents<int,3>,layout> f_data, f_data_2, dynamic_data,velocity_data,rho_data_;
    // Take a subspan that only considers the first two indices
// Define the full type for the submdspan
    exper::mdspan<T, exper::dextents<int, 2>, layout> rho_data;
  exper::mdspan<Flags, exper::dextents<int,3>,layout> flags_data;

  D2Q9lattice(int nx, int ny, T llb) : nx(nx), ny(ny), llb(llb),
                         buffer(nx * ny * q * 2 + nx * ny * 3+ nx * ny * number_dynamic_scalars, 1.0),
                                                                       flags_buffer(nx * ny * q,bulk)// Adjust buffer size accordingly
  {
    f_data = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data(), nx, ny, q);
    f_data_2 = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data.size(), nx, ny, q);
    velocity_data = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data.size()+ f_data_2.size(), nx, ny, d);
    rho_data_ = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data.size()+ f_data_2.size() + velocity_data.size(), nx, ny,0);
    dynamic_data = exper::mdspan<T, exper::dextents<int,3>,layout>(buffer.data() + f_data.size()+ f_data_2.size() + velocity_data.size() + rho_data_.size(), nx, ny,number_dynamic_scalars);

    flags_data = exper::mdspan<Flags, exper::dextents<int,3>,layout>(flags_buffer.data(), nx, ny,q);

      // Take the subspan that only considers the first two indexes
    rho_data = exper::submdspan(rho_data_, exper::full_extent, exper::full_extent, 0/*std::make_pair(0, 1)*/);
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
        velocity_data(x, y, 0) = ux;
        velocity_data(x, y, 1) = uy;
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
    g.velocity_data(x, y, 0) = ux / rho;
    g.velocity_data(x, y, 1) = uy / rho;
  });
}

std::vector<std::array<T,2> > generateNACAAirfoil(std::array<T,2> origin, uint length, unsigned int tesselation,
                                                  const std::string& naca, double aoa_deg) {
    std::vector<std::array<T,2> > points;
    double t = std::stoi(naca.substr(2, 2)) / 100.0;
    double m = std::stoi(naca.substr(0, 1)) / 100.0;
    double p = std::stoi(naca.substr(1, 1)) / 10.0;

    double aoa_rad = M_PI * aoa_deg / 180.0; // Convert aoa to radians

    for (unsigned int i = 0; i <= tesselation; ++i) {
        double x = ((double)i) / tesselation;
        double yt = 5 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * pow(x, 2) + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));
        double yc;
        if (x <= p) {
            yc = m / pow(p, 2) * (2 * p * x - pow(x, 2));
        } else {
            yc = m / pow(1-p, 2) * ((1 - 2 * p) + 2 * p * x - pow(x, 2));
        }
        double theta = atan(m / pow(p, 2) * (2 * p - 2 * x));

        // Apply rotation using rotation matrix
        double x1 = (x - yt * sin(theta))*length;
        double x2 = (yc + yt * cos(theta))*length;
        double xr = x1 * cos(aoa_rad) - x2 * sin(aoa_rad);
        double yr = x1 * sin(aoa_rad) + x2 * cos(aoa_rad);

        points.push_back(std::array<T,2>{xr + origin[0], yr + origin[1]});
    }

    for (int i = tesselation-1; i >= 0; --i) {
        double x = ((double)i) / tesselation;
        double yt = 5 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * pow(x, 2) + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));
        double yc;
        if (x <= p) {
            yc = m / pow(p, 2) * (2 * p * x - pow(x, 2));
        } else {
            yc = m / pow(1-p, 2) * ((1 - 2 * p) + 2 * p * x - pow(x, 2));
        }
        double theta = atan(m / pow(p, 2) * (2 * p - 2 * x));

        // Apply rotation using rotation matrix
        double x1 = (x + yt * sin(theta))*length;
        double x2 = (yc - yt * cos(theta))*length;
        double xr = x1 * cos(aoa_rad) - x2 * sin(aoa_rad);
        double yr = x1 * sin(aoa_rad) + x2 * cos(aoa_rad);

        points.push_back(std::array<T,2>{xr + origin[0], yr + origin[1]});
    }

    return points;
}

auto line_segments_flags_initialization(D2Q9lattice& g, const std::vector<std::array<T, 2>>& segments){
    // indexes
    auto xs = std::views::iota(0, g.nx);
    auto ys = std::views::iota(0, g.ny);
    auto is = std::views::iota(0, g.q);
    auto xis = std::views::cartesian_product(xs, is);
    auto yis = std::views::cartesian_product(ys, is);
    auto xyis = std::views::cartesian_product(xs,ys, is);

    auto getMinimumPositive = [](const auto& iterable) -> std::optional<T> {
        std::optional<T> minValue;
        for (const auto& value : iterable) {
            if (value > 0 && (!minValue || value < *minValue)) {
                minValue = value;
            }
        }
        return minValue;
    };

    // Lambda for line segment intersection
    auto segment_intersect_segment = [segments,getMinimumPositive](T x1, T y1, T x2, T y2) -> std::optional<T> {
        std::vector<T> intersections;
        for (auto it = segments.begin(); it != segments.end(); ++it){
            const auto& segment = *it;
            const auto& segment_next = it != std::prev(segments.end()) ? *(it +1) : segments.front();
            T s1_x = x2 - x1;
            T s1_y = y2 - y1;
            T s2_x = segment_next[0] - segment[0];
            T s2_y = segment_next[1] - segment[1];

            T s = (-s1_y * (x1 - segment[0]) + s1_x * (y1 - segment[1])) / (-s2_x * s1_y + s1_x * s2_y);
            T t = (s2_x * (y1 - segment[1]) - s2_y * (x1 - segment[0])) / (-s2_x * s1_y + s1_x * s2_y);

            if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
                intersections.push_back(std::sqrt(s1_x * s1_x + s1_y * s1_y) * t);
        }

        return getMinimumPositive(intersections);
    };

    std::for_each(xyis.begin(),xyis.end(),[&g,segment_intersect_segment](auto xyi){
        auto [x,y,i] = xyi;

        if (x == 0 and g.cx[i] == -1)
            g.flags_data(x, y, i) = g.inlet;
        else if (x == (g.nx - 1) and g.cx[i] > 0)
            g.flags_data(x, y, i) = g.outlet;
        else if (y == 0 and g.cy[i] < 0)
            g.flags_data(x, y, i) = g.outlet;
        else if (y == g.ny-1 and g.cy[i] > 0)
            g.flags_data(x, y, i) = g.outlet;
        else {
            auto intersection =
                    segment_intersect_segment(x, y, x + g.cx[i], y + g.cy[i]);
            if (intersection.has_value() and intersection.value() < g.cnorm[i]) {
                g.flags_data(x, y, i) = g.hwbb;
                g.dynamic_data(x, y, i) = intersection.value();
            } else {
                g.flags_data(x, y, i) = g.bulk;
            }
        }
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



  T cx = g.nx/3.-0.5+std::numeric_limits<T>::epsilon();
  T cy = g.ny/2.-0.5+std::numeric_limits<T>::epsilon();
  T radius = g.ny/10.+std::numeric_limits<T>::epsilon();

  auto getMinimumPositive = [](const auto& iterable) -> std::optional<T> {
    std::optional<T> minValue;
    for (const auto& value : iterable) {
      if (value > 0 && (!minValue || value < *minValue)) {
        minValue = value;
      }
    }
    return minValue;
  };


// Lambda that determines the intersection points of a circle and a line segment.
    auto circle_intersect_segment = [cx, cy, radius](T x1, T y1, T x2, T y2)-> std::array<T, 2> {
        // Calculate the difference in x and y coordinates of the end points of the segment.
        T dx = x2 - x1;
        T dy = y2 - y1;

        // Calculate the difference in x and y coordinates between
        // the initial point of the segment and the center of the circle.
        T fx = x1 - cx;
        T fy = y1 - cy;

        // Coefficients for the quadratic equation ax^2 + bx + c = 0
        // derived from the equation of the circle and line segment.
        T a = dx * dx + dy * dy;
        T b = 2 * (fx * dx + fy * dy);
        T c = fx * fx + fy * fy - radius * radius;

        // Calculate the discriminant of the quadratic equation to determine if there are intersection points.
        T discriminant = b * b - 4 * a * c;

        // If the discriminant is less than zero, there are no real intersection points.
        if (discriminant < 0)
            return std::array<T, 2>{NAN, NAN};

        // Calculate the square root of the discriminant.
        discriminant = std::sqrt(discriminant);

        // Compute the two solutions of the quadratic equation,
        // which correspond to the parameter values of the intersection points.
        T t1 = (-b - discriminant) / (2 * a);
        T t2 = (-b + discriminant) / (2 * a);


        // Return the parameter values of the intersection points.
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

    if (x == 0 and g.cx[i] == -1)
      g.flags_data(x, y, i) = g.inlet;
    else if (x == (g.nx - 1) and g.cx[i] > 0)
      g.flags_data(x, y, i) = g.outlet;
    else if (/*x > 0 and */y == 0 and g.cy[i] < 0)
      g.flags_data(x, y, i) = g.outlet;
    else if (/*x > 0 and */y == g.ny-1 and g.cy[i] > 0)
      g.flags_data(x, y, i) = g.outlet;
    else if (is_near(x, y)) {
      auto intersection =
          getMinimumPositive(circle_intersect_segment(x, y, x + g.cx[i], y + g.cy[i]));
      if (intersection.has_value() and intersection.value() < g.cnorm[i]) {
        g.flags_data(x, y, i) = g.hwbb;
        g.dynamic_data(x, y, i) = intersection.value();
      }
    } else /*if (not is_near(x,y))*/ g.flags_data(x, y, i) = g.bulk;
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
        auto& iopp = g.opposite[i];
      T cu = g.cx[i] * ux + g.cy[i] * uy;
      T u_sq = ux * ux + uy * uy;
      T cu_sq = cu * cu;
      feq[i] = g.w[i] * rho * (1.0 + 3. * cu + 4.5 * cu_sq - 1.5 * u_sq);
      T feq_iopp = g.w[i] * rho * (1.0 - 3. * cu + 4.5 * cu_sq - 1.5 * u_sq);
      tmpf[i] = (1.0 - omega) * tmpf[i] + omega * feq[i];
      T tmpf_iopp = (1.0 - omega) * tmpf[iopp] + omega * feq_iopp;
      int x_stream = x + g.cx[i];
      int y_stream = y + g.cy[i];
      // Variables to handle periodic boundary conditions
      int x_stream_periodic = (x_stream + g.nx) % g.nx;
      int y_stream_periodic = (y_stream + g.ny) % g.ny;

//      auto a = g.hwbb;
      // Handle periodic and bounce-back boundary conditions
      if (g.flags_data(x,y,i) == g.hwbb) {
          T q = g.dynamic_data(x,y,i)/g.cnorm[i];
        g.f_data_2(x, y, iopp) = tmpf[i];
//          g.f_data_2(x, y, iopp) = q * 0.5*(tmpf[i]+tmpf_iopp) + (1.-q)*0.5*(g.f_data(x, y, i)+g.f_data(x, y, iopp));
      } else if (g.flags_data(x,y,i) == g.inlet) {
        g.f_data_2(x, y, g.opposite[i]) = tmpf[i] - 2.0 * g.invCslb2 * (ulb * g.cx[i]) * g.w[i];
      } else if (g.flags_data(x,y,i) == g.outlet) {
        //do nothing
      }  else { // periodic
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
      g.velocity_data(x, y, 0) = ux;
      g.velocity_data(x, y, 1) = uy;

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
  int nx = 1200;
  int ny = 600;
  T llb = ny/11.;

  // Setup D2Q9lattice and initial conditions
  auto g = std::make_unique<D2Q9lattice>(nx, ny,llb);

  // indexes
  auto xs = std::views::iota(0, g->nx);
  auto ys = std::views::iota(0, g->ny);
  auto is = std::views::iota(0, g->q);
  auto xis = std::views::cartesian_product(xs, is);
  auto yis = std::views::cartesian_product(ys, is);
  auto xys = std::views::cartesian_product(xs, ys);
  auto xyis = std::views::cartesian_product(xs,ys, is);

  // nondimentional numbers
  T Re =10000;
  T Ma = 0.125;

  // reference dimensions
  T ulb = Ma * g->cslb;
  T nu = ulb * llb / Re;
  T taubar = nu * g->invCslb2;
  T tau = taubar + 0.5;

  T Tlb = g->nx / ulb;
  // Time-stepping loop parameters
  int num_steps = Tlb; 1000;
  int outputIter = /*50;*/num_steps / 20;

  printf("T_lb = %f\n", Tlb);
  printf("num_steps = %d\n", num_steps);
  printf("warm_up_iter = %d\n", warm_up_iter);
  printf("u_lb = %f\ntau = %f\n", ulb,tau);

//  cylinder_flags_initialization(*g);
  line_segments_flags_initialization(*g,generateNACAAirfoil(std::array<T,2>{g->nx/3.+0.1,g->ny/2.+0.1},g->ny/1.5,g->ny, "2412",-10));


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

      // Access the underlying raw data pointer;
      D2Q9lattice::Flags* flags_data = g->flags_data.data_handle();
      auto flags_iterable = std::span<D2Q9lattice::Flags>(flags_data,g->flags_buffer.size());
      // Cast to `float*`
      std::vector<T> float_data;
      std::transform(flags_iterable.begin(),flags_iterable.end(),std::back_inserter(float_data),[](auto&& flag){return static_cast<T>(flag);});
      // Create the new `mdspan` with `float` type
      exper::mdspan<T, exper::dextents<int, 3>, layout> float_mdspan(float_data.data(),g->nx,g->ny,g->q);

      std::vector<std::pair<std::string, exper::mdspan<T, exper::dextents<int,3>,layout> > > fields = {
          {"velocity", g->velocity_data},
//          {"rho", g->rho_data_},
          {"flags", float_mdspan}
      };

      std::string filename = "output_" + std::to_string(t) + ".vtk";
      auto before_out = std::chrono::high_resolution_clock::now();
      writeVTK2D(filename, std::views::cartesian_product(ys, xs), fields, nx, ny);
      auto after_out = std::chrono::high_resolution_clock::now();

      output_time += after_out - before_out;
    }

    collide_stream_two_populations(*g, ulb, tau);

    std::for_each(std::execution::par_unseq, yis.begin(), yis.end(),
                  [f = g->f_data, nx, ny](auto yi) {
                    auto [y, i] = yi;
                    f(nx - 1, y, i) = f(nx - 2, y, i);
                  });
    std::for_each(std::execution::par_unseq, xis.begin(), xis.end(),
                  [f = g->f_data, nx, ny](auto xi) {
                    auto [x, i] = xi;
                    f(x, ny - 1, i) = f(x, ny - 2, i);
                    f(x, 0, i) = f(x, 1, i);
                  });
  }

  // End time measurement
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time /*- output_time*/;

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