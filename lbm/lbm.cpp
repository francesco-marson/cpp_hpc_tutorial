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
#include <experimental/linalg>
#include "statistics.h"

// Define a type alias for convenience
using T = float;
namespace exper = std::experimental;
namespace linalg = std::experimental::linalg;
using layout = exper::layout_left;
using rnk2 = exper::dextents<int, 2>;
using rnk3 = exper::dextents<int, 3>;
using rnk4 = exper::dextents<int, 4>;

#include <variant>
#include <iostream>

template<typename T, typename LayoutPolicy>
using MdspanVariant = std::variant<
exper::mdspan<T, rnk2, LayoutPolicy>,
exper::mdspan<T, rnk3, LayoutPolicy>,
exper::mdspan<T, rnk4, LayoutPolicy>
>;

template<typename T, typename LayoutPolicy>
void writeVTK2D(
        const std::string &filename,
        tl::cartesian_product_view<std::ranges::iota_view<int, int>, std::ranges::iota_view<int, int>> grid,
        std::vector<std::pair<std::string, MdspanVariant<T, LayoutPolicy>>> fields,
        int NX, int NY,
        const std::vector <std::array<T, 2>> *segments = nullptr // Optional segment overlay
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

    for (const auto& [field_name, field_data_variant] : fields) {
        std::visit([&](auto&& field_data) {
            using MDSpanType = std::decay_t<decltype(field_data)>;
            constexpr int Rank = MDSpanType::rank_dynamic(); // Determine the rank of the mdspan

            int num_components = 1;
            int num_components2 = 1;
            if constexpr (Rank == 3) {
            num_components = field_data.extent(Rank - 1);
            }
            else if constexpr (Rank == 4) {
            num_components = field_data.extent(Rank - 2);
            num_components2 = field_data.extent(Rank - 1);
            }

            if (Rank == 2 or num_components == 1) {
                out << "SCALARS " << field_name << " double\n";
                out << "LOOKUP_TABLE default\n";
            } else if (Rank == 3 and num_components <= 3) {
                out << "VECTORS " << field_name << " double\n";
            } else {
                out << "TENSORS6 " << field_name << " double\n";
            }

            for (const auto& [iy, ix] : grid) {
//                if constexpr(Rank == 2) out << std::scientific << field_data(ix, iy) << ' ';
//                else if constexpr(Rank == 3) out << std::scientific << field_data(ix, iy, 0) << ' ';
//                else if constexpr(Rank == 4) out << std::scientific << field_data(ix, iy, 0,0) << ' ';
                for (int ic = 0; ic < num_components; ++ic) {
                    if constexpr(Rank == 2) out << std::scientific << field_data(ix, iy) << ' ';
                    else if constexpr(Rank == 3) out << std::scientific << field_data(ix, iy, ic) << ' ';
                    else if constexpr(Rank == 4)
                    {
                        for (int ic2 = 0; ic2 < num_components2; ++ic2)
                            out << std::scientific << field_data(ix, iy, ic, ic2) << ' ';
                    };
                }
                if (Rank == 3 and num_components == 2) {
                    out << 0 << ' ';
                }
                else if (Rank == 4 and num_components == 2) {
                    out << 0 << ' ' << 0 << ' ';
                }
                out << '\n';
            }
        }, field_data_variant);
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
  const int c[2][9] = {
        { 0, 1, 0, -1, 0, 1, -1, -1, 1 },
        { 0, 0, 1, 0, -1, 1, 1, -1, -1 }
    };
  static constexpr std::array<T, 9> cnormSqrt = {0, 1, 1, 1, 1, 2, 2, 2, 2};
  const std::array<T, 9> cnorm = {0, 1, 1, 1, 1, sqrt((T)2), sqrt((T)2), sqrt((T)2), sqrt((T)2)};
  static constexpr int d = 2;
  static constexpr int q = 9;
  const int number_dynamic_scalars = q;
  const std::array<int, 9> opposite = {0, 3, 4, 1, 2, 7, 8, 5, 6};
  // Precomputed indices for 90° rotations
  // index =                                      {0, 1, 2, 3, 4, 5, 6, 7, 8};
//  const std::array<int, 9> clockwise_90 =     {0, 4, 1, 2, 3, 8, 5, 6, 7};
//  const std::array<int, 9> anticlockwise_90 = {0, 2, 3, 4, 1, 6, 7, 8, 5};
  T cslb = 1. / sqrt(3.);
  T cslb2 = cslb * cslb;
  T invCslb = 1. / cslb;
  T invCslb2 = invCslb * invCslb;

  std::vector<T> buffer;
  std::vector<Flags> flags_buffer;
    exper::mdspan<T, rnk2, layout> rhob_matrix,turbulent_energy_matrix,turbulent_dissipation_matrix;
  exper::mdspan<T, rnk3,layout> f_matrix, f_matrix_2, dynamic_matrix,velocity_matrix,rhob_matrix_;
  exper::mdspan<T, rnk4,layout> strain_matrix;

    // Take a subspan that only considers the first two indices
// Define the full type for the submdspan
  exper::mdspan<Flags, rnk3,layout> flags_matrix;

  D2Q9lattice(int nx, int ny, T llb) : nx(nx), ny(ny), llb(llb),
                                       buffer(nx * ny * q * 2 + nx * ny * 3 + nx * ny * number_dynamic_scalars +
                                              nx * ny * 4 + nx * ny * 2, 1.0),
                                                                       flags_buffer(nx * ny * q,bulk)

  {
    f_matrix = exper::mdspan<T, rnk3,layout>(buffer.data(), nx, ny, q);
    f_matrix_2 = exper::mdspan<T, rnk3,layout>(buffer.data() + f_matrix.size(), nx, ny, q);
    auto velocity_matrix_starting_index = buffer.data() + f_matrix.size()+ f_matrix_2.size();
    velocity_matrix = exper::mdspan<T, rnk3,layout>(velocity_matrix_starting_index, nx, ny, d);
    auto rhob_matrix_starting_index = velocity_matrix_starting_index + velocity_matrix.size();
    rhob_matrix_ = exper::mdspan<T, rnk3,layout>(rhob_matrix_starting_index, nx, ny,0);
    auto dynamic_matrix_starting_index = rhob_matrix_starting_index + rhob_matrix_.size();
    dynamic_matrix = exper::mdspan<T, rnk3,layout>(dynamic_matrix_starting_index, nx, ny,number_dynamic_scalars);
    auto stresses_starting_index = dynamic_matrix_starting_index + dynamic_matrix.size();
    strain_matrix = exper::mdspan<T, rnk4,layout>(stresses_starting_index, nx, ny,2,2);
      auto turbulent_energy_starting_index = stresses_starting_index + strain_matrix.size();
      turbulent_energy_matrix = exper::mdspan<T, rnk2, layout>(turbulent_energy_starting_index, nx, ny);
      auto turbulent_dissipation_starting_index = turbulent_energy_starting_index + turbulent_energy_matrix.size();
      turbulent_dissipation_matrix = exper::mdspan<T, rnk2, layout>(turbulent_dissipation_starting_index, nx, ny);

    flags_matrix = exper::mdspan<Flags, rnk3,layout>(flags_buffer.data(), nx, ny,q);

      // Take the subspan that only considers the first two indexes
    rhob_matrix = exper::submdspan(rhob_matrix_, exper::full_extent, exper::full_extent, 0/*std::make_pair(0, 1)*/);
    initialize();
  }

  void swap() {
    std::swap(f_matrix, f_matrix_2);
  }

  inline int oppositeIndex(int i) const {
    return opposite[i];
  }

  void initialize() {
    for (int x = 0; x < nx; ++x) {
      for (int y = 0; y < ny; ++y) {
        T rhob_val = 0.0;
        T ux = 0.0;
        T uy = 0.0;
        rhob_matrix(x, y) = rhob_val;
        velocity_matrix(x, y, 0) = ux;
        velocity_matrix(x, y, 1) = uy;
        for (int i = 0; i < 9; ++i) {
          T cu = 3.0 * (cx[i] * ux + cy[i] * uy);
          T u_sq = 1.5 * (ux * ux + uy * uy);
          f_matrix(x, y, i) = w[i] * (rhob_val+1.0) * (1 + cu + 0.5 * cu * cu - u_sq)-w[i];
          f_matrix_2(x, y, i) = f_matrix(x, y, i); // Initialize f_2 the same way as f
        }
      }
    }
  }
};


// Data structure for the D2Q9lattice
struct D2Q9latticePalabos {
  enum Flags{bulk,hwbb,inlet,outlet,symmetry};
  int nx, ny;
  const T llb;
  const T w[9] = {
          (T)4/(T)9, (T)1/(T)36, (T)1/(T)9, (T)1/(T)36, (T)1/(T)9,
                     (T)1/(T)36, (T)1/(T)9, (T)1/(T)36, (T)1/(T)9
  };
  const std::array<int, 9> cx = {0, -1, -1, -1,  0,  1, 1, 1, 0};
  const std::array<int, 9> cy = {0,  1,  0, -1, -1, -1, 0, 1, 1};
    const int c[2][9] = {
            {0, -1, -1, -1,  0,  1, 1, 1, 0},
            {0,  1,  0, -1, -1, -1, 0, 1, 1}
    };
  static constexpr std::array<T, 9> cnormSqrt = { 0, 2, 1, 2, 1, 2, 1, 2, 1 };
  const std::array<T, 9> cnorm = { 0, sqrt((T)2), 1, sqrt((T)2), 1, sqrt((T)2), 1, sqrt((T)2), 1 };
  static constexpr int d = 2;
  static constexpr int q = 9;
  const int number_dynamic_scalars = q;
  const std::array<int, 9> opposite = {0,5,6,7,8,1,2,3,4};
  // Precomputed indices for 90° rotations
  // index =                                      {0, 1, 2, 3, 4, 5, 6, 7, 8};
//  const std::array<int, 9> clockwise_90 =     {0, 4, 1, 2, 3, 8, 5, 6, 7};
//  const std::array<int, 9> anticlockwise_90 = {0, 2, 3, 4, 1, 6, 7, 8, 5};
  T cslb = 1. / sqrt(3.);
  T cslb2 = cslb * cslb;
  T invCslb = 1. / cslb;
  T invCslb2 = invCslb * invCslb;

  std::vector<T> buffer;
  std::vector<Flags> flags_buffer;
    exper::mdspan<T, rnk2, layout> rhob_matrix,turbulent_energy_matrix,turbulent_dissipation_matrix;
  exper::mdspan<T, rnk3,layout> f_matrix, f_matrix_2, dynamic_matrix,velocity_matrix,rhob_matrix_;
  exper::mdspan<T, rnk4,layout> strain_matrix;

    // Take a subspan that only considers the first two indices
// Define the full type for the submdspan
  exper::mdspan<Flags, rnk3,layout> flags_matrix;

    D2Q9latticePalabos(int nx, int ny, T llb) : nx(nx), ny(ny), llb(llb),
                         buffer(nx * ny * q * 2 + nx * ny * 3+ nx * ny * number_dynamic_scalars+ nx * ny * 4+
                                                                                                 nx * ny * 4 + nx * ny * 2, 1.0),
                                                                       flags_buffer(nx * ny * q,bulk)

  {
    f_matrix = exper::mdspan<T, rnk3,layout>(buffer.data(), nx, ny, q);
    f_matrix_2 = exper::mdspan<T, rnk3,layout>(buffer.data() + f_matrix.size(), nx, ny, q);
    auto velocity_matrix_starting_index = buffer.data() + f_matrix.size()+ f_matrix_2.size();
    velocity_matrix = exper::mdspan<T, rnk3,layout>(velocity_matrix_starting_index, nx, ny, d);
    auto rhob_matrix_starting_index = velocity_matrix_starting_index + velocity_matrix.size();
    rhob_matrix_ = exper::mdspan<T, rnk3,layout>(rhob_matrix_starting_index, nx, ny,0);
    auto dynamic_matrix_starting_index = rhob_matrix_starting_index + rhob_matrix_.size();
    dynamic_matrix = exper::mdspan<T, rnk3,layout>(dynamic_matrix_starting_index, nx, ny,number_dynamic_scalars);
    auto stresses_starting_index = dynamic_matrix_starting_index + dynamic_matrix.size();
    strain_matrix = exper::mdspan<T, rnk4,layout>(stresses_starting_index, nx, ny,2,2);
      auto turbulent_energy_starting_index = stresses_starting_index + strain_matrix.size();
      turbulent_energy_matrix = exper::mdspan<T, rnk2, layout>(turbulent_energy_starting_index, nx, ny);
      auto turbulent_dissipation_starting_index = turbulent_energy_starting_index + turbulent_energy_matrix.size();
      turbulent_dissipation_matrix = exper::mdspan<T, rnk2, layout>(turbulent_dissipation_starting_index, nx, ny);

    flags_matrix = exper::mdspan<Flags, rnk3,layout>(flags_buffer.data(), nx, ny,q);

      // Take the subspan that only considers the first two indexes
    rhob_matrix = exper::submdspan(rhob_matrix_, exper::full_extent, exper::full_extent, 0/*std::make_pair(0, 1)*/);
    initialize();
  }

  void swap() {
    std::swap(f_matrix, f_matrix_2);
  }

  inline int oppositeIndex(int i) const {
    return opposite[i];
  }

  void initialize() {
    for (int x = 0; x < nx; ++x) {
      for (int y = 0; y < ny; ++y) {
        T rhob_val = 0.0;
        T ux = 0.0;
        T uy = 0.0;
        rhob_matrix(x, y) = rhob_val;
        velocity_matrix(x, y, 0) = ux;
        velocity_matrix(x, y, 1) = uy;
        for (int i = 0; i < 9; ++i) {
          T cu = 3.0 * (cx[i] * ux + cy[i] * uy);
          T u_sq = 1.5 * (ux * ux + uy * uy);
          f_matrix(x, y, i) = w[i] * (rhob_val+1.0) * (1 + cu + 0.5 * cu * cu - u_sq)-w[i];
          f_matrix_2(x, y, i) = f_matrix(x, y, i); // Initialize f_2 the same way as f
        }
      }
    }
  }
};


struct D2Q37lattice {
    enum Flags { bulk, hwbb, inlet, outlet, symmetry };
    int nx, ny;
    const T llb;
    const T w[37] = {
            0.23315066913235250228650, 0.00028341425299419821740, 0.00535304900051377523273,
            0.05766785988879488203006, 0.00101193759267357547541, 0.00535304900051377523273,
            0.00028341425299419821740, 0.10730609154221900241246, 0.01420821615845075026469,
            0.00024530102775771734547, 0.00028341425299419821740, 0.00535304900051377523273,
            0.05766785988879488203006, 0.00101193759267357547541, 0.00535304900051377523273,
            0.00028341425299419821740, 0.10730609154221900241246, 0.01420821615845075026469,
            0.00024530102775771734547, 0.00028341425299419821740, 0.00535304900051377523273,
            0.05766785988879488203006, 0.00101193759267357547541, 0.00535304900051377523273,
            0.00028341425299419821740, 0.10730609154221900241246, 0.01420821615845075026469,
            0.00024530102775771734547, 0.00028341425299419821740, 0.00535304900051377523273,
            0.05766785988879488203006, 0.00101193759267357547541, 0.00535304900051377523273,
            0.00028341425299419821740, 0.10730609154221900241246, 0.01420821615845075026469,
            0.00024530102775771734547};
    const std::array<int, 37> cx = {0, -1, -1, -1, -2, -2, -3, -1, -2, -3, -3, -2, -1, -2, -1, -1, 0, 0, \
0, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3, 2, 1, 2, 1, 1, 0, 0, 0};
    const std::array<int, 37> cy = {0, 3, 2, 1, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -3, -1, -2, -3, \
-3, -2, -1, -2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 1, 2, 3};
    const int c[2][37] ={
            {0, -1, -1, -1, -2, -2, -3, -1, -2, -3, -3, -2, -1, -2, -1, -1, 0, 0, \
0, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3, 2, 1, 2, 1, 1, 0, 0, 0},
            {0, 3, 2, 1, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -3, -1, -2, -3, \
-3, -2, -1, -2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 1, 2, 3}
    };
    const std::array<T, 37> cnormSqr = {
            0,  10, 5, 2, 8, 5,  10, 1, 4, 9,  10, 5, 2, 8, 5,  10, 1, 4, 9,
            10, 5,  2, 8, 5, 10, 1,  4, 9, 10, 5,  2, 8, 5, 10, 1,  4, 9};
    const std::array<T, 37> cnorm = {0,sqrt(10),sqrt(5),sqrt(2),2*sqrt(2),sqrt(5),sqrt(10),1,2,3,
                                     sqrt(10),sqrt(5),sqrt(2),2*sqrt(2),sqrt(5),sqrt(10),1,2,3,
                                     sqrt(10),sqrt(5),sqrt(2),2*sqrt(2),sqrt(5),sqrt(10),1,2,3,
                                     sqrt(10),sqrt(5),sqrt(2),2*sqrt(2),sqrt(5),sqrt(10),1,2,3};
    const int d = 2;
    const int q = 37;
    const int number_dynamic_scalars = q;
    const std::array<int, 37> opposite = {0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, \
35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    T cslb2 = 1. / (1.19697977039307435897239 * 1.19697977039307435897239);
    T cslb = sqrt(cslb2);
    T invCslb2 = 1.19697977039307435897239 * 1.19697977039307435897239;
    T invCslb = sqrt(invCslb2);

    std::vector<T> buffer;
    std::vector<Flags> flags_buffer;
    exper::mdspan<T, rnk2, layout> rhob_matrix;
    exper::mdspan<T, rnk3, layout> f_matrix, f_matrix_2, dynamic_matrix, velocity_matrix, rhob_matrix_;
    exper::mdspan<T, rnk4, layout> strain_matrix;

    // Define the full type for the submdspan
    exper::mdspan<Flags, rnk3, layout> flags_matrix;

    D2Q37lattice(int nx, int ny, T llb)
            : nx(nx), ny(ny), llb(llb),
              buffer(nx * ny * q * 2 + nx * ny * 3 + nx * ny * number_dynamic_scalars + nx * ny * 4, 1.0),
              flags_buffer(nx * ny * q, bulk)
    {
        f_matrix = exper::mdspan<T, rnk3, layout>(buffer.data(), nx, ny, q);
        f_matrix_2 = exper::mdspan<T, rnk3, layout>(buffer.data() + f_matrix.size(), nx, ny, q);
        auto velocity_matrix_starting_index = buffer.data() + f_matrix.size() + f_matrix_2.size();
        velocity_matrix = exper::mdspan<T, rnk3, layout>(velocity_matrix_starting_index, nx, ny, d);

        auto rhob_matrix_starting_index = velocity_matrix_starting_index + velocity_matrix.size();
        rhob_matrix_ = exper::mdspan<T, rnk3, layout>(rhob_matrix_starting_index, nx, ny, 0);

        auto dynamic_matrix_starting_index = rhob_matrix_starting_index + rhob_matrix_.size();
        dynamic_matrix = exper::mdspan<T, rnk3, layout>(dynamic_matrix_starting_index, nx, ny, number_dynamic_scalars);

        auto stresses_starting_index = dynamic_matrix_starting_index + dynamic_matrix.size();
        strain_matrix = exper::mdspan<T, rnk4, layout>(stresses_starting_index, nx, ny, 2, 2);

        flags_matrix = exper::mdspan<Flags, rnk3, layout>(flags_buffer.data(), nx, ny, q);

        rhob_matrix = exper::submdspan(rhob_matrix_, exper::full_extent, exper::full_extent, 0);

        initialize();
    }

    void swap() {
        std::swap(f_matrix, f_matrix_2);
    }

    inline int oppositeIndex(int i) const {
        return opposite[i];
    }

    void initialize() {
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                T rhob_val = 0.0;
                T ux = 0.0;
                T uy = 0.0;
                rhob_matrix(x, y) = rhob_val;
                velocity_matrix(x, y, 0) = ux;
                velocity_matrix(x, y, 1) = uy;
                for (int i = 0; i < 37; ++i) {
                    T cu = 3.0 * (cx[i] * ux + cy[i] * uy);
                    T u_sq = 1.5 * (ux * ux + uy * uy);
                    f_matrix(x, y, i) = w[i] * (rhob_val + 1.0) * (1 + cu + 0.5 * cu * cu - u_sq) - w[i];
                    f_matrix_2(x, y, i) = f_matrix(x, y, i);
                }
            }
        }
    }
};


// Function to compute gradient of m2sgs
void computeGradient(const exper::mdspan<T, rnk4, layout>& m2sgs,
                     exper::mdspan<T, rnk3, layout>& dm2sgs) {

    auto nx = m2sgs.extent(0);
    auto ny = m2sgs.extent(1);

    // Check extents matching
    assert(nx == dm2sgs.extent(0) && ny == dm2sgs.extent(1));

    // Cartesian product of indexes
    auto xs = std::views::iota(0, nx);
    auto ys = std::views::iota(0, ny);
    auto msgs_is = std::views::iota(0, 2); // assuming m2sgs has 2x2 matrices, i.e., i and j are in {0,1}
    auto xys = std::views::cartesian_product(msgs_is, ys, xs);

    // Compute the gradient in parallel
    std::for_each(std::execution::par, xys.begin(), xys.end(),
                  [&dm2sgs, &m2sgs, nx, ny](auto coord) {
                      auto [msgs_i, y, x] = coord;
                      T grad_x = 0.0;
                      T grad_y = 0.0;

                      // 6th order finite differences coefficients
                      constexpr T coeff[4] = {1.0 / 60.0, -3.0 / 20.0, 3.0 / 4.0, 0.0};

                      // Calculate gradient in x direction for m2sgs(x, y, msgs_i, 0)
                      for (int k = 1; k <= 3; ++k) {
                          int xp_k = (x + k + nx) % nx;
                          int xm_k = (x - k + nx) % nx;

                          grad_x += coeff[k - 1] * (m2sgs(xp_k, y, msgs_i, 0) - m2sgs(xm_k, y, msgs_i, 0));
                      }
                      grad_x /= 2; // Because we consider central difference

                      // Calculate gradient in y direction for m2sgs(x, y, msgs_i, 1)
                      for (int k = 1; k <= 3; ++k) {
                          int yp_k = (y + k + ny) % ny;
                          int ym_k = (y - k + ny) % ny;

                          grad_y += coeff[k - 1] * (m2sgs(x, yp_k, msgs_i, 1) - m2sgs(x, ym_k, msgs_i, 1));
                      }
                      grad_y /= 2; // Because we consider central difference

                      // Sum both derivatives and store them in dm2sgs(x, y, msgs_i)
                      dm2sgs(x, y, msgs_i) = grad_x + grad_y;
                  });
}


std::vector<T> getFiniteDifferenceCoefficients(int order) {
    switch (order) {
        case 2:
            return {1.0 / 2.0};
        case 4:
            return {2.0 / 3.0, -1.0 / 12.0};
        case 6:
            return {3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
        case 8:
            return {4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0};
        default:
            throw std::invalid_argument("Unsupported order");
    }
}


void computeTurbulentEnergy(const exper::mdspan <T, rnk3, layout> &velocity,
                            exper::mdspan <T, rnk2, layout> &turbulent_energy,
                            int order = 6) {
    auto nx = velocity.extent(0);
    auto ny = velocity.extent(1);

    // Check extents matching
    assert(nx == turbulent_energy.extent(0) && ny == turbulent_energy.extent(1));

    // Get the coefficients and kernel half-width for a specific order
    const auto coeff = getFiniteDifferenceCoefficients(order);
    int kernel_half_width = (coeff.size() - 1) / 2 + 1;

    // Cartesian product of indexes
    auto xs = std::views::iota(0, nx);
    auto ys = std::views::iota(0, ny);
    auto xys = std::views::cartesian_product(xs, ys);

    // Compute turbulent kinetic energy in parallel
    std::for_each(std::execution::par, xys.begin(), xys.end(),
                  [&turbulent_energy, &velocity, nx, ny](auto coord) {
                      auto [x, y] = coord;

                      // Get velocity fluctuations (assuming mean flow is subtracted)
                      T u_prime = velocity(x, y, 0);
                      T v_prime = velocity(x, y, 1);

                      // Compute turbulent kinetic energy as 0.5 * (u'²  + v'²)
                      turbulent_energy(x, y) = 0.5 * (u_prime * u_prime + v_prime * v_prime);
                  });
}

void computeTurbulentDissipation(const exper::mdspan<T, rnk3, layout>& velocity,
                                 exper::mdspan<T, rnk2, layout>& turbulent_dissipation,
                                 const exper::mdspan<T, rnk4, layout>& strain_tensor,
                                 int order = 6) {
    auto nx = velocity.extent(0);
    auto ny = velocity.extent(1);

    // Check extents matching
    assert(nx == turbulent_dissipation.extent(0) && ny == turbulent_dissipation.extent(1));

    // Get the coefficients and kernel half-width for a specific order
    const auto coeff = getFiniteDifferenceCoefficients(order);
    int kernel_half_width = (coeff.size() - 1) / 2 + 1;

    // Cartesian product of indexes
    auto xs = std::views::iota(0, nx);
    auto ys = std::views::iota(0, ny);
    auto xys = std::views::cartesian_product(xs, ys);

    // Compute the turbulent dissipation in parallel
    std::for_each(std::execution::par, xys.begin(), xys.end(),
                  [&turbulent_dissipation, &strain_tensor, nx, ny](auto coord) {
                      auto [x, y] = coord;

                      // Compute the magnitude of the strain rate tensor
                      T Sxx = strain_tensor(x, y, 0, 0);
                      T Syy = strain_tensor(x, y, 1, 1);
                      T Sxy = strain_tensor(x, y, 0, 1);

                      // Compute the turbulent dissipation rate
                      T dissipation = 2.0 * (Sxx * Sxx + Syy * Syy + 2.0 * Sxy * Sxy);

                      // Store the result in the turbulent_dissipation matrix
                      turbulent_dissipation(x, y) = dissipation;
                  });
}


void computeStrainTensor(const exper::mdspan<T, rnk3, layout>& velocity,
                         exper::mdspan<T, rnk4, layout>& strain_tensor,
                         int order = 6) {

    auto nx = velocity.extent(0);
    auto ny = velocity.extent(1);

    // Check extents matching
    assert(nx == strain_tensor.extent(0) && ny == strain_tensor.extent(1));

    // Get the coefficients and kernel half-width for a specific order
    const auto coeff = getFiniteDifferenceCoefficients(order);
    int kernel_half_width = (coeff.size() - 1) / 2 + 1;

    // Cartesian product of indexes
    auto xs = std::views::iota(0, nx);
    auto ys = std::views::iota(0, ny);
    auto xys = std::views::cartesian_product(xs, ys);

    // Compute the strain tensor in parallel
    std::for_each(std::execution::par, xys.begin(), xys.end(),
                  [&strain_tensor, &velocity, nx, ny, coeff=coeff.data(), kernel_half_width](auto coord) {
                      auto [x, y] = coord;
                      T du_dx = 0.0;
                      T du_dy = 0.0;
                      T dv_dx = 0.0;
                      T dv_dy = 0.0;

                      // Calculate gradient in the x direction for u and v components
                      for (int k = 1; k <= kernel_half_width; ++k) {
                          int xp_k = (x + k + nx) % nx;
                          int xm_k = (x - k + nx) % nx;

                          du_dx += coeff[k - 1] * (velocity(xp_k, y, 0) - velocity(xm_k, y, 0));
                          dv_dx += coeff[k - 1] * (velocity(xp_k, y, 1) - velocity(xm_k, y, 1));
                      }
                      du_dx /= 2; // Central difference adjustment
                      dv_dx /= 2; // Central difference adjustment

                      // Calculate gradient in the y direction for u and v components
                      for (int k = 1; k <= kernel_half_width; ++k) {
                          int yp_k = (y + k + ny) % ny;
                          int ym_k = (y - k + ny) % ny;

                          du_dy += coeff[k - 1] * (velocity(x, yp_k, 0) - velocity(x, ym_k, 0));
                          dv_dy += coeff[k - 1] * (velocity(x, yp_k, 1) - velocity(x, ym_k, 1));
                      }
                      du_dy /= 2; // Central difference adjustment
                      dv_dy /= 2; // Central difference adjustment

                      // Compute the components of the strain tensor
                      strain_tensor(x, y, 0, 0) = 2 * du_dx;      // Tau_xx
                      strain_tensor(x, y, 1, 1) = 2 * dv_dy;      // Tau_yy
                      strain_tensor(x, y, 0, 1) = (du_dy + dv_dx); // Tau_xy
                      strain_tensor(x, y, 1, 0) = strain_tensor(x, y, 0, 1);  // Tau_yx = Tau_xy
                  });
}



template <typename Lattice>
void computeMoments(Lattice& g, bool even = true, bool before_cs = true) {
    assert(before_cs);
    auto xs = std::views::iota(0, g.nx);
    auto ys = std::views::iota(0, g.ny);
    auto coords = std::views::cartesian_product(ys, xs);

    // Parallel loop to compute moments
    std::for_each(std::execution::par_unseq, coords.begin(), coords.end(), [&g, even](auto coord) {
        auto [y, x] = coord;
        T rhob = 0.0, ux = 0.0, uy = 0.0;
        for (int i = 0; i < g.q; ++i) {
            int iPop = even ? i : g.opposite[i];
            rhob += g.f_matrix(x, y, iPop);
            ux += g.f_matrix(x, y, iPop) * g.cx[i];
            uy += g.f_matrix(x, y, iPop) * g.cy[i];
        }
        g.rhob_matrix_(x, y, 0) = rhob;
        T rho = rhob + 1.0;
        g.velocity_matrix(x, y, 0) = ux / rho;
        g.velocity_matrix(x, y, 1) = uy / rho;
    });
}

std::vector<std::array<T,2> > generateNACAAirfoil(std::array<T,2> origin, uint length, unsigned int tesselation,
                                                  const std::string& naca, double aoa_deg) {
    std::vector<std::array<T,2> > points;
    double t = std::stoi(naca.substr(2, 2)) / 100.0; // Thickness
    double m = std::stoi(naca.substr(0, 1)) / 100.0; // Maximum camber
    double p = std::stoi(naca.substr(1, 1)) / 10.0;  // Position of maximum camber

    double aoa_rad = M_PI * aoa_deg / 180.0; // Convert aoa to radians

    for (unsigned int i = 0; i <= tesselation; ++i) {
        double x = ((double)i) / tesselation;
        double yt = 5 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * pow(x, 2) + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));
        double yc = 0.0; // Default for symmetric airfoils

        // Calculate camber line only if m > 0 (non-symmetric airfoil)
        if (m > 0) {
            if (x <= p) {
                yc = m / pow(p, 2) * (2 * p * x - pow(x, 2));
            } else {
                yc = m / pow(1-p, 2) * ((1 - 2 * p) + 2 * p * x - pow(x, 2));
            }
        }

        double theta = (m > 0) ? atan(m / pow(p, 2) * (2 * p - 2 * x)) : 0.0; // Angle for rotation

        // Apply rotation using rotation matrix
        double x1 = (x - yt * sin(theta)) * length;
        double x2 = (yc + yt * cos(theta)) * length;
        double xr = x1 * cos(aoa_rad) - x2 * sin(aoa_rad);
        double yr = x1 * sin(aoa_rad) + x2 * cos(aoa_rad);

        points.push_back(std::array<T,2>{xr + origin[0], yr + origin[1]});
    }

    for (int i = tesselation-1; i >= 0; --i) {
        double x = ((double)i) / tesselation;
        double yt = 5 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * pow(x, 2) + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));
        double yc = 0.0; // Default for symmetric airfoils

        // Calculate camber line only if m > 0 (non-symmetric airfoil)
        if (m > 0) {
            if (x <= p) {
                yc = m / pow(p, 2) * (2 * p * x - pow(x, 2));
            } else {
                yc = m / pow(1-p, 2) * ((1 - 2 * p) + 2 * p * x - pow(x, 2));
            }
        }

        double theta = (m > 0) ? atan(m / pow(p, 2) * (2 * p - 2 * x)) : 0.0; // Angle for rotation

        // Apply rotation using rotation matrix
        double x1 = (x + yt * sin(theta)) * length;
        double x2 = (yc - yt * cos(theta)) * length;
        double xr = x1 * cos(aoa_rad) - x2 * sin(aoa_rad);
        double yr = x1 * sin(aoa_rad) + x2 * cos(aoa_rad);

        points.push_back(std::array<T,2>{xr + origin[0], yr + origin[1]});
    }

    // Add closing segment if needed
    if (points.front()[0] != points.back()[0] || points.front()[1] != points.back()[1]) {
        points.push_back(points.front());  // Add first point again to close the contour
    }

    return points;
}

std::vector<std::array<T,2> > generateNACAAirfoil_non_uniform(std::array<T,2> origin, uint length, unsigned int tesselation,
                                                  const std::string& naca, double aoa_deg) {
    std::vector<std::array<T,2> > points;
    points.reserve(2 * tesselation + 2); // Pre-allocate memory

    double t = std::stoi(naca.substr(2, 2)) / 100.0; // Thickness
    double m = std::stoi(naca.substr(0, 1)) / 100.0; // Maximum camber
    double p = std::stoi(naca.substr(1, 1)) / 10.0;  // Position of maximum camber

    double aoa_rad = M_PI * aoa_deg / 180.0; // Convert aoa to radians

    // Use finer tessellation near leading and trailing edges
    auto generatePoint = [&](double x, bool upper) -> std::array<T, 2> {
        // NACA thickness distribution
        double x_root = std::sqrt(x); // Use square root for finer leading edge resolution
        double yt =
                5 * t * (0.2969 * x_root - 0.1260 * x - 0.3516 * pow(x, 2) + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));

        double yc = 0.0;
        double dycdx = 0.0;

        // Calculate camber line and its derivative only if m > 0 (non-symmetric airfoil)
        if (m > 0) {
            if (x <= p) {
                yc = m / pow(p, 2) * (2 * p * x - pow(x, 2));
                dycdx = 2 * m / pow(p, 2) * (p - x);
            } else {
                yc = m / pow(1-p, 2) * ((1 - 2 * p) + 2 * p * x - pow(x, 2));
                dycdx = 2 * m / pow(1 - p, 2) * (p - x);
            }
        }

        double theta = atan(dycdx);
        double sign = upper ? 1 : -1;

        // Apply rotation using rotation matrix with improved precision
        double x1 = (x - sign * yt * sin(theta)) * length;
        double x2 = (yc + sign * yt * cos(theta)) * length;
        double xr = x1 * cos(aoa_rad) - x2 * sin(aoa_rad);
        double yr = x1 * sin(aoa_rad) + x2 * cos(aoa_rad);

        return std::array < T, 2 > {xr + origin[0], yr + origin[1]};
    };

    // Generate points with non-uniform spacing (clustered near leading and trailing edges)
    auto firstPoint = generatePoint(0.0, true); // Store first point
    points.push_back(firstPoint);

    for (unsigned int i = 1; i <= tesselation; ++i) {
        double x = 0.5 * (1.0 - cos(M_PI * i / tesselation)); // Cosine spacing
        points.push_back(generatePoint(x, true));
    }

    // Generate lower surface points (reverse order)
    for (int i = tesselation-1; i >= 0; --i) {
        double x = 0.5 * (1.0 - cos(M_PI * i / tesselation)); // Cosine spacing
        points.push_back(generatePoint(x, false));
    }

    // Add first point again to close the airfoil
    points.push_back(firstPoint); // Use stored first point for exact closure

    // Ensure minimum segment length and add intermediate points if necessary
    const double min_segment_length = length / (10.0 * tesselation);
    std::vector <std::array<T, 2>> refined_points;
    refined_points.reserve(points.size() * 2);

    for (size_t i = 0; i < points.size() - 1; ++i) {
        refined_points.push_back(points[i]);

        double dx = points[i + 1][0] - points[i][0];
        double dy = points[i + 1][1] - points[i][1];
        double segment_length = std::sqrt(dx * dx + dy * dy);

// More aggressive refinement for better intersection detection
        if (segment_length > min_segment_length) {
            int n_intermediate = static_cast<int>(std::ceil(segment_length / (min_segment_length * 0.5)));
            for (int j = 1; j < n_intermediate; ++j) {
                double t = static_cast<double>(j) / n_intermediate;
                refined_points.push_back(std::array < T, 2 > {
                        points[i][0] + t * dx,
                        points[i][1] + t * dy
                });
            }
    }
    }
    refined_points.push_back(points.back()); // Add final point (same as first)

    // Verify closure
    const double closure_tolerance = 1e-10;
    bool is_closed = std::abs(refined_points.front()[0] - refined_points.back()[0]) < closure_tolerance &&
                     std::abs(refined_points.front()[1] - refined_points.back()[1]) < closure_tolerance;

    if (!is_closed) {
        std::cerr << "Warning: Airfoil curve is not properly closed!" << std::endl;
    }

    return refined_points;
}


template<typename Lattice>
auto line_segments_flags_initialization(Lattice& g, const std::vector<std::array<T, 2>>& segments){
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

    // Lambda for line segment intersection with improved numerical stability
    auto segment_intersect_segment = [segments,getMinimumPositive](T x1, T y1, T x2, T y2) -> std::optional<T> {
        std::vector<T> intersections;
        const T EPSILON = std::numeric_limits<T>::epsilon(); // Use system epsilon for type T

        for (auto it = segments.begin(); it != segments.end(); ++it){
            const auto& segment = *it;
            const auto& segment_next = it != std::prev(segments.end()) ? *(it +1) : segments.front();
            T s1_x = x2 - x1;
            T s1_y = y2 - y1;
            T s2_x = segment_next[0] - segment[0];
            T s2_y = segment_next[1] - segment[1];

            // Check if lines are parallel (cross product near zero)
            T denominator = (-s2_x * s1_y + s1_x * s2_y);
            if (std::abs(denominator) < EPSILON) continue;

            T s = (-s1_y * (x1 - segment[0]) + s1_x * (y1 - segment[1])) / denominator;
            T t = (s2_x * (y1 - segment[1]) - s2_y * (x1 - segment[0])) / denominator;

            // Slightly expand the intersection acceptance range
            if (s >= -EPSILON && s <= 1 + EPSILON && t >= -EPSILON && t <= 1 + EPSILON) {
                // Clamp values to valid range
                s = std::clamp(s, 0.0f, 1.0f);
                t = std::clamp(t, 0.0f, 1.0f);

                T distance = std::sqrt(s1_x * s1_x + s1_y * s1_y) * t;
                if (distance > EPSILON) { // Only include non-zero distances
                    intersections.push_back(distance);
                }
            }
        }

        return getMinimumPositive(intersections);
    };

    std::for_each(xyis.begin(),xyis.end(),[&g,segment_intersect_segment](auto xyi){
        auto [x,y,i] = xyi;

        if (x == 0 and g.cx[i] == -1)
            g.flags_matrix(x, y, i) = g.inlet;
        else if (x == (g.nx - 1) and g.cx[i] > 0)
            g.flags_matrix(x, y, i) = g.outlet;
//        else if (y == 0 and g.cy[i] < 0)
//            g.flags_matrix(x, y, i) = g.outlet;
//        else if (y == g.ny-1 and g.cy[i] > 0)
//            g.flags_matrix(x, y, i) = g.outlet;
        else {
            auto intersection =
                    segment_intersect_segment(x, y, x + g.cx[i], y + g.cy[i]);
            if (intersection.has_value() and intersection.value() <= g.cnorm[i]) {
                g.flags_matrix(x, y, i) = g.hwbb;
                g.dynamic_matrix(x, y, i) = intersection.value();
            } else {
                g.flags_matrix(x, y, i) = g.bulk;
            }
        }
    });
}

template<typename Lattice>
auto cylinder_flags_initialization(Lattice& g){
  // indexes
  auto xs = std::views::iota(0, g.nx);
  auto ys = std::views::iota(0, g.ny);
  auto is = std::views::iota(0, g.q);
  auto iyxs = std::views::cartesian_product(is,ys, xs);



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
  std::for_each(iyxs.begin(),iyxs.end(),[&g,cx,cy,radius,circle_intersect_segment,getMinimumPositive,is_near](auto iyx){
    auto [i,y,x] = iyx;

    if (x == 0 and g.cx[i] == -1)
      g.flags_matrix(x, y, i) = g.inlet;
    else if (x == (g.nx - 1) and g.cx[i] > 0)
      g.flags_matrix(x, y, i) = g.outlet;
    else if (/*x > 0 and */y == 0 and g.cy[i] < 0)
      g.flags_matrix(x, y, i) = g.outlet;
    else if (/*x > 0 and */y == g.ny-1 and g.cy[i] > 0)
      g.flags_matrix(x, y, i) = g.outlet;
    else if (is_near(x, y)) {
      auto intersection =
          getMinimumPositive(circle_intersect_segment(x, y, x + g.cx[i], y + g.cy[i]));
      if (intersection.has_value() and intersection.value() < g.cnorm[i]) {
        g.flags_matrix(x, y, i) = g.hwbb;
        g.dynamic_matrix(x, y, i) = intersection.value();
      }
    } else /*if (not is_near(x,y))*/ g.flags_matrix(x, y, i) = g.bulk;
});
}

struct CM{

enum {
    // Order 0
    M00 = 0,

    // Order 1
    M10 = 1,
    M01 = 2,

    // Order 2
    M20 = 3,
    M02 = 4,
    M11 = 5,

    // Order 3
    M21 = 6,
    M12 = 7,

    // Order 4
    M22 = 8,
};
enum { F00 = 0, FMP = 1, FM0 = 2, FMM = 3, F0M = 4, FPM = 5, FP0 = 6, FPP = 7, F0P = 8 };// use this to generalize

// Optimized way to compute CHs based on Palabos ordering of discrete velocities
std::array<T,9> HMcomputeMoments(
        std::array<T,9> const &cell, const D2Q9latticePalabos& g)
{
    std::array<T,9> HM;
    std::array<T, 9> f;
    for (int i = 0; i < g.q; ++i) {
    f[i] = cell[i] + g.w[i];
    HM[i] = 0.;
    }
    
    double X_M1 = f[1] + f[2] + f[3];
    double X_P1 = f[5] + f[6] + f[7];
    double X_0 = f[0] + f[4] + f[8];
    
    double Y_M1 = f[3] + f[4] + f[5];
    double Y_P1 = f[1] + f[7] + f[8];
    
    // Order 0
    HM[M00] = X_M1 + X_P1 + X_0;
    
    // Order 1
    HM[M10] = X_P1 - X_M1;
    HM[M01] = Y_P1 - Y_M1;
    
    // Order 2
    HM[M20] = X_P1 + X_M1;
    HM[M02] = Y_P1 + Y_M1;
    HM[M11] = -f[1] + f[3] - f[5] + f[7];
    
    // Order 3
    HM[M21] = f[1] - f[3] - f[5] + f[7];
    HM[M12] = -f[1] - f[3] + f[5] + f[7];
    
    // Order 4
    HM[M22] = f[1] + f[3] + f[5] + f[7];
    
    T rho = HM[M00];
    T invRho = 1. / rho;
    for (int i = 0; i < g.q; ++i) {
        HM[i] *= invRho;
    }
    
    // We come back to Hermite moments
    T cs4 = g.cslb2 * g.cslb2;
    HM[M20] -= g.cslb2;
    HM[M02] -= g.cslb2;
    
    HM[M21] -= g.cslb2 * HM[M01];
    HM[M12] -= g.cslb2 * HM[M10];
    
    HM[M22] -= (g.cslb2 * (HM[M20] + HM[M02]) + cs4);

    return HM;
};

auto HMcomputeEquilibriumMoments(T rho, std::array<T,2> u, const std::array<T, 9> tmpf, D2Q9latticePalabos& g) -> const std::array<T,9>
{
    std::array<T,9> HMeq;
    // Order 0
    HMeq[M00] = rho;
    
    // Order 1
    HMeq[M10] = u[0];
    HMeq[M01] = u[1];
    
    // Order 2
    HMeq[M20] = u[0] * u[0];
    HMeq[M02] = u[1] * u[1];
    HMeq[M11] = u[0] * u[1];
    
    // Order 3
    HMeq[M21] = HMeq[M20] * u[1];
    HMeq[M12] = HMeq[M02] * u[0];
    
    // Order 4
    HMeq[M22] = HMeq[M20] * HMeq[M02];
    return HMeq;
};


auto HMcomputeEquilibriumMomentsClosure(T rho, std::array<T,2> u, T uxx, T uxy, T uyy, const std::array<T, 9> tmpf, D2Q9latticePalabos& g) -> const std::array<T,9>
{
    std::array<T,9> HMeq;
    // Order 0
    HMeq[M00] = rho;

    // Order 1
    HMeq[M10] = u[0];
    HMeq[M01] = u[1];

    // Order 2
    HMeq[M20] = uxx;
//    HMeq[M20] = u[0] * u[0];
    HMeq[M02] = uyy;
//    HMeq[M02] = u[1] * u[1];
    HMeq[M11] = uxy;
//    HMeq[M11] = u[0] * u[1];

    // Order 3
    HMeq[M21] = HMeq[M20] * u[1];
    HMeq[M12] = HMeq[M02] * u[0];
//    HMeq[M21] = u[0]*u[0] * u[1];
//    HMeq[M12] = u[1]*u[1] * u[0];

    // Order 4
    HMeq[M22] = u[0]*u[0] * u[1]*u[1];
    return HMeq;
};


  // General way to compute HMs
  std::array<T,9> HMcomputeMoments2(
          std::array<T,9> const cell, const D2Q9latticePalabos& g){

        std::array<T,9> HM;
      std::array<T, 9> f;
      for (int i = 0; i<9; ++i) {
          f[i] = cell[i] + g.w[i];
          HM[i] = 0.;
      }

      T Hxx = 0.;
      T Hyy = 0.;

      for (int i = 0; i<9; ++i) {

          Hxx = g.cx[i] * g.cx[i] - g.cslb2;
          Hyy = g.cy[i] * g.cy[i] - g.cslb2;

          // Order 0
          HM[M00] += f[i];

          // Order 1
          HM[M10] += g.cx[i] * f[i];
          HM[M01] += g.cy[i] * f[i];

          // Order 2
          HM[M20] += Hxx * f[i];
          HM[M02] += Hyy * f[i];
          HM[M11] += g.cx[i] * g.cy[i] * f[i];

          // Order 3
          HM[M21] += Hxx * g.cy[i] * f[i];
          HM[M12] += g.cx[i] * Hyy * f[i];

          // Order 4
          HM[M22] += Hxx * Hyy * f[i];
      }

      T rho = HM[M00];
      T invRho = 1. / rho;
      for (int i = 0; i<9; ++i) {
          HM[i] *= invRho;
      }
      return HM;
  }

// Equilibrium populations based on 9 moments can be computed using either RM, HM, CM, HM or
// Gauss-Hermite formalisms. Here we use central hermite moments (HMs)
void HMcomputeEquilibrium(
        T rho, std::array<T,9> const &HMeq, std::array<T,9> &eq, T cs2)
{
    std::array<T, 2> u(HMeq[1], HMeq[2]);
    std::array<T, 9> RMeq;
    T cs4 = cs2 * cs2;

    RMeq[M20] = HMeq[M20] + cs2;
    RMeq[M02] = HMeq[M02] + cs2;

    RMeq[M11] = HMeq[M11];

    RMeq[M21] = HMeq[M21] + cs2 * u[1];
    RMeq[M12] = HMeq[M12] + cs2 * u[0];

    RMeq[M22] = HMeq[M22] + cs2 * (HMeq[M20] + HMeq[M02]) + cs4;

    eq[F00] = rho * (1. - RMeq[M20] - RMeq[M02] + RMeq[M22]);

    eq[FP0] = 0.5 * rho * (u[0] + RMeq[M20] - RMeq[M12] - RMeq[M22]);
    eq[FM0] = 0.5 * rho * (-u[0] + RMeq[M20] + RMeq[M12] - RMeq[M22]);

    eq[F0P] = 0.5 * rho * (u[1] + RMeq[M02] - RMeq[M21] - RMeq[M22]);
    eq[F0M] = 0.5 * rho * (-u[1] + RMeq[M02] + RMeq[M21] - RMeq[M22]);

    eq[FPP] = 0.25 * rho * (RMeq[M11] + RMeq[M21] + RMeq[M12] + RMeq[M22]);
    eq[FMP] = 0.25 * rho * (-RMeq[M11] + RMeq[M21] - RMeq[M12] + RMeq[M22]);
    eq[FPM] = 0.25 * rho * (-RMeq[M11] - RMeq[M21] + RMeq[M12] + RMeq[M22]);
    eq[FMM] = 0.25 * rho * (RMeq[M11] - RMeq[M21] - RMeq[M12] + RMeq[M22]);
}

std::array<T,9> HMcollide(
        T rho,  std::array<T, 2> const &u,
        std::array<T,9> const &HM,    // hermite moments
        std::array<T,9> const &HMeq,  // Equilibrium moments (hermite)
        std::array<T, 9> const &omega, D2Q9latticePalabos& g)
{
    std::array<T,9> cell;
    T omega1 = omega[0];
    T omega2 = omega[1];
    T omega3 = omega[2];
    T omega4 = omega[3];

    T omegaBulk = omega[4];
    T omegaPlus = (omegaBulk + omega1) / 2.;   // Notation used by Fei
    T omegaMinus = (omegaBulk - omega1) / 2.;  // Notation used by Fei

    T cs4 = g.cslb2 * g.cslb2;

    // Post-collision moments.
    std::array<T, 9> HMcoll;
    std::array<T, 9> RMcoll;

    // Collision in the Hermite moment space
    // Order 2 (non-diagonal collision so that we can easily modify the bulk viscosity)
    HMcoll[M20] =
            HM[M20] - omegaPlus * (HM[M20] - HMeq[M20]) - omegaMinus * (HM[M02] - HMeq[M02]);
    HMcoll[M02] =
            HM[M02] - omegaMinus * (HM[M20] - HMeq[M20]) - omegaPlus * (HM[M02] - HMeq[M02]);

    HMcoll[M11] = (1. - omega2) * HM[M11] + omega2 * HMeq[M11];

    // Order 3
    HMcoll[M21] = (1. - omega3) * HM[M21] + omega3 * HMeq[M21];
    HMcoll[M12] = (1. - omega3) * HM[M12] + omega3 * HMeq[M12];

    // Order 4
    HMcoll[M22] = (1. - omega4) * HM[M22] + omega4 * HMeq[M22];

    // Come back to RMcoll using relationships between HMs and RMs
    RMcoll[M20] = HMcoll[M20] + g.cslb2;
    RMcoll[M02] = HMcoll[M02] + g.cslb2;

    RMcoll[M11] = HMcoll[M11];

    RMcoll[M21] = HMcoll[M21] + g.cslb2 * u[1];
    RMcoll[M12] = HMcoll[M12] + g.cslb2 * u[0];

    RMcoll[M22] = HMcoll[M22] + g.cslb2 * (HMcoll[M20] + HMcoll[M02]) + cs4;

    // Compute post collision populations from RM
    cell[F00] = rho * (1. - RMcoll[M20] - RMcoll[M02] + RMcoll[M22]);

    cell[FP0] = 0.5 * rho * (u[0] + RMcoll[M20] - RMcoll[M12] - RMcoll[M22]);
    cell[FM0] = 0.5 * rho * (-u[0] + RMcoll[M20] + RMcoll[M12] - RMcoll[M22]);

    cell[F0P] = 0.5 * rho * (u[1] + RMcoll[M02] - RMcoll[M21] - RMcoll[M22]);
    cell[F0M] = 0.5 * rho * (-u[1] + RMcoll[M02] + RMcoll[M21] - RMcoll[M22]);

    cell[FPP] = 0.25 * rho * (RMcoll[M11] + RMcoll[M21] + RMcoll[M12] + RMcoll[M22]);
    cell[FMP] = 0.25 * rho * (-RMcoll[M11] + RMcoll[M21] - RMcoll[M12] + RMcoll[M22]);
    cell[FPM] = 0.25 * rho * (-RMcoll[M11] - RMcoll[M21] + RMcoll[M12] + RMcoll[M22]);
    cell[FMM] = 0.25 * rho * (RMcoll[M11] - RMcoll[M21] - RMcoll[M12] + RMcoll[M22]);

    for (int i = 0; i < g.q; ++i) {
        cell[i] -= g.w[i];
    }
    return cell;
}
};

template <typename Lattice>
void collide_stream_two_populations(Lattice &g, T ulb, T tau) {
  T omega = 1.0 / tau;

    // indexes
    auto xs = std::views::iota(0, g.nx);
    auto ys = std::views::iota(0, g.ny);
    auto is = std::views::iota(0, g.q);
    auto ixs = std::views::cartesian_product(is, xs);
    auto iys = std::views::cartesian_product(is, ys);
    auto yxs = std::views::cartesian_product(ys, xs);
    auto xys = std::views::cartesian_product(xs, ys);
    auto iyxs = std::views::cartesian_product(is,ys, xs);
    CM cm;


  // Parallel loop ensuring thread safety
  std::for_each(std::execution::par_unseq, yxs.begin(), yxs.end(), [&g, omega,tau, ulb,&cm](auto idx) {
    auto [y,x] = idx;
    std::array<T, 9> tmpf{0};
    std::array<T, 9> feq{0};
    T rhob = 0.0, ux = 0.0, uy = 0.0;
    T uxx = 0.0, uxy = 0.0, uyy = 0.0;
    for (int i = 0; i < g.q; ++i) {
      tmpf[i] = g.f_matrix(x, y, i);
      rhob += tmpf[i];
      ux   += (tmpf[i]+g.w[i]) * g.cx[i];
      uy   += (tmpf[i]+g.w[i]) * g.cy[i];
      uxx  += (tmpf[i]+g.w[i]) * (g.cx[i]* g.cx[i]-g.cslb2);
      uyy  += (tmpf[i]+g.w[i]) * (g.cy[i]* g.cy[i]-g.cslb2);
      uxy  += (tmpf[i]+g.w[i]) * g.cx[i]* g.cy[i];
    }
    std::array<T,2> u {ux,uy};
    auto alpha = std::views::iota(0,2);
    auto beta = std::views::iota(0,2);
    auto ab = std::views::cartesian_product(alpha,beta);
//macroscopic density and velocities
    rhob = g.rhob_matrix(x,y);
    T rho = rhob +1.0;
    ux /= rho;
    uy /= rho;
//    uxx += -(tau*(tau-0.5)-(tau-0.5))*g.cslb2*rho*g.strain_matrix(x,y,0,0);
//    uxy += -(tau*(tau-0.5)-(tau-0.5))*g.cslb2*rho*g.strain_matrix(x,y,0,1);
//    uyy += -(tau*(tau-0.5)-(tau-0.5))*g.cslb2*rho*g.strain_matrix(x,y,1,1);
    uxx /= rho;
    uxy /= rho;
    uyy /= rho;
//    ux = g.velocity_matrix(x,y,0);
//    uy = g.velocity_matrix(x,y,1);
      std::array<T,9> ffneq{0};
      for (int i = 0; i < g.q; ++i) {
          std::for_each(ab.begin(),ab.end(),[&ffneq = ffneq[i],&g,i,&tau,&rho,&x,&y](auto&& idx){
              auto [b,a] = idx;
              ffneq +=  (g.c[a][i]* g.c[b][i]-g.cslb2*(a==b?1:0))*g.strain_matrix(x,y,a,b)*0.5;
          });
          ffneq[i] *= -g.w[i]*rho/g.cslb2;
      }
      // Compute
      auto HMeq = cm.HMcomputeEquilibriumMoments(rho,std::array<T,2>{ux,uy},tmpf,g);
//      auto HMeq = cm.HMcomputeEquilibriumMomentsClosure(rho,std::array<T,2>{ux,uy},uxx,uxy,uyy,tmpf,g);
//      auto HM = cm.HMcomputeMoments2(tmpf,g);
//      auto HMneq = cm.HMcomputeMoments2(ffneq,g);
//      std::array<T,9> HMeq2{0};
//      for (int i = 0; i < 9; ++i) {
//          HMeq2[i] = HM[i]-HMeq[i];
//      }
//      // Order 3
//      HMeq2[cm.M21] = u[0]*u[0] * u[1];
//      HMeq2[cm.M12] =  u[1]* u[1] * u[0];
//
//      // Order 4
//      HMeq2[cm.M22] = HMeq2[cm.M20] * HMeq2[cm.M02];

//      HMeq[cm.M20] = HM[cm.M20];
//      HMeq[cm.M02] = HM[cm.M02];
//      HMeq[cm.M11] = HM[cm.M11];
//      HMeq = HM;
      T omega2 = omega;//1./(tau-0.5);
//      omega = 2;
      feq = cm.HMcollide(rho,std::array<T,2>{ux,uy},HMeq,HMeq,std::array<T,9>{omega2,omega2,0.1,0.1,0.1,0.1,1,1,1},g);
//      printf("%f,%f",tmpf[1], tmpf2[1]);


      T Cs = 0.1;
      T delta = 1.;
      // Compute strain rate tensor components
      T Sxx = g.strain_matrix(x,y,0,0)*0.5;
      T Syy = g.strain_matrix(x,y,1,1)*0.5;
      T Sxy = g.strain_matrix(x,y,1,0)*0.5;

      // Compute the magnitude of the strain rate tensor
      T S_mag = std::sqrt(2.0 * (Sxx * Sxx + Syy * Syy + 2.0 * Sxy * Sxy));

      // Compute the eddy viscosity
      T eddy_viscosity = 2.*(Cs * delta) * (Cs * delta) * S_mag;
      T tau_eddy = eddy_viscosity/g.cslb2+0.5/*+tau*/;
      T viscosity = (tau -0.5)*g.cslb2;
      T tau_smago = eddy_viscosity/g.cslb2+tau;
      T omega_eddy = 1./tau_eddy;
      T omega_smago = 1./tau_smago;
      T timeL = g.turbulent_energy_matrix(x,y)/g.turbulent_dissipation_matrix(x,y);
      T timeK = sqrt(viscosity/g.turbulent_dissipation_matrix(x,y));

//      printf("%e,%e\n",omega,omega_eddy);
    // Compute equilibrium distributions
      for (int i = 0; i < g.q; ++i) {
          auto &iopp = g.opposite[i];
          T cu = g.cx[i] * ux + g.cy[i] * uy;
          T u_sq = ux * ux + uy * uy;
          T cu_sq = cu * cu;
// Third order term
//          double feq_third_order = (1.0 / 6.0) * pow(g.invCslb2, 3) * cu * cu_sq -
//                                   0.5 * g.invCslb2 * cu * u_sq;

// Fourth order term
//          double feq_fourth_order = (1.0 / 24.0) * pow(g.invCslb2, 4) * cu_sq * cu_sq -
//                                    0.5 * pow(g.invCslb2, 2) * cu_sq * u_sq +
//                                    (1.0 / 8.0) * pow(g.invCslb2, 2) * u_sq * u_sq;

//          u_sq = uxx + uyy;
//          cu_sq = g.cx[i] * g.cx[i] * uxx + 2. * g.cx[i] * g.cy[i] * uxy + g.cy[i] * g.cy[i] * uyy;
// Second order term
//          double feq_second_order = 1.0 + g.invCslb2 * cu +
//                                    0.5 * g.invCslb2 * g.invCslb2 * cu_sq -
//                                    0.5 * g.invCslb2 * u_sq;


//          feq[i] = g.w[i] * rho * (feq_second_order + feq_third_order /*+ feq_fourth_order*/) - g.w[i];


//      feq[i] = g.w[i] * rho * (1.0 + g.invCslb2 * cu + 0.5*g.invCslb2*g.invCslb2 * cu_sq - 0.5*g.invCslb2*u_sq)-g.w[i];
//      T feq_iopp = g.w[i] * (rhob+(T)1.0) * (1.0 - 3. * cu + 4.5 * cu_sq - 1.5 * u_sq)-g.w[i];
//
          // Collide step
          T feq_sgs = tmpf[i]-ffneq[i] - feq[i];
          T feq_sgs_opp = tmpf[iopp]-ffneq[iopp] - feq[iopp];
          T fneq = tmpf[i] - feq[i];
          T fneq_opp = tmpf[iopp] - feq[iopp];
          T omega_pr = omega - 1.0;
          T omega_th = omega;
          T omega_nm = 0;
          T omega_sgs = (omega-omega_eddy)*omega/omega_eddy;//0.86*omega;
//          T omega_sgs = 1./timeK;//0.86*omega;
//          tmpf[i] = (1.0 - omega) * tmpf[i] + omega * feq[i];
          tmpf[i] = (1.0 - omega_smago) * tmpf[i] + omega_smago * feq[i];
//          tmpf[i] = (feq[i] + feq_sgs + ffneq[i]) - omega * fneq + omega_sgs * feq_sgs;
//          tmpf[i] = (feq[i] + /*feq_sgs +*/ ffneq[i]) - omega * ffneq[i];

//        T tmpf_iopp = (1.0 - omega) * tmpf[iopp] + omega * feq[iopp];
        T tmpf_iopp = (1.0 - omega_smago) * tmpf[iopp] + omega_smago * feq[iopp];
//        T tmpf_iopp = (feq[iopp] + feq_sgs_opp + ffneq[iopp]) - omega * fneq_opp + omega_sgs * feq_sgs_opp;

          // Streaming with consideration for periodic boundaries
          int x_stream = x + g.cx[i];
          int y_stream = y + g.cy[i];
          int x_stream_periodic = (x_stream + g.nx) % g.nx;
          int y_stream_periodic = (y_stream + g.ny) % g.ny;

//      auto a = g.hwbb;
      // Handle periodic and bounce-back boundary conditions
      if (g.flags_matrix(x,y,i) == g.hwbb) {
          T q = g.dynamic_matrix(x,y,i)/g.cnorm[i];
//        g.f_matrix_2(x, y, iopp) = tmpf[i];
          g.f_matrix_2(x, y, iopp) = q * 0.5*(tmpf[i]+tmpf_iopp) + (1.-q)*0.5*(g.f_matrix(x, y, i)+g.f_matrix(x, y, iopp));
      } else if (g.flags_matrix(x,y,i) == g.inlet) {
        g.f_matrix_2(x, y, g.opposite[i]) = tmpf[i] - 2.0 * g.invCslb2 * (ulb * g.cx[i]) * g.w[i];
      } else if (g.flags_matrix(x,y,i) == g.outlet) {
        //do nothing
      }  else { // periodic
        g.f_matrix_2(x_stream_periodic, y_stream_periodic, i) = tmpf[i];
      }
    }
  });
  g.swap();
}

template <typename T>
void initializeDipoleWallCollision(D2Q9lattice &g, T Re, T nu, T r0, T x1, T y1, T x2, T y2) {
    // indexes
    auto xs = std::views::iota(0, g.nx);
    auto ys = std::views::iota(0, g.ny);
    auto is = std::views::iota(0, g.q);
    auto coords = std::views::cartesian_product(ys, xs);

    T eta_e = (Re * nu) / (r0 * r0);

    int nx = g.nx;
    int ny = g.ny;

    std::for_each(coords.begin(), coords.end(), [&g, eta_e, r0, x1, y1, x2, y2, nx, ny](auto coord) {
        auto [y, x] = coord;
        T rhob_val = 0.0; // Initial density value
        T ux = 0.0;      // Initial x-velocity
        T uy = 0.0;      // Initial y-velocity

        // Radius and calculations for first monopole (positive vorticity)
        T rx1 = static_cast<T>(x) - x1;
        T ry1 = static_cast<T>(y) - y1;
        T r1_squared = rx1 * rx1 + ry1 * ry1;

        // Radius and calculations for second monopole (negative vorticity)
        T rx2 = static_cast<T>(x) - x2;
        T ry2 = static_cast<T>(y) - y2;
        T r2_squared = rx2 * rx2 + ry2 * ry2;

        // Define the velocities as per the given formulas
        T u0 =  - 0.5 * std::abs(eta_e) * (static_cast<T>(y) - y1) * exp(-r1_squared / (r0 * r0))
                + 0.5 * std::abs(eta_e) * (static_cast<T>(y) - y2) * exp(-r2_squared / (r0 * r0));
        T v0 =    0.5 * std::abs(eta_e) * (static_cast<T>(x) - x1) * exp(-r1_squared / (r0 * r0))
                - 0.5 * std::abs(eta_e) * (static_cast<T>(x) - x2) * exp(-r2_squared / (r0 * r0));

        ux = u0;
        uy = v0;

        // Ensure zero velocities at the boundaries as prescribed
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
            ux = 0.0;
            uy = 0.0;
        }

        // Set lattice node values
        g.rhob_matrix_(x, y, 0) = rhob_val;
        g.velocity_matrix(x, y, 0) = ux;
        g.velocity_matrix(x, y, 1) = uy;

        // Initialize equilibrium distribution functions
        for (int i = 0; i < g.q; ++i) {
            T cu = g.cx[i] * ux + g.cy[i] * uy;
            T u_sq =  (ux * ux + uy * uy);
            g.f_matrix(x, y, i) = g.w[i] * (rhob_val+(T)1) * (1.0 + g.invCslb2 * cu + 0.5*g.invCslb2*g.invCslb2 * cu * cu - 0.5*g.invCslb2 *u_sq)-g.w[i];
            g.f_matrix_2(x, y, i) = g.f_matrix(x, y, i); // Initially set f_2 equal to f

            // Handle boundary conditions
            if (x == 0 && g.cx[i] == -1)
                g.flags_matrix(x, y, i) = g.hwbb;
            else if (x == (nx - 1) && g.cx[i] == 1)
                g.flags_matrix(x, y, i) = g.hwbb;
            else if (y == 0 && g.cy[i] == -1)
                g.flags_matrix(x, y, i) = g.hwbb;
            else if (y == (ny - 1) && g.cy[i] == 1)
                g.flags_matrix(x, y, i) = g.hwbb;
            else
                g.flags_matrix(x, y, i) = g.bulk;
        }
    });
}



template <typename T, typename Lattice>
void initializeDoubleShearLayer(Lattice &g, T U0, T alpha=80, T delta=0.05) {
  int nx = g.nx;
  int ny = g.ny;
  T L = nx;

  // Initialize density and velocity fields
  for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
      T rhob_val = 0.0;

      // Define velocities based on the double shear layer profile
      T ux = U0 * std::tanh(alpha * (0.25 - std::abs((static_cast<T>(y) / static_cast<T>(ny)) - 0.5)));
      T uy = U0 * delta * std::sin(2.0 * M_PI * (static_cast<T>(x) / static_cast<T>(nx) + 0.25));

      g.rhob_matrix_(x, y, 0) = rhob_val;
      g.velocity_matrix(x, y, 0) = ux;
      g.velocity_matrix(x, y, 1) = uy;

      // Initialize populations based on the equilibrium distribution
      for (int i = 0; i < g.q; ++i) {
          T cu = g.cx[i] * ux + g.cy[i] * uy;
          T u_sq =  (ux * ux + uy * uy);
          g.f_matrix(x, y, i) = g.w[i] * (rhob_val+(T)1) * (1.0 + g.invCslb2 * cu + 0.5*g.invCslb2*g.invCslb2 * cu * cu - 0.5*g.invCslb2 *u_sq)-g.w[i];
          g.f_matrix_2(x, y, i) = g.f_matrix(x, y, i); // Initially set f_2 equal to f
      }
    }
  }
}
// Generate airfoil with higher resolution
void writeSegmentsVTK(const std::string& filename, const std::vector<std::array<T,2>>& points) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::ios_base::failure("Failed to open file");
    }

    // Write VTK header
    out << "# vtk DataFile Version 3.0\n";
    out << "Line Segments\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";

    // Write points
    out << "POINTS " << points.size() << " float\n";
    for (const auto& point : points) {
        out << point[0] << " " << point[1] << " 0.0\n";
    }

    // Write lines connecting consecutive points
    int numLines = points.size() - 1;
    out << "LINES " << numLines << " " << numLines * 3 << "\n";
    for (int i = 0; i < numLines; ++i) {
        out << "2 " << i << " " << (i + 1) << "\n";
    }

    out.close();
}

int main() {
  int warm_up_iter = 1000;

  // numerical resolution
  int nx = 400;
  int ny = 100;
  T llb = nx/4./*/11.*/;
//  T llb = nx/*/11.*/;

  // Setup D2Q9lattice and initial conditions
  auto g = std::make_unique<D2Q9latticePalabos>(nx, ny,llb);
    auto& gg = *g;

  // indexes
  auto xs = std::views::iota(0, g->nx);
  auto ys = std::views::iota(0, g->ny);
  auto is = std::views::iota(0, g->q);
  auto a1s = std::views::iota(0, 2);
  auto a2s = std::views::iota(0, 2);
  auto ixs = std::views::cartesian_product(is, xs);
  auto iys = std::views::cartesian_product(is, ys);
  auto yxs = std::views::cartesian_product(ys, xs);
  auto iyxs = std::views::cartesian_product(is,ys, xs);
  auto a1a2yxs = std::views::cartesian_product(a1s,a2s,ys, xs);

  // nondimentional numbers
  T Re =1e6;
  T Ma = 0.12;

  // reference dimensions
  T ulb = Ma * g->cslb;
  T nu = ulb * llb / Re;
  T taubar = nu * g->invCslb2;
  T tau = taubar + 0.5;

  T Tlb = g->nx / ulb;
  // Time-stepping loop parameters
  int num_steps = 2.0*Tlb;
  int outputIter = num_steps / 20;
  int start_output = 0.0*Tlb;

  printf("T_lb = %f\n", Tlb);
  printf("num_steps = %d\n", num_steps);
  printf("outputIter = %d\n", outputIter);
  printf("warm_up_iter = %d\n", warm_up_iter);
  printf("u_lb = %f\ntau = %f\nomega=%f\n", ulb,tau,1./tau);



    auto airfoil_points = generateNACAAirfoil(
            std::array < T, 2 > {g->nx / 3. + 0.112, g->ny / 2. + 0.1012},
            g->llb,
            g->llb * 4, // Increased tessellation for better resolution
            "0012",
            -15.2
    );
// Write airfoil segments to VTK file
writeSegmentsVTK("airfoil_segments.vtk", airfoil_points);
    line_segments_flags_initialization(*g, airfoil_points);


  // Initialize the D2Q9lattice with the double shear layer
//  initializeDoubleShearLayer(*g, ulb,(T)100.,(T)0.1);
//  initializeDoubleShearLayer(*g, ulb,(T)80.,(T)0.05);

  // Start time measurement
  auto start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> output_time(0.0);




  // Loop over time steps
  for (int t = 0; t <= num_steps; ++t) {

    if (t == warm_up_iter)
      start_time = std::chrono::high_resolution_clock::now();

    // Toggle between different streaming steps
    //    bool even = (t % 2 == 0);

    // Output results every outputIter iterations
      // Compute macroscopic variables using the new function
      computeMoments(*g); // Dereference the unique_ptr to pass the reference.
      computeStrainTensor(g->velocity_matrix,g->strain_matrix,6);

computeTurbulentEnergy(g->velocity_matrix,g->turbulent_energy_matrix,6);
        computeTurbulentDissipation(g->velocity_matrix, g->turbulent_dissipation_matrix, g->strain_matrix, 6);

    if (t >= start_output and t % outputIter == 0) {

      // Access the underlying raw data pointer;
//      D2Q9lattice::Flags* flags_matrix = g->flags_matrix.data_handle();
//      auto flags_iterable = std::span<D2Q9lattice::Flags>(flags_matrix,g->flags_buffer.size());
//      // Cast to `float*`
//      std::vector<T> float_data;
//      std::transform(flags_iterable.begin(),flags_iterable.end(),std::back_inserter(float_data),[](auto&& flag){return static_cast<T>(flag);});
//      // Create the new `mdspan` with `float` type
//      exper::mdspan<T, rnk3, layout> float_mdspan(float_data.data(),g->nx,g->ny,g->q);


        // Prepare fields for VTK output
        std::vector<std::pair<std::string, MdspanVariant<T, layout>>> fields = {
                {"velocity", g->velocity_matrix},
                {"rhob", g->rhob_matrix},
                {"stresses", g->strain_matrix},
                {"turbulent_energy", g->turbulent_energy_matrix},
                {"turbulent_dissipation", g->turbulent_dissipation_matrix}
        };
//        fields.push_back(std::make_pair("velocity", g->velocity_matrix));



      std::string filename = "output_" + std::to_string(t) + ".vtk";
      auto before_out = std::chrono::high_resolution_clock::now();

        printf("writing results at iter = %d, convective time = %f\n", t, (T)t/(T)Tlb);
writeVTK2D(filename, std::views::cartesian_product(ys, xs), fields, nx, ny/*, &airfoil_points*/);

      auto after_out = std::chrono::high_resolution_clock::now();

      output_time += after_out - before_out;
    }

    collide_stream_two_populations(*g, ulb, tau);

    std::for_each(std::execution::par_unseq, iys.begin(), iys.end(),
                  [&gg, nx, ny](auto iy) {
                    auto [i, y] = iy;
                    if (gg.flags_matrix(nx - 1, y, i) == gg.outlet) gg.f_matrix(nx - 1, y, i) = gg.f_matrix(nx - 2, y, i);
                  });
    std::for_each(std::execution::par_unseq, ixs.begin(), ixs.end(),
                  [&gg, nx, ny](auto ix) {
                    auto [i,x] = ix;
                      if (gg.flags_matrix(x, ny - 1, i) == gg.outlet) gg.f_matrix(x, ny - 1, i) = gg.f_matrix(x, ny - 2, i);
                      if (gg.flags_matrix(x, 0, i)      == gg.outlet) gg.f_matrix(x, 0, i) = gg.f_matrix(x, 1, i);
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

//      auto lastx = exper::submdspan(g->f_matrix, nx-1, exper::full_extent, exper::full_extent);
//      auto beforelastx = exper::submdspan(g->f_matrix, nx-2, exper::full_extent, exper::full_extent);
//      auto lasty = exper::submdspan(g->f_matrix,exper::full_extent, ny-1, exper::full_extent);
//      auto beforelasty = exper::submdspan(g->f_matrix, exper::full_extent, ny-2, exper::full_extent);