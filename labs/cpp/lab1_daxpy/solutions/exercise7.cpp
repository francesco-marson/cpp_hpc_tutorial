/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <ranges>
#include <algorithm>
#include <execution>
#include <mdspan>

/// Intialize vectors `x` and `y`: parallel algorithm version
void initialize(std::vector<double> &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  std::for_each_n(std::execution::par, std::views::iota(0).begin(), x.size(), [x = x.data()](int i) {
    x[i] = (double)i;
  });
  std::fill_n(std::execution::par, y.data(), y.size(), 2.);
}

/// 2D DAXPY: AX + Y: parallel algorithm version
void daxpy(double a, std::vector<double> &x, std::vector<double> &y, size_t ncols = 1) {
  assert(x.size() == y.size());
  if (x.size() % ncols != 0) { 
      std::cerr << "ERROR: size " << x.size() << " not divisible by " << ncols << std::endl; 
      std::abort(); 
  }
  size_t nrows = x.size() / ncols;

  // DONE: construct extents object.
  std::dextents<size_t, 2> extents(nrows, ncols);
  // DONE: construct the layout mapping object:
  auto mapping = std::layout_left::mapping(extents);
  // DONE: construct the mdspans from the mapping:
  std::mdspan xs { x.data(), mapping };
  std::mdspan ys { y.data(), mapping };
  std::for_each_n(std::execution::par,
                  std::views::iota(0).begin(), nrows, [=](int row) {
        for (size_t col = 0; col < ncols; ++col) {
            ys(row, col) += a * xs(row, col);
        }
  });
}

// Check solution
bool check(double a, std::vector<double> const &y);

int main(int argc, char *argv[]) {
  // Read CLI arguments, the first argument is the name of the binary:
  if (argc != 2) {
    std::cerr << "ERROR: Missing length argument!" << std::endl;
    return 1;
  }

  // Read length of vector elements
  long long n = std::stoll(argv[1]);

  // Allocate the vector
  std::vector<double> x(n, 0.), y(n, 0.);
  double a = 2.0;

  initialize(x, y);

  daxpy(a, x, y);

  if (!check(a, y)) {
    std::cerr << "ERROR!" << std::endl;
    return 1;
  }

  std::cerr << "Check: OK, ";

  // Measure bandwidth in [GB/s]
  using clk_t = std::chrono::steady_clock;
  daxpy(a, x, y);
  auto start = clk_t::now();
  int nit = 100;
  for (int it = 0; it < nit; ++it) {
    daxpy(a, x, y);
  }
  auto seconds = std::chrono::duration<double>(clk_t::now() - start).count(); // Duration in [s]
  // Amount of bytes transferred from/to chip.
  // x is read, y is read and written:
  auto gigabytes = 3. * (double)x.size() * (double)sizeof(double) * (double)nit * 1.e-9; // GB
  auto sz_gb = 2. * (double)x.size() * (double)sizeof(double) * 1e-9;
  std::cerr << "Problem size: " << sz_gb << " [GB], Bandwidth [GB/s]: " << (gigabytes / seconds) << std::endl;

  return 0;
}

bool check(double a, std::vector<double> const &y) {
  double tolerance = 2. * std::numeric_limits<double>::epsilon();
  for (std::size_t i = 0; i < y.size(); ++i) {
    double should = a * i + 2.;
    if (std::abs(y[i] - should) > tolerance)
      return false;
  }
  return true;
}
