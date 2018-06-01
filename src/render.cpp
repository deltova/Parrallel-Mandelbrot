#include "render.hpp"
#include <mmintrin.h>
#include <atomic>
#include <pmmintrin.h>
#include <immintrin.h>
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include <cstdint>
#include <cassert>
#include <iostream>
#include <vector>
#include <queue>
#include <numeric>
#include <mutex>
#include <fstream>

struct rgb8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

rgb8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  constexpr float x0 = 1.f / 4.f;
  constexpr float x1 = 2.f / 4.f;
  constexpr float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgb8_t{0, g, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgb8_t{0, 255, b};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgb8_t{r, 255, 0};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgb8_t{255, b, 0};
  }
}

float histogramm(int histo[], int n_iter, int iter_t, int total, int n_iterations)
{
    float hue = 0;
    for (int iter = 0; iter != iter_t + 1; ++iter)
        hue += histo[iter];
    return hue / (float)total;
}

__m256 load_float_256(int val, int max)
{
    float a[8] __attribute__ ((aligned (16)));
    for (int i = 0; i < 8; ++i)
    {
        a[i] = -2.5 + (((float)(val + i) / ((float)max - 1)) * 3.5);
    }
    return *(__m256*)a;
}
__m256 y_load_float_256(int val, int max)
{
    float a[8] __attribute__ ((aligned (16)));
    for (int i = 0; i < 8; ++i)
    {
        a[i] = -1.0 + (((float)val / (float)(max - 1)) * 2.0);
    }
    return *(__m256*)a;
}

__m256 load_single_float(float val)
{
    float a[8] __attribute__ ((aligned (16)));
    for (int i = 0; i < 8; ++i)
    {
        a[i] = val;
    }
    return *(__m256*)a;
}
__m256 load_single_int(int val)
{
    int a[8] __attribute__ ((aligned (16)));
    for (int i = 0; i < 8; ++i)
    {
        a[i] = val;
    }
    return *(__m256*)a;
}

void render_unsmid(std::byte* buffer,
            int width,
            int height,
            std::ptrdiff_t stride,
            int n_iterations)
{
  //std::vector<int> histo(n_iterations, 0);
  auto max_y = height / 2;
  if (height % 2 != 0)
    max_y++;
  int histo[n_iterations];
  memset(histo, 0, n_iterations * sizeof(int));
  int iter_history[width * height];

  //calculate every iteration
  for (int y = 0; y < max_y; ++y)
  {
    for (int x = 0; x < width; x++)
    {
        float x0 = -2.5 + (((float)(x) / ((float)width - 1)) * 3.5);
        auto y0 = -1.0 + (((float)y / (float)(height - 1)) * 2.0);

        auto x_float = 0.0;
        auto y_float = 0.0;
        int iter = 0;
        //x_float * x_float + y_float * y_float < 2.0 * 2.0 &&
        for (; x_float * x_float + y_float * y_float < 4.0
             && iter < n_iterations; ++iter)
        {
            //x_float * x_float - y_float * y_float + x0;
            auto xtemp = x_float * x_float - y_float * y_float + x0;
            y_float = 2.0 * x_float * y_float + y0;
            x_float = xtemp;
        }
        histo[iter]++;
        iter_history[y * width + x] = iter;
    }
  }

  int total = 0;
  for (int i = 0; i < n_iterations; ++i)
    total += histo[i];

  for (int y = 0; y < max_y; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      rgb8_t* str = reinterpret_cast<rgb8_t*>(buffer + stride * y + x * sizeof(rgb8_t));
      rgb8_t* str2 = reinterpret_cast<rgb8_t*>(buffer + stride * (height - y - 1) + x * sizeof(rgb8_t));
      int current = iter_history[y * width + x];
      if (current == n_iterations)
        *str = rgb8_t{0,0,0};
      else
      {
        *str = heat_lut(histogramm(histo, n_iterations, current, total,
        n_iterations));
      }
      *str2 = *str;
    }
  }
}

void render(std::byte* buffer,
            int width,
            int height,
            std::ptrdiff_t stride,
            int n_iterations)
{
  //std::vector<int> histo(n_iterations, 0);
  auto max_y = height / 2;
  if (height % 2 != 0)
    max_y++;
  int histo[n_iterations];
  memset(histo, 0, n_iterations * sizeof(int));
  //int iter_history[width * height];
  std::vector<int> iter_history(width * height);
  //iter_history[0] = 0;

  //calculate every iteration
  for (int y = 0; y < max_y; ++y)
  {
    for (int x = 0; x < width; x += 8)
    {
        auto x0 = load_float_256(x, width);
        auto y0 = y_load_float_256(y, height);

        auto x_float = load_single_float(0.0);
        auto y_float = load_single_float(0.0);
        auto iterations = load_single_int(0);
        int iter = 0;
        //x_float * x_float + y_float * y_float < 2.0 * 2.0 &&
        for (; iter < n_iterations; ++iter)
        {
            auto calc_cond = x_float * x_float + y_float * y_float;
            auto compar = load_single_float(4.0);
            auto mask = _mm256_cmp_ps(calc_cond,  compar, _CMP_LT_OQ);
            //auto test = _mm256_testc_ps(cond, load_single_float(0.0));
            if (!_mm256_movemask_ps(mask))
                break;
            iterations = _mm256_blendv_ps(iterations, iterations +
            load_single_float(1), mask);
            //x_float * x_float - y_float * y_float + x0;
            auto xtemp = x_float * x_float - y_float * y_float + x0;
            y_float = 2.0 * x_float * y_float + y0;
            x_float = xtemp;
        }
        for (int i = 0; i < 8; ++i)
        {
            int nb_iter = iterations[i];
            if (x + i< width ){
                histo[nb_iter]++;
                iter_history[y * width + (x + i)] = nb_iter;
            }
        }
    }
  }

  int total = 0;
  for (int i = 0; i < n_iterations; ++i)
    total += histo[i];

  for (int y = 0; y < max_y; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      rgb8_t* str = reinterpret_cast<rgb8_t*>(buffer + stride * y + x * sizeof(rgb8_t));
      rgb8_t* str2 = reinterpret_cast<rgb8_t*>(buffer + stride * (height - y - 1) + x * sizeof(rgb8_t));
      int current = iter_history[y * width + x];
      if (current == n_iterations)
        *str = rgb8_t{0,0,0};
      else
      {
        *str = heat_lut(histogramm(histo, n_iterations, current, total,
        n_iterations));
      }
      *str2 = *str;
    }
  }
}

void render_mt(std::byte* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations)
{
  auto max_x = height / 2;
  if (height % 2 != 0)
    max_x++;
  std::atomic<int> histo_ato[n_iterations];
  for (int i = 0; i < n_iterations; ++i)
    histo_ato[i].store(0, std::memory_order_relaxed);
  std::vector<int> iter_history(width * height);
  auto inner_loop_history = [&](int y) {
    for (int x = 0; x < width; x += 8)
    {
        auto x0 = load_float_256(x, width);
        auto y0 = y_load_float_256(y, height);
        auto x_float = _mm256_set1_ps(0.0);
        auto y_float = _mm256_set1_ps(0.0);
        auto iterations = load_single_int(0);
        int iter = 0;
        for (; iter < n_iterations; ++iter)
        {
            auto calc_cond = x_float * x_float + y_float * y_float;
            auto compar = _mm256_set1_ps(4.0);
            auto mask = _mm256_cmp_ps(calc_cond,  compar, _CMP_LT_OQ);
            if (!_mm256_movemask_ps(mask))
                break;
            iterations = _mm256_blendv_ps(iterations, iterations +
            _mm256_set1_ps(1), mask);
            auto xtemp = x_float * x_float - y_float * y_float + x0;
            y_float = 2.0 * x_float * y_float + y0;
            x_float = xtemp;
        }

        for (int i = 0; i < 8; ++i)
        {
            if (x + i < width)
            {
                histo_ato[(int)iterations[i]].fetch_add(1);;
                iter_history[y * width + (x + i)] = iterations[i];
            }
        }
    }
  };

  /*for (int i = 0; i < height / 2 + 1; ++i)
    inner_loop_history(i);*/
  tbb::parallel_for(0, max_x, 1, inner_loop_history);
  int histo[n_iterations];
  int total = 0;
  auto accumulate = [&](int i){
    auto str = histo_ato[i].load(std::memory_order_relaxed);
    histo[i] = str;
    total += str;
  };
  for (int i = 0; i < n_iterations; ++i) accumulate(i);
  auto inner_loop = [&](int y)
  {
    for (int x = 0; x < width; ++x)
    {
      int current = iter_history[y * width + x];
      rgb8_t* str = reinterpret_cast<rgb8_t*>(buffer + stride * y + x * sizeof(rgb8_t));
      rgb8_t* str2 = reinterpret_cast<rgb8_t*>(buffer + stride * (height - y - 1) + x * sizeof(rgb8_t));
      if (current == n_iterations)
        *str = rgb8_t{0,0,0};
      else
        *str = heat_lut(histogramm(histo, n_iterations, current, total,
        n_iterations));
      *str2 = *str;
    }
  };
  tbb::parallel_for(0, max_x, 1, inner_loop);
  //for (auto toto = 0; toto < height / 2; ++toto) inner_loop(toto);
}
