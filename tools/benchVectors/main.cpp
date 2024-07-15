#include "plumed/tools/Vector.h"

#include <benchmark/benchmark.h>

#include <array>
#include <random>
#include <vector>

using rng = std::mt19937_64;
using dist_d = std::uniform_real_distribution<double>;
using dist_i = std::uniform_int_distribution<int>;
rng gen(0);
using namespace PLMD;

template <unsigned n>
static void PLMDVectorG_defaultCtor(benchmark::State &state) {
  for (auto _ : state) {
    VectorGeneric<n> v;
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(PLMDVectorG_defaultCtor<1>);
BENCHMARK(PLMDVectorG_defaultCtor<2>);
BENCHMARK(PLMDVectorG_defaultCtor<3>);
BENCHMARK(PLMDVectorG_defaultCtor<4>);
BENCHMARK(PLMDVectorG_defaultCtor<5>);

template <unsigned n>
static void PLMDVectorG_copyCtor(benchmark::State &state) {
  std::vector<VectorGeneric<n>> v(10);
  for (auto _ : state) {
    v[0] = VectorGeneric<n>();
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(PLMDVectorG_copyCtor<1>);
BENCHMARK(PLMDVectorG_copyCtor<2>);
BENCHMARK(PLMDVectorG_copyCtor<3>);
BENCHMARK(PLMDVectorG_copyCtor<4>);
BENCHMARK(PLMDVectorG_copyCtor<5>);

static void PLMDVectorG_InitDouble(benchmark::State &state) {

  dist_d dist(0, 10);
  std::array<double, 3> a;
  // std::vector<double> a(3);
  size_t n = 3;
  for (size_t i = 0; i < n; ++i) {
    a[i] = dist(gen);
  }
  for (auto _ : state) {
    Vector v(a[0], a[1], a[2]);
    benchmark::DoNotOptimize(v);
  }
}

BENCHMARK(PLMDVectorG_InitDouble);

static void PLMDVectorG_InitDouble_increment(benchmark::State &state) {

  dist_d dist(0, 10);
  double a = dist(gen);
  double b = dist(gen);
  double c = dist(gen);

  for (auto _ : state) {
    benchmark::DoNotOptimize(Vector(a++, b++, c++));
  }
}

BENCHMARK(PLMDVectorG_InitDouble_increment);

static void PLMDVectorG_InitWithConversion(benchmark::State &state) {

  dist_i dist(0, 10);
  int a = dist(gen);
  int b = dist(gen);
  int c = dist(gen);

  for (auto _ : state) {
    benchmark::DoNotOptimize(Vector(a++, b++, c++));
  }
}
BENCHMARK(PLMDVectorG_InitWithConversion);

template <typename T>
static void PLMDVectorG_InitWithConversionArg(benchmark::State &state) {

  dist_d dist(0, 10);
  T a = dist(gen);
  T b = dist(gen);
  T c = dist(gen);
  for (auto _ : state) {
    benchmark::DoNotOptimize(Vector(a, b, c));
  }
}
BENCHMARK(PLMDVectorG_InitWithConversionArg<int>);
BENCHMARK(PLMDVectorG_InitWithConversionArg<float>);
BENCHMARK(PLMDVectorG_InitWithConversionArg<double>);

static void PLMDVectorG_Sum(benchmark::State &state) {

  dist_d dist(0, 10);
  Vector a;
  Vector b;
  size_t n = 3;
  for (size_t i = 0; i < n; ++i) {
    a[i] = dist(gen);
    b[i] = dist(gen);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a + b);
  }
}
BENCHMARK(PLMDVectorG_Sum);

static void PLMDVectorG_DotProduct(benchmark::State &state) {

  dist_d dist(0, 10);
  Vector a;
  Vector b;
  size_t n = 3;
  for (size_t i = 0; i < n; ++i) {
    a[i] = dist(gen);
    b[i] = dist(gen);
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(dotProduct(a, b));
  }
}
BENCHMARK(PLMDVectorG_DotProduct);

template <unsigned n>
static void PLMDVectorG_VTimesDouble(benchmark::State &state) {

  dist_d dist(0, 10);
  VectorGeneric<n> a;
  for (size_t i = 0; i < n; ++i) {
    a[i] = dist(gen);
  }
  double b = dist(gen);
  for (auto _ : state) {
    benchmark::DoNotOptimize(a * b);
  }
}
template <unsigned n>
static void PLMDVectorG_DoubleTimesV(benchmark::State &state) {

  dist_d dist(0, 10);
  VectorGeneric<n> b;
  for (size_t i = 0; i < n; ++i) {
    b[i] = dist(gen);
  }
  double a = dist(gen);
  for (auto _ : state) {
    benchmark::DoNotOptimize(a * b);
  }
}

BENCHMARK(PLMDVectorG_VTimesDouble<3>);
BENCHMARK(PLMDVectorG_DoubleTimesV<3>);
BENCHMARK(PLMDVectorG_VTimesDouble<4>);
BENCHMARK(PLMDVectorG_DoubleTimesV<4>);
BENCHMARK(PLMDVectorG_VTimesDouble<8>);
BENCHMARK(PLMDVectorG_DoubleTimesV<8>);
BENCHMARK(PLMDVectorG_VTimesDouble<12>);
BENCHMARK(PLMDVectorG_DoubleTimesV<12>);

template <unsigned n> static void PLMDVectorG_Modulo(benchmark::State &state) {

  dist_d dist(0, 10);
  VectorGeneric<n> a;
  for (size_t i = 0; i < n; ++i) {
    a[i] = dist(gen);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.modulo());
  }
}
BENCHMARK(PLMDVectorG_Modulo<3>);
BENCHMARK(PLMDVectorG_Modulo<4>);
BENCHMARK(PLMDVectorG_Modulo<8>);
BENCHMARK(PLMDVectorG_Modulo<12>);

template <unsigned n>
static void PLMDVectorG_stdVectorModulo2(benchmark::State &state) {

  dist_d dist(0, 10);
  std::vector<VectorGeneric<n>> v(state.range(0));
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = VectorGeneric<n>();
    for (size_t j = 0; j < n; ++j) {
      v[i][j] = dist(gen);
    }
  }
  for (auto _ : state) {
    for (size_t i = 0; i < v.size(); ++i) {
      benchmark::DoNotOptimize(v[i].modulo2());
    }
  }
}
BENCHMARK(PLMDVectorG_stdVectorModulo2<3>)->Range(8, 8 << 8);
BENCHMARK(PLMDVectorG_stdVectorModulo2<4>)->Range(8, 8 << 8);
BENCHMARK(PLMDVectorG_stdVectorModulo2<8>)->Range(8, 8 << 8);
BENCHMARK(PLMDVectorG_stdVectorModulo2<12>)->Range(8, 8 << 8);

template <unsigned n> static void PLMDVectorG_Modulo2(benchmark::State &state) {

  dist_d dist(0, 10);
  VectorGeneric<n> a;
  for (size_t i = 0; i < n; ++i) {
    a[i] = dist(gen);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.modulo2());
  }
}
BENCHMARK(PLMDVectorG_Modulo2<2>);
BENCHMARK(PLMDVectorG_Modulo2<3>);
BENCHMARK(PLMDVectorG_Modulo2<4>);
BENCHMARK(PLMDVectorG_Modulo2<14>);

BENCHMARK_MAIN();