/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2019-2023 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "plumed/config/Config.h"
#include "plumed/tools/DLLoader.h"
#include "plumed/tools/Log.h"
#include "plumed/tools/PlumedHandle.h"
#include "plumed/tools/Random.h"
#include "plumed/tools/Stopwatch.h"
#include "plumed/tools/Tools.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// this is a sugar for changing idea faster about the rng
using generator = std::mt19937;
namespace PLMD {

/// This base class contains members that are movable with default operations
struct KernelBase {
  std::string path;
  std::string plumed_dat;
  PlumedHandle handle;
  Stopwatch stopwatch;
  std::vector<long long int> timings;
  double comparative_timing = -1.0;
  double comparative_timing_error = -1.0;
  KernelBase(const std::string &path_, const std::string &plumed_dat_,
             Log *log_)
      : path(path_), plumed_dat(plumed_dat_), handle([&]() {
          if (path_ == "this")
            return PlumedHandle();
          else
            return PlumedHandle::dlopen(path.c_str());
        }()),
        stopwatch(*log_) {}
};

/// Local structure handling a kernel and the related timers.
/// This structure specifically contain the Log, which needs special treatment
/// in move semantics
struct Kernel : public KernelBase {
  Log *log = nullptr;
  Kernel(const std::string &path_, const std::string &the_plumed_dat, Log *log_)
      : KernelBase(path_, the_plumed_dat, log_), log(log_) {}

  ~Kernel() {
    if (log) {
      (*log) << "\n";
      (*log) << "Kernel:      " << path << "\n";
      (*log) << "Input:       " << plumed_dat << "\n";
      if (comparative_timing > 0.0) {
        (*log).printf("Comparative: %.3f +- %.3f\n", comparative_timing,
                      comparative_timing_error);
      }
    }
  }

  Kernel(Kernel &&other) noexcept
      : KernelBase(std::move(other)), log(other.log) {
    other.log = nullptr; // ensure no log is done in the moved away object
  }

  Kernel &operator=(Kernel &&other) noexcept {
    if (this != &other) {
      KernelBase::operator=(std::move(other));
      log = other.log;
      other.log = nullptr; // ensure no log is done in the moved away object
    }
    return *this;
  }
};

// atom distributions

class UniformSphericalVector {
  // double rminCub;
  double rCub;

public:
  // assuming rmin=0
  UniformSphericalVector(const double rmax)
      : rCub(rmax * rmax * rmax /*-rminCub*/) {}
  PLMD::Vector operator()(Random &rng) {
    double rho = std::cbrt(/*rminCub + */ rng.RandU01() * rCub);
    double theta = std::acos(2.0 * rng.RandU01() - 1.0);
    double phi = 2.0 * PLMD::pi * rng.RandU01();
    return Vector(rho * sin(theta) * cos(phi), rho * sin(theta) * sin(phi),
                  rho * cos(theta));
  }
};

/// Acts as a template for any distribution
struct AtomDistribution {
  virtual void positions(std::vector<Vector> &posToUpdate, unsigned /*step*/,
                         Random &) = 0;
  virtual void box(std::vector<double> &box, unsigned /*natoms*/,
                   unsigned /*step*/, Random &) {
    std::fill(box.begin(), box.end(), 0);
  };
  virtual ~AtomDistribution() noexcept {}
};

struct theLine : public AtomDistribution {
  void positions(std::vector<Vector> &posToUpdate, unsigned step,
                 Random &rng) override {
    auto nat = posToUpdate.size();
    UniformSphericalVector usv(0.5);

    for (unsigned i = 0; i < nat; ++i) {
      posToUpdate[i] = Vector(i, 0, 0) + usv(rng);
    }
  }
};

struct uniformSphere : public AtomDistribution {
  void positions(std::vector<Vector> &posToUpdate, unsigned /*step*/,
                 Random &rng) override {

    // giving more or less a cubic udm of volume for each atom: V=nat
    const double rmax =
        std::cbrt((3.0 / (4.0 * PLMD::pi)) * posToUpdate.size());

    UniformSphericalVector usv(rmax);
    auto s = posToUpdate.begin();
    auto e = posToUpdate.end();
    // I am using the iterators:this is slightly faster,
    //  enough to overcome the cost of the vtable that I added
    for (unsigned i = 0; s != e; ++s, ++i) {
      *s = usv(rng);
    }
  }
  void box(std::vector<double> &box, unsigned natoms, unsigned /*step*/,
           Random &) override {
    const double rmax = 2.0 * std::cbrt((3.0 / (4.0 * PLMD::pi)) * natoms);
    box[0] = rmax;
    box[1] = 0.0;
    box[2] = 0.0;
    box[3] = 0.0;
    box[4] = rmax;
    box[5] = 0.0;
    box[6] = 0.0;
    box[7] = 0.0;
    box[8] = rmax;
  }
};

struct twoGlobs : public AtomDistribution {
  virtual void positions(std::vector<Vector> &posToUpdate, unsigned /*step*/,
                         Random &rng) {
    // I am using two unigform spheres and 2V=n
    const double rmax =
        std::cbrt((3.0 / (8.0 * PLMD::pi)) * posToUpdate.size());

    UniformSphericalVector usv(rmax);
    std::array<Vector, 2> centers{
        PLMD::Vector{0.0, 0.0, 0.0},
        // so they do not overlap
        PLMD::Vector{2.0 * rmax, 2.0 * rmax, 2.0 * rmax}};
    std::generate(posToUpdate.begin(), posToUpdate.end(), [&]() {
      // RandInt is only declared
      //  return usv (rng) + centers[rng.RandInt(1)];
      return usv(rng) + centers[rng.RandU01() > 0.5];
    });
  }

  virtual void box(std::vector<double> &box, unsigned natoms, unsigned /*step*/,
                   Random &) {

    const double rmax = 4.0 * std::cbrt((3.0 / (8.0 * PLMD::pi)) * natoms);
    box[0] = rmax;
    box[1] = 0.0;
    box[2] = 0.0;
    box[3] = 0.0;
    box[4] = rmax;
    box[5] = 0.0;
    box[6] = 0.0;
    box[7] = 0.0;
    box[8] = rmax;
  };
};

struct uniformCube : public AtomDistribution {
  void positions(std::vector<Vector> &posToUpdate, unsigned /*step*/,
                 Random &rng) override {
    // giving more or less a cubic udm of volume for each atom: V = nat
    const double rmax = std::cbrt(static_cast<double>(posToUpdate.size()));

    // std::generate(posToUpdate.begin(),posToUpdate.end(),[&]() {
    //   return Vector (rndR(rng),rndR(rng),rndR(rng));
    // });
    auto s = posToUpdate.begin();
    auto e = posToUpdate.end();
    // I am using the iterators:this is slightly faster,
    //  enough to overcome the cost of the vtable that I added
    for (unsigned i = 0; s != e; ++s, ++i) {
      *s = Vector(rng.RandU01() * rmax, rng.RandU01() * rmax,
                  rng.RandU01() * rmax);
    }
  }
  void box(std::vector<double> &box, unsigned natoms, unsigned /*step*/,
           Random &) override {
    //+0.05 to avoid overlap
    const double rmax = std::cbrt(natoms) + 0.05;
    box[0] = rmax;
    box[1] = 0.0;
    box[2] = 0.0;
    box[3] = 0.0;
    box[4] = rmax;
    box[5] = 0.0;
    box[6] = 0.0;
    box[7] = 0.0;
    box[8] = rmax;
  }
};

struct tiledSimpleCubic : public AtomDistribution {
  void positions(std::vector<Vector> &posToUpdate, unsigned /*step*/,
                 Random &rng) override {
    // Tiling the space in this way will not tests 100% the pbc, but
    // I do not think that write a spacefilling curve, like Hilbert, Peano or
    // Morton could be a good idea, in this case
    const unsigned rmax =
        std::ceil(std::cbrt(static_cast<double>(posToUpdate.size())));

    auto s = posToUpdate.begin();
    auto e = posToUpdate.end();
    // I am using the iterators:this is slightly faster,
    //  enough to overcome the cost of the vtable that I added
    for (unsigned k = 0; k < rmax && s != e; ++k) {
      for (unsigned j = 0; j < rmax && s != e; ++j) {
        for (unsigned i = 0; i < rmax && s != e; ++i) {
          *s = Vector(i, j, k);
          ++s;
        }
      }
    }
  }
  void box(std::vector<double> &box, unsigned natoms, unsigned /*step*/,
           Random &) override {
    const double rmax = std::ceil(std::cbrt(static_cast<double>(natoms)));
    ;
    box[0] = rmax;
    box[1] = 0.0;
    box[2] = 0.0;
    box[3] = 0.0;
    box[4] = rmax;
    box[5] = 0.0;
    box[6] = 0.0;
    box[7] = 0.0;
    box[8] = rmax;
  }
};
std::unique_ptr<AtomDistribution>
getAtomDistribution(std::string_view atomicDistr) {
  std::unique_ptr<AtomDistribution> distribution;
  if (atomicDistr == "line") {
    distribution = std::make_unique<theLine>();
  } else if (atomicDistr == "cube") {
    distribution = std::make_unique<uniformCube>();
  } else if (atomicDistr == "sphere") {
    distribution = std::make_unique<uniformSphere>();
  } else if (atomicDistr == "globs") {
    distribution = std::make_unique<twoGlobs>();
  } else if (atomicDistr == "sc") {
    distribution = std::make_unique<tiledSimpleCubic>();
  } else {
    plumed_error()
        << R"(The atomic distribution can be only "line", "cube", "sphere", "globs" and "sc", the input was ")"
        << atomicDistr << '"';
  }
  return distribution;
}

void plumedBenchmark(benchmark::State &state, std::string kernelPath,
                     std::string plumedFile, int nsteps, unsigned natoms,
                     std::string adString) {
  std::unique_ptr<AtomDistribution> distribution =
      getAtomDistribution(adString);
  struct FileDeleter {
    void operator()(FILE *f) const noexcept {
      if (f)
        std::fclose(f);
    }
  };
  std::unique_ptr<FILE, FileDeleter> out{std::fopen("benchmark.out", "w")};
  Log log;
  log.link(out.get());
  Kernel kernel(kernelPath, plumedFile, &log);
  // deterministic initializations to avoid issues with MPI
  generator rng;
  PLMD::Random atomicGenerator;
  int plumedStopCondition = 0;

  log.setLinePrefix("BENCH:  ");
  log << "Welcome to PLUMED benchmark\n";
  std::vector<Kernel> kernels;

  const auto initial_time = std::chrono::high_resolution_clock::now();

  auto &p(kernel.handle);

  // if (Communicator::plumedHasMPI() && domain_decomposition){
  //  p.cmd("setMPIComm", &pc.Get_comm());
  //                  }
  p.cmd("setRealPrecision", (int)sizeof(double));
  p.cmd("setMDLengthUnits", 1.0);
  p.cmd("setMDChargeUnits", 1.0);
  p.cmd("setMDMassUnits", 1.0);
  p.cmd("setMDEngine", "benchmarks");
  p.cmd("setTimestep", 1.0);
  p.cmd("setPlumedDat", plumedFile.c_str());
  //   p.cmd("setLog", out);
  p.cmd("setNatoms", natoms);
  p.cmd("init");

  std::vector<double> cell(9);
  std::vector<double> virial(9);
  std::vector<Vector> pos(natoms);
  std::vector<Vector> forces(natoms);
  std::vector<double> masses(natoms, 1);
  std::vector<double> charges(natoms, 0);
  std::vector<int> shuffled_indexes;
  std::iota(shuffled_indexes.begin(), shuffled_indexes.end(), 0);
  int step = 0;
  for (auto _ : state) {
    distribution->positions(pos, step, atomicGenerator);
    distribution->box(cell, natoms, step, atomicGenerator);
    double *pos_ptr;
    double *for_ptr;
    double *charges_ptr;
    double *masses_ptr;
    int *indexes_ptr = nullptr;
    int n_local_atoms;
    pos_ptr = &pos[0][0];
    for_ptr = &forces[0][0];
    charges_ptr = &charges[0];
    masses_ptr = &masses[0];
    n_local_atoms = natoms;
    indexes_ptr = shuffled_indexes.data();

    auto &p(kernel.handle);

    p.cmd("setStep", step);
    p.cmd("setStopFlag", &plumedStopCondition);
    p.cmd("setForces", for_ptr, {n_local_atoms, 3});
    p.cmd("setBox", &cell[0], {3, 3});
    p.cmd("setVirial", &virial[0], {3, 3});
    p.cmd("setPositions", pos_ptr, {n_local_atoms, 3});
    p.cmd("setMasses", masses_ptr, {n_local_atoms});
    p.cmd("setCharges", charges_ptr, {n_local_atoms});
    //   if (shuffled) {
    //     p.cmd("setAtomsNlocal", n_local_atoms);
    //     p.cmd("setAtomsGatindex", indexes_ptr, {n_local_atoms});
    //   }
    p.cmd("prepareCalc");

    p.cmd("performCalc");

    ++step;
  }
}
} // namespace PLMD

int main(int argc, char **argv) {

  std::string kp = "./libplumedKernel.so";
  std::string plumedPath = "./plumed.dat";
  benchmark::RegisterBenchmark("try", PLMD::plumedBenchmark, kp, plumedPath,
                               1000, 1000, "sc");
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}