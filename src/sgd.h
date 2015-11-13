#pragma once
#include <cmath>
#include <vector>
#include <dmlc/data.h>
#include "sample_set.h"
#include "config.h"


namespace sgd {

class SgdSolver {
 public:
  typedef float real_t;
  typedef std::size_t IndexType;
  typedef ::dmlc::Row<IndexType> Row;
  typedef ::admm::SampleSet SampleSet;

  SgdSolver();
  ~SgdSolver();

  void Init(std::size_t dim,
            real_t step_size_init,
            real_t l_2_init,
            const std::vector<real_t>& offset_init,
            const std::vector<real_t>& reg_offset_init);

  void Run(SampleSet& sample_set);

  real_t Predict(const Row& x);

  std::vector<real_t> GetWeight() const;

  real_t step_size_;
 private:
  std::size_t dim_;
  real_t l_2_;
  std::vector<real_t> weight_;
  std::vector<real_t> offset_;
  std::vector<real_t> reg_offset_;
}; 

}
