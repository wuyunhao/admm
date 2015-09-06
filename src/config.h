#pragma once
#include <vector>
#include "sample_set.h"

namespace admm {
/*!
 * \brief Admm parameters, including iterative step size, variance
 *        of global weights w_0 and variance of bias weights v_t, 
 *        in addition to global weights w_0.
 */
class AdmmConfig {
 public:
  typedef float real_t;
  typedef uint32_t IndexType;
  typedef ::dmlc::Row<IndexType> Row;
  
  AdmmConfig(real_t step_size_init,
             real_t global_var_init,
             real_t bias_var_init,
             std::size_t dim_init) : step_size(step_size_init), global_var(global_var_init), bias_var(bias_var_init) {
      global_weights.resize(dim_init);
      for(auto i = 0u; i < global_weights.size(); ++i) {
          global_weights[i] = 0;
      }
  }
  
  AdmmConfig() {
  }
  
  ~AdmmConfig() {
  }
  
  real_t step_size;
  real_t global_var;
  real_t bias_var;
  std::vector<real_t> global_weights;
};

class FtrlConfig {
  typedef float real_t;
 public:
  FtrlConfig() {
  }
  
  FtrlConfig(const AdmmConfig& admm_params) : alpha(1), beta(1), niter(1), dim(admm_params.global_weights.size()) {
  }
  
  ~FtrlConfig() {
  }
  
  real_t alpha;
  real_t beta;
  real_t l_1;
  real_t l_2;
  std::size_t niter;
  std::size_t dim;
};
} // namespace admm
