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
  
  AdmmConfig() {
  }
  
  ~AdmmConfig() {
  }
  
  real_t step_size;
  real_t global_var;
  real_t bias_var;
  real_t ftrl_alpha;
  std::size_t dim;
  int passes;
  std::vector<real_t> global_weights;

  std::string train_path;
  std::string test_path;
  std::string output_path;
};

class FtrlConfig {
  typedef float real_t;
 public:
  FtrlConfig() {
  }
  
  FtrlConfig(const AdmmConfig& admm_params) : alpha(admm_params.ftrl_alpha), beta(1), dim(admm_params.dim) {
  }
  
  ~FtrlConfig() {
  }

  real_t alpha;
  real_t beta;
  real_t l_1;
  real_t l_2;
  std::size_t dim;
  int passes;
  std::string train_path;
  std::string test_path;
  std::string output_path;
};
} // namespace admm
