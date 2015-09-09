#pragma once
#include <vector>
#include <dmlc/logging.h>
#include <dmlc/data.h>
#include "config.h"
#include "sample_set.h"

namespace admm{
/*!
 *\brief Master node update the global weight vector w_0
 */
class Master {
 public:
  typedef float real_t;

  bool GlobalUpdate(const std::vector<real_t>& workers, AdmmConfig& admm_params, int num_worker);
};
} // namespace admm
