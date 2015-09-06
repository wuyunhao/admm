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
  typedef std::vector<real_t> Vec;

  bool global_update(const std::vector<Vec>& workers, AdmmConfig& admm_params);
};
} // namespace admm
