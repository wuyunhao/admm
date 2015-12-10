#pragma once
#include <string>
#include "config.h"

namespace admm {
class ArgParser {
 public:
  void ADMMParse(const char* file, AdmmConfig& admm_params);
  void FTRLParse(const char* file, FtrlConfig& ftrl_params);

  std::vector<std::string> psid;
  std::vector<int> num_part;
  std::string load_path;
};
} // namespace admm
