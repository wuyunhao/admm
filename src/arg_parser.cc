#include "arg_parser.h"
#include <fstream>
#include <vector>
#include <map>
#include <rabit.h>
#include <cstdlib>

namespace admm {
  
void ArgParser::ADMMParse(const char* file, AdmmConfig& admm_params) {
  std::fstream in;
  in.open(file, std::fstream::in);
  if (! in.is_open()) {
    rabit::TrackerPrintf("config file open failed. \n");
    return;
  }
  std::map<std::string, std::string> dict;
  std::string key, value;
  while (!in.eof()) {
    in >> key;
    in >> value;
    if (in.fail()) break;
    if (value == "=") {
      in >> value;
      dict[key] = value;
    }
    if (value == ":") {
      in >> value;
      psid.push_back(key);
      num_part.push_back(atoi(value.c_str()));
    }
  }
  in.close();

  admm_params.train_path = dict["train_path"];
  admm_params.test_path = dict["test_path"];
  admm_params.output_path = dict["output_path"];

  admm_params.step_size = atof(dict["step_size"].c_str());
  admm_params.bias_var = atof(dict["bias_var"].c_str());
  admm_params.global_var = atof(dict["global_var"].c_str());
  admm_params.ftrl_alpha = atof(dict["ftrl_alpha"].c_str());
  admm_params.passes = atoi(dict["passes"].c_str());
  admm_params.dim = static_cast<size_t>(atoi(dict["dim"].c_str()));

  admm_params.global_weights.resize(admm_params.dim);
  std::fill(admm_params.global_weights.begin(), admm_params.global_weights.end(), 0.0f);

  if (dict.count("load_path"))
    load_path = dict["load_path"];
}

void ArgParser::FTRLParse(const char* file, FtrlConfig& ftrl_params) {
  std::fstream in;
  in.open(file, std::fstream::in);
  if (! in.is_open()) {
    rabit::TrackerPrintf("config file open failed. \n");
    return;
  }
  std::map<std::string, std::string> dict;
  std::string key, value;
  while (!in.eof()) {
    in >> key;
    in >> value;
    if (value == "=") {
      in >> value;
      dict[key] = value;
    }
    if (value == ":") {
      in >> value;
      psid.push_back(key);
      num_part.push_back(atoi(value.c_str()));
    }
  }
  in.close();

  ftrl_params.train_path = dict["train_path"];
  ftrl_params.test_path = dict["test_path"];
  ftrl_params.output_path = dict["output_path"];

  ftrl_params.alpha = atof(dict["ftrl_alpha"].c_str());
  ftrl_params.beta = atof(dict["ftrl_beta"].c_str());
  ftrl_params.l_1 = atof(dict["l_1"].c_str());
  ftrl_params.l_2 = atof(dict["l_2"].c_str());
  ftrl_params.passes = atoi(dict["passes"].c_str());
  ftrl_params.dim = static_cast<size_t>(atoi(dict["dim"].c_str()));

  if (dict.count("load_path"))
    load_path = dict["load_path"];
}
}
