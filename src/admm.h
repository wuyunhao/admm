#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "workers.h"
#include "master.h"
#include "config.h"
#include "metrics.h"
#include "arg_parser.h"

class LocalModel : public dmlc::Serializable {
  public:
   ::admm::Worker worker_processor_;

   void Load(dmlc::Stream *fi) {
     dmlc::istream is(fi);
     worker_processor_.base_vec_.clear();
     worker_processor_.bias_vec_.clear();
     worker_processor_.langr_vec_.clear();
     worker_processor_.nt_vec_.clear();
     worker_processor_.zt_vec_.clear();
     std::string value;
     int index = 0;
     while(std::getline(is, value)) {
       if (value.size() == 0) continue;
       size_t loc = value.find(':');
       if (loc == std::string::npos) {
         ++index;
         if (index == 1) {
           worker_processor_.base_vec_.resize(std::stoi(value));
           std::fill(worker_processor_.base_vec_.begin(), worker_processor_.base_vec_.end(), 0);
         } else if (index == 2) {
           worker_processor_.bias_vec_.resize(std::stoi(value));
           std::fill(worker_processor_.bias_vec_.begin(), worker_processor_.bias_vec_.end(), 0);
         } else if (index == 3) {
           worker_processor_.langr_vec_.resize(std::stoi(value));
           std::fill(worker_processor_.langr_vec_.begin(), worker_processor_.langr_vec_.end(), 0);
         } else if (index == 4) {
           worker_processor_.nt_vec_.resize(std::stoi(value));
           std::fill(worker_processor_.nt_vec_.begin(), worker_processor_.nt_vec_.end(), 0);
         } else if (index == 5) {
           worker_processor_.zt_vec_.resize(std::stoi(value));
           std::fill(worker_processor_.zt_vec_.begin(), worker_processor_.zt_vec_.end(), 0);
         }
         continue;
       }

       if (index == 1) {
         // w_t recovery
         worker_processor_.base_vec_[std::stoi(value.substr(0, loc))] = std::stof(value.substr(loc+1));
       } else if (index == 2) {
         // v_t recovery
         worker_processor_.bias_vec_[std::stoi(value.substr(0, loc))] = std::stof(value.substr(loc+1));
       } else if (index == 3) {
         // alppha_t recovery
         worker_processor_.langr_vec_[std::stoi(value.substr(0, loc))] = std::stof(value.substr(loc+1));
       } else if (index == 4) {
         // nt recovery
         worker_processor_.nt_vec_[std::stoi(value.substr(0, loc))] = std::stof(value.substr(loc+1));
       } else if (index == 5) {
         // zt recovery
         worker_processor_.zt_vec_[std::stoi(value.substr(0, loc))] = std::stof(value.substr(loc+1));
       }
     }
   }

   void Save(dmlc::Stream *fo) const {
     dmlc::ostream os(fo);
     os << -1 << ":" << worker_processor_.base_vec_[0] << '\n';

     for (size_t i = 1; i < worker_processor_.base_vec_.size(); ++i) {
       float val = worker_processor_.base_vec_[i] + worker_processor_.bias_vec_[i]; 
       if (val > exp(-6.0f) || val < -exp(-6.0f))
         os << i-1 << ":" << val << '\n';
     }
   }

   void SaveState(dmlc::Stream *fo) {
     dmlc::ostream os(fo);
     os << worker_processor_.base_vec_.size() << '\n';
     for (size_t i = 0; i < worker_processor_.base_vec_.size(); ++i) {
       if (worker_processor_.base_vec_[i] != 0)
         os << i << ':' << worker_processor_.base_vec_[i] << '\n';
     }
     os << worker_processor_.bias_vec_.size() << '\n';
     for (size_t i = 0; i < worker_processor_.bias_vec_.size(); ++i) {
       if (worker_processor_.bias_vec_[i] != 0)
         os << i << ':' << worker_processor_.bias_vec_[i] << '\n';
     }
     os << worker_processor_.langr_vec_.size() << '\n';
     for (size_t i = 0; i < worker_processor_.langr_vec_.size(); ++i) {
       if (worker_processor_.langr_vec_[i] != 0)
         os << i << ':' << worker_processor_.langr_vec_[i] << '\n';
     }
     os << worker_processor_.nt_vec_.size() << '\n';
     for (size_t i = 0; i < worker_processor_.nt_vec_.size(); ++i) {
       if (worker_processor_.nt_vec_[i] != 0)
         os << i << ':' << worker_processor_.nt_vec_[i] << '\n';
     }
     os << worker_processor_.zt_vec_.size() << '\n';
     for (size_t i = 0; i < worker_processor_.zt_vec_.size(); ++i) {
       if (worker_processor_.zt_vec_[i] != 0)
         os << i << ':' << worker_processor_.zt_vec_[i] << '\n';
     }
   }

   void InitModel(std::size_t fdim, ::admm::ArgParser& arg_parser) {
     worker_processor_.InitWorker(fdim);
     int sum_part = 0;
     int pre_sum_part = sum_part;
     size_t No_psid = 0;
     for (size_t i = 0; i < arg_parser.psid.size(); ++i) {
       sum_part += arg_parser.num_part[i];
       if (sum_part > rabit::GetRank()) {
         No_psid = i;
         break;
       }
       pre_sum_part = sum_part;
     }
     worker_processor_.psid_ = arg_parser.psid[rabit::GetRank()];
     worker_processor_.num_part_ = arg_parser.num_part[rabit::GetRank()];
     worker_processor_.partid_ = rabit::GetRank() - pre_sum_part;
   }
};

class GlobalModel : public dmlc::Serializable {
  public:
   ::admm::AdmmConfig admm_params_;

   void Load(dmlc::Stream *fi) {
     dmlc::istream is(fi);
     admm_params_.global_weights.clear();
     is >> admm_params_.step_size;
     is >> admm_params_.global_var;
     is >> admm_params_.bias_var;
     is >> admm_params_.dim;
     admm_params_.global_weights.resize(admm_params_.dim);
     std::fill(admm_params_.global_weights.begin(), admm_params_.global_weights.end(), 0);
     while(!is.eof()) {
       std::string index;
       std::string value;
       std::getline(is, index, ':');
       std::getline(is, value, '\n');
       if (is.fail()) break;
       admm_params_.global_weights[atoi(index.c_str())] = atof(value.c_str());
     }
   }
   void Save(dmlc::Stream *fo) const {
     dmlc::ostream os(fo);
     os << admm_params_.step_size << '\n';
     os << admm_params_.global_var << '\n';
     os << admm_params_.bias_var << '\n';
     os << admm_params_.dim << '\n';
     for (size_t i = 0; i < admm_params_.dim; ++i) {
       if (admm_params_.global_weights[i] != 0)
         os << i << ':' << admm_params_.global_weights[i] << '\n';
     }
   }
   void InitModel(const char* conf, ::admm::ArgParser& arg_parser) {
     arg_parser.ADMMParse(conf, admm_params_);
   }
};
