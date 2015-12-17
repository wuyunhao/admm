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
     while(!is.eof()) {
       // w_t recovery
       is >> value;
       if (is.fail()) break;
       float val = atof(value.c_str());
       worker_processor_.base_vec_.push_back(val);

       // v_t recovery
       is >> value;
       val = atof(value.c_str());
       worker_processor_.bias_vec_.push_back(val);

       // alppha_t recovery
       is >> value;
       val = atof(value.c_str());
       worker_processor_.langr_vec_.push_back(val);

       // nt recovery
       is >> value;
       val = atof(value.c_str());
       worker_processor_.nt_vec_.push_back(val);

       // zt recovery
       is >> value;
       val = atof(value.c_str());
       worker_processor_.zt_vec_.push_back(val);
     }
   }

   void Save(dmlc::Stream *fo) const {
     dmlc::ostream os(fo);
     for (size_t i = 0; i < worker_processor_.base_vec_.size(); ++i) {
       os << i+1 << ":" << worker_processor_.base_vec_[i] + worker_processor_.bias_vec_[i] << '\n';
     }
   }

   void SaveState(dmlc::Stream *fo) {
     dmlc::ostream os(fo);
     for (size_t i = 0; i < worker_processor_.base_vec_.size(); ++i) {
       os << std::left << std::setw(17) << worker_processor_.base_vec_[i]
          << std::left << std::setw(17) << worker_processor_.bias_vec_[i]
          << std::left << std::setw(17) << worker_processor_.langr_vec_[i]
          << std::left << std::setw(17) << worker_processor_.nt_vec_[i]
          << std::left << std::setw(17) << worker_processor_.zt_vec_[i] << '\n';
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
     while(!is.eof()) {
       std::string line;
       std::getline(is, line, ':');
       std::getline(is, line, '\n');
       if (is.fail()) break;
       admm_params_.global_weights.push_back(atof(line.c_str()));
     }
   }
   void Save(dmlc::Stream *fo) const {
     dmlc::ostream os(fo);
     os << admm_params_.step_size << '\n';
     os << admm_params_.global_var << '\n';
     os << admm_params_.bias_var << '\n';
     os << admm_params_.dim << '\n';
     for (size_t i = 0; i < admm_params_.dim; ++i) {
       os << i+1 << ':' << admm_params_.global_weights[i] << '\n';
     }
   }
   void InitModel(const char* conf, ::admm::ArgParser& arg_parser) {
     arg_parser.ADMMParse(conf, admm_params_);
   }
};
