#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "workers.h"
#include "master.h"
#include "config.h"
#include "metrics.h"
#include "arg_parser.h"

using namespace dmlc;
using namespace rabit;

class LocalModel : public dmlc::Serializable {
  public:
   ::admm::Worker worker_processor_;

   void Load(dmlc::Stream *fi) {
     dmlc::istream is(fi);
     std::string value;
     while(!is.eof()) {
       // w_t recovery
       is >> value;
       float val = atof(value.c_str());
       worker_processor_.base_vec_.push_back(val);

       // v_t recovery
       is >> value;
       val = atof(value.c_str());
       worker_processor_.bias_vec_.push_back(val);
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
     for (size_t i = 0; i < worker_processor_.bias_vec_.size(); ++i) {
       os << worker_processor_.base_vec_[i] << ' ' << worker_processor_.bias_vec_[i] << '\n';
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
     worker_processor_.psid_ = arg_parser.psid[No_psid];
     worker_processor_.num_part_ = arg_parser.num_part[No_psid];
     worker_processor_.partid_ = rabit::GetRank() - pre_sum_part;
   }
};

class GlobalModel : public dmlc::Serializable {
  public:
   ::admm::AdmmConfig admm_params_;

   void Load(dmlc::Stream *fi) {
     dmlc::istream is(fi);
     is >> admm_params_.step_size;
     is >> admm_params_.global_var;
     is >> admm_params_.bias_var;
     is >> admm_params_.dim;
     while(!is.eof()) {
       std::string line;
       std::getline(is, line, ':');
       std::getline(is, line, '\n');
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

int main(int argc, char* argv[]) {
  InitLogging(argv[0]);
  rabit::Init(argc, argv);
  
  LocalModel local_model;
  GlobalModel global_model;
  Metrics metrics;
  ::admm::Master master_processor;

  int iter = rabit::LoadCheckPoint(&global_model);
  if (iter == 0) {
    ::admm::ArgParser arg_parser;
    global_model.InitModel(argv[1], arg_parser);
    local_model.InitModel(global_model.admm_params_.dim, arg_parser);
  }

  std::string test_name = global_model.admm_params_.test_path + local_model.worker_processor_.psid_ + "_aggregated/part-00000"; 

  rabit::TrackerPrintf("Initialization finished\n");

  int max_iter = global_model.admm_params_.passes;
  std::size_t dim = global_model.admm_params_.dim;
  std::vector<float> tmp;
  tmp.resize(dim);

  for (int r = iter; r < max_iter; ++r) {
    std::fill(tmp.begin(), tmp.end(), 0.0f);   
    ::admm::SampleSet* train_set = new ::admm::SampleSet;
    ::admm::SampleSet* test_set = new ::admm::SampleSet; 

    rabit::TrackerPrintf("start allreduce \n");
    auto lazy_ftrl = [&]()
    {
      //local_model.worker_processor_.BiasUpdate(*train_set, *test_set, global_model.admm_params_);
      local_model.worker_processor_.BaseUpdate(*train_set, *test_set, global_model.admm_params_);
      local_model.worker_processor_.GetWeights(global_model.admm_params_, tmp);
    };
    rabit::Allreduce<op::Sum>(&tmp[0], dim, lazy_ftrl);

    std::vector<std::vector<float>> group_w;
    group_w.push_back(global_model.admm_params_.global_weights);
    //group_w.push_back(local_model.worker_processor_.bias_vec_);

    delete train_set;
    CHECK(test_set->Initialize(test_name, 0, 1)); 
    metrics.LogLoss(*test_set, group_w, false);
    metrics.Auc(*test_set, group_w, false);
    delete test_set;

    master_processor.GlobalUpdate(tmp, global_model.admm_params_, rabit::GetWorldSize());
    local_model.worker_processor_.LangrangeUpdate(global_model.admm_params_);
    rabit::LazyCheckPoint(&global_model);

    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }

  std::string local_params = global_model.admm_params_.output_path + "admm_weight_" + local_model.worker_processor_.psid_ + 
                             std::to_string(local_model.worker_processor_.partid_);
  auto *stream = dmlc::Stream::Create(&local_params[0], "w");
  local_model.Save(stream);
  delete stream;

  //if (rabit::GetRank() == 0) {
  std::string global_file = global_model.admm_params_.output_path + "global_params" + local_model.worker_processor_.psid_ + 
                             std::to_string(local_model.worker_processor_.partid_);
  auto *streama(dmlc::Stream::Create(&global_file[0], "w"));
  global_model.Save(streama);
  delete streama;
  //}

  rabit::Finalize();
  return 0;
}
