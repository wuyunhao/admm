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

using namespace dmlc;
using namespace rabit;

class LocalModel : public dmlc::Serializable {
  public:
   ::admm::Worker worker_processor_;

   void Load(dmlc::Stream *fi) {
   }

   void Save(dmlc::Stream *fo) const {
     dmlc::ostream os(fo);
     for (size_t i = 0; i < worker_processor_.base_vec_.size(); ++i) {
       os << worker_processor_.base_vec_[i] << ' ';
     }
     os << '\n';
     for (size_t i = 0; i < worker_processor_.bias_vec_.size(); ++i) {
       os << worker_processor_.bias_vec_[i] << ' ';
     }
   }

   void SaveAuc(dmlc::Stream *fo, ::admm::SampleSet &sample_set) {
     dmlc::ostream os(fo);
     sample_set.Rewind();
     while(sample_set.Next()) {
       dmlc::Row<std::size_t> x = sample_set.GetData();
       os << x.label << ' ';
     }
     os << '\n';

     sample_set.Rewind();
     while(sample_set.Next()) {
       dmlc::Row<std::size_t> x = sample_set.GetData();
       auto inner_product = x.SDot(&worker_processor_.bias_vec_[0], worker_processor_.bias_vec_.size()) + x.SDot(&worker_processor_.base_vec_[0], worker_processor_.base_vec_.size());
       auto predict = 1.0f/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
       os << predict << ' ';
     }
   }

   void InitModel(std::size_t fdim) {
     worker_processor_.InitWorker(fdim);
   }
};

class GlobalModel : public dmlc::Serializable {
  public:
   ::admm::AdmmConfig admm_params_;

   void Load(dmlc::Stream *fi) {
   }
   void Save(dmlc::Stream *fo) const {
     dmlc::ostream os(fo);
     os << admm_params_.step_size << '\n';
     os << admm_params_.global_var << '\n';
     os << admm_params_.bias_var << '\n';
     os << admm_params_.dim << '\n';
     for (size_t i = 0; i < admm_params_.dim; ++i) {
       os << admm_params_.global_weights[i] << ' ';
     }
   }
   void InitModel(float global_var,
                  float bias_var,
                  float step_size,
                  float alpha,
                  std::size_t dim) {
     admm_params_.Init(global_var, bias_var, step_size, alpha, dim);
   }
};

int main(int argc, char* argv[]) {
  InitLogging(argv[0]);
  rabit::Init(argc, argv);
  
  LocalModel local_model;
  GlobalModel global_model;
  Metrics metrics;

  int max_iter = atoi(argv[6]);
  std::string train_path = argv[7];
  std::string test_path = argv[8];
  std::string output_path = argv[9];
  std::string pid_name = argv[10 + rabit::GetRank()];
    
  std::string train_name = train_path + pid_name; 
  std::string test_name = test_path + pid_name; 

  int iter = rabit::LoadCheckPoint(&global_model);
  if (iter == 0) {
    global_model.InitModel(atof(argv[1]),
                           atof(argv[2]),
                           atof(argv[3]),
                           atof(argv[4]),
                           atoi(argv[5]));
    local_model.InitModel(atoi(argv[5]));
  }
  rabit::TrackerPrintf("Initialization finished\n");

  std::size_t dim = global_model.admm_params_.dim;
  ::admm::Master master_processor;
  std::vector<float> tmp;
  tmp.resize(dim);

  for (int r = iter; r < max_iter; ++r) {
    std::fill(tmp.begin(), tmp.end(), 0.0f);   
    ::admm::SampleSet* train_set = new ::admm::SampleSet;
    CHECK(train_set->Initialize(train_name, 0, 1));
    ::admm::SampleSet* test_set = new ::admm::SampleSet; 

    rabit::TrackerPrintf("start allreduce \n");
    auto lazy_ftrl = [&]()
    {
      //local_model.worker_processor_.BiasUpdate(*train_set, *test_set, global_model.admm_params_);
      local_model.worker_processor_.BaseUpdate(*train_set, *test_set, global_model.admm_params_);
      local_model.worker_processor_.GetWeights(global_model.admm_params_, tmp);
    };
    rabit::Allreduce<op::Sum>(&tmp[0], dim, lazy_ftrl);


    master_processor.GlobalUpdate(tmp, global_model.admm_params_, rabit::GetWorldSize());
    local_model.worker_processor_.LangrangeUpdate(*train_set, global_model.admm_params_);
    rabit::LazyCheckPoint(&global_model);


    std::vector<std::vector<float>> group_w;
    group_w.push_back(local_model.worker_processor_.base_vec_);
    //group_w.push_back(local_model.worker_processor_.bias_vec_);

    metrics.LogLoss(*train_set, group_w, true);
    metrics.Auc(*train_set, group_w, true);
    delete train_set;

    CHECK(test_set->Initialize(test_name, 0, 1)); 
    metrics.LogLoss(*test_set, group_w, false);
    metrics.Auc(*test_set, group_w, false);
    delete test_set;

    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }
  //std::string local_auc = dest_path + "admm_train_auc_" + pid_name;
  //auto *streamb(dmlc::Stream::Create(&local_auc[0], "w"));
  //local_model.SaveAuc(streamb, test_set);
  //delete streamb;

  std::string local_params = output_path + "admm_weight_" + pid_name;
  auto *stream = dmlc::Stream::Create(&local_params[0], "w");
  local_model.Save(stream);
  delete stream;

  //if (rabit::GetRank() == 0) {
  std::string global_file = output_path + "global_params" + pid_name;
  auto *streama(dmlc::Stream::Create(&global_file[0], "w"));
  global_model.Save(streama);
  delete streama;
  //}

  rabit::Finalize();
  return 0;
}
