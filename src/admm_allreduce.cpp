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

using namespace dmlc;
using namespace rabit;

float Predict(dmlc::Row<std::size_t>& x, std::vector<float>& weight_vec) {
  auto* ptr_weight = &weight_vec[0];
  auto inner_product = x.SDot(ptr_weight, weight_vec.size());

  return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

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
     os << '\n';
     for (size_t i = 0; i < worker_processor_.langr_vec_.size(); ++i) {
       os << worker_processor_.langr_vec_[i] << ' ';
     }

   }
   void LogLoss(::admm::SampleSet &sample_set) {
     std::vector<float> final_weights(worker_processor_.base_vec_.size());
     for (size_t i = 0; i < final_weights.size(); ++i) {
        final_weights[i] = worker_processor_.base_vec_[i] + worker_processor_.bias_vec_[i];
     }
     float sum = 0.0;
     int count = 0;
     sample_set.Rewind();
     while(sample_set.Next()) {
       dmlc::Row<std::size_t> x = sample_set.GetData();
       auto predict = Predict(x,  final_weights);
       sum += (int)x.label == 1? -log(predict): -log(1.0f - predict);
       count++;
     }
     rabit::TrackerPrintf("[INFO] LOGLOSS is %f\n", sum/count);
   }
   void SaveAuc(dmlc::Stream *fo, ::admm::SampleSet &sample_set) {
     dmlc::ostream os(fo);
     sample_set.Rewind();
     while(sample_set.Next()) {
       dmlc::Row<std::size_t> x = sample_set.GetData();
       os << x.label << ' ';
     }
     os << '\n';

     std::vector<float> final_weights(worker_processor_.base_vec_.size());
     for (size_t i = 0; i < final_weights.size(); ++i) {
        final_weights[i] = worker_processor_.base_vec_[i] + worker_processor_.bias_vec_[i];
     }

     sample_set.Rewind();
     while(sample_set.Next()) {
       dmlc::Row<std::size_t> x = sample_set.GetData();
       auto predict = Predict(x, final_weights);
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
  ::admm::SampleSet train_set;

  std::string path = argv[6];
  std::string pid_name(5, '0'); 
  if (argc > 8) {
    pid_name = argv[8];
  } else {
    sprintf(&pid_name[0], "%05d", rabit::GetRank() + 1);
  }
  std::string train_name = path + pid_name + ".train";
  std::string test_name = path + pid_name + ".test";
  CHECK(train_set.Initialize(train_name, 0, 1));

  //get the test set
  ::admm::SampleSet test_set;
  CHECK(test_set.Initialize(test_name, 0, 1)); 

  int max_iter = atoi(argv[7]);
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

    auto lazy_ftrl = [&]()
    {
      local_model.worker_processor_.BiasUpdate(train_set, global_model.admm_params_);
      local_model.worker_processor_.BaseUpdate(train_set, global_model.admm_params_);
      local_model.worker_processor_.GetWeights(global_model.admm_params_, tmp);
    };

    rabit::TrackerPrintf("begin allreduce \n");

    LOG(INFO) << "the " << rabit::GetRank() << " reduce begins " << rabit::VersionNumber() << "\n";
    rabit::Allreduce<op::Sum>(&tmp[0], dim, lazy_ftrl);
    LOG(INFO) << "the " << rabit::GetRank() << " reduce ends " << rabit::VersionNumber() << "\n";

    rabit::TrackerPrintf("allreduce finished\n");

    master_processor.GlobalUpdate(tmp, global_model.admm_params_, rabit::GetWorldSize());

    local_model.worker_processor_.LangrangeUpdate(train_set, global_model.admm_params_);
    rabit::LazyCheckPoint(&global_model);
    local_model.LogLoss(train_set);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
    std::string local_auc = path + "admm_auc_" + pid_name + "_" + std::to_string(r); 
    auto *streamb(dmlc::Stream::Create(&local_auc[0], "w"));
    local_model.SaveAuc(streamb, train_set);
    delete streamb;
  }

  std::string local_params = path + "local_params_" + pid_name;
  auto *stream = dmlc::Stream::Create(&local_params[0], "w");
  local_model.Save(stream);
  delete stream;

  if (rabit::GetRank() == 0) {
    std::string global_file = path + "global_params";
    auto *streama(dmlc::Stream::Create(&global_file[0], "w"));
    global_model.Save(streama);
    delete streama;
  }

  rabit::Finalize();
}
