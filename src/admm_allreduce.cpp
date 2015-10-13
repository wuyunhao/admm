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

float Predict(dmlc::Row<std::size_t>& x, std::vector<float>& weight_vec) {
  auto inner_product = x.SDot(&weight_vec[0], weight_vec.size());

  return 1.0f/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
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
   void LogLoss(::admm::SampleSet &sample_set, ::admm::AdmmConfig &admm_params, bool T) {
     float sum = 0;
     int count = 0;
     sample_set.Rewind();
     while(sample_set.Next()) {
       dmlc::Row<std::size_t> x = sample_set.GetData();
       auto inner_product = x.SDot(&worker_processor_.bias_vec_[0], worker_processor_.bias_vec_.size()) + x.SDot(&worker_processor_.base_vec_[0], worker_processor_.base_vec_.size());
       auto predict = 1.0f/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
       sum += (int)x.label == 1? -log(predict): -log(1 - predict);
       count++;
     }
     if (T)
       rabit::TrackerPrintf("The %d processor train LogLoss is %f \n", rabit::GetRank(), sum/count);
     else
       rabit::TrackerPrintf("The %d processor test LogLoss is %f \n", rabit::GetRank(), sum/count);
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
  Metrics metrics;

  std::string path = argv[6];
  std::string pid_name(5, '0'); 
  if (argc > 8) {
    pid_name = argv[8];
  } else {
    sprintf(&pid_name[0], "%05d", rabit::GetRank() + 1);
  }
  std::string train_name = path + pid_name + ".train";
  std::string test_name = path + pid_name + ".test";

  ::admm::SampleSet train_set;
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
      local_model.worker_processor_.BiasUpdate(train_set, test_set, global_model.admm_params_);
      //LOG(INFO) << r << "th BiasUpdate Logloss \n";
      //local_model.LogLoss(train_set, global_model.admm_params_);
      local_model.worker_processor_.BaseUpdate(train_set, test_set, global_model.admm_params_);
      //LOG(INFO) << r << "th BaseUpdate Logloss \n";
      //local_model.LogLoss(train_set, global_model.admm_params_);
      local_model.worker_processor_.GetWeights(global_model.admm_params_, tmp);
    };
    rabit::Allreduce<op::Sum>(&tmp[0], dim, lazy_ftrl);

    master_processor.GlobalUpdate(tmp, global_model.admm_params_, rabit::GetWorldSize());
    local_model.worker_processor_.LangrangeUpdate(train_set, global_model.admm_params_);
    rabit::LazyCheckPoint(&global_model);

    //local_model.LogLoss(train_set, global_model.admm_params_, true);
    //local_model.LogLoss(test_set, global_model.admm_params_, false);

    std::vector<std::vector<float>> group_w;
    group_w.push_back(local_model.worker_processor_.base_vec_);
    group_w.push_back(local_model.worker_processor_.bias_vec_);
    metrics.LogLoss(train_set, group_w, true);
    metrics.LogLoss(test_set, group_w, false);

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
