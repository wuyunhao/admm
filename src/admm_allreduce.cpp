#include <memory>
#include <cstdlib>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "workers.h"
#include "master.h"
#include "config.h"

using namespace dmlc;
using namespace rabit;

//typedef std::size_t IndexType;
//typedef ::dmlc::Row<IndexType> Row;

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
       os << Predict(x, final_weights) << ' ';
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
                  std::size_t dim) {
     admm_params_.Init(global_var, bias_var, step_size, dim);
   }
};

int main(int argc, char* argv[]) {
  InitLogging(argv[0]);
  rabit::Init(argc, argv);
  
  LocalModel local_model;
  GlobalModel global_model;
  ::admm::SampleSet sample_set;
  CHECK(sample_set.Initialize(argv[5], rabit::GetRank(), rabit::GetWorldSize()));

  int max_iter = 3;
  int iter = rabit::LoadCheckPoint(&global_model);
  if (iter == 0) {
    global_model.InitModel(atof(argv[1]),
                           atof(argv[2]),
                           atof(argv[3]),
                           atoi(argv[4]));
    local_model.InitModel(atoi(argv[4]));
  }

  rabit::TrackerPrintf("Initialization finished\n");

  std::size_t dim = global_model.admm_params_.dim;
  ::admm::Master master_processor;
  std::vector<float> tmp;

  for (int r = iter; r < max_iter; ++r) {
    tmp.resize(dim);
    std::fill(tmp.begin(), tmp.end(), 0.0f);   

    auto lazy_ftrl = [&]()
    {
      local_model.worker_processor_.BiasUpdate(sample_set, global_model.admm_params_);
      local_model.worker_processor_.BaseUpdate(sample_set, global_model.admm_params_);
      local_model.worker_processor_.GetWeights(global_model.admm_params_, tmp);
    };

    rabit::TrackerPrintf("allreduce begined\n");

    LOG(INFO) << "the " << rabit::GetRank() << " reduce begins " << rabit::VersionNumber() << "\n";
    rabit::Allreduce<op::Sum>(&tmp[0], tmp.size(), lazy_ftrl);
    LOG(INFO) << "the " << rabit::GetRank() << " reduce ends " << rabit::VersionNumber() << "\n";

    rabit::TrackerPrintf("allreduce finished\n");

    //if (rabit::GetRank() == 0) {
    master_processor.GlobalUpdate(tmp, global_model.admm_params_, rabit::GetWorldSize());
    //}
    //rabit::Broadcast(&(global_model.admm_params_.global_weights), 0);

    local_model.worker_processor_.LangrangeUpdate(sample_set, global_model.admm_params_);
    rabit::LazyCheckPoint(&global_model);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }

  std::string path = "data/"; // "hdfs://ns1/user/yunhao1/admm/";

  std::string local_file = path + "local_params_" + std::to_string(rabit::GetRank());
  auto *stream = dmlc::Stream::Create(&local_file[0], "w");
  local_model.Save(stream);
  delete stream;

  if (rabit::GetRank() == 0) {
    std::string global_file = path + "global_params";
    auto *streama(dmlc::Stream::Create(&global_file[0], "w"));
    global_model.Save(streama);
    delete streama;
  }

  if (rabit::GetRank() == 0) {
    std::string auc = path + "admm_auc"; 
    auto *streamb(dmlc::Stream::Create(&auc[0], "w"));
    //get the test set
    ::admm::SampleSet test_set;
    CHECK(test_set.Initialize(argv[6], rabit::GetRank(),1)); 
    local_model.SaveAuc(streamb, test_set);
    delete streamb;
  }

  rabit::Finalize();
}
