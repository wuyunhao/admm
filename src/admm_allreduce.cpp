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
   void InitModel(std::size_t fdim) {
     worker_processor_.InitWorker(fdim);
   }
};

class GlobalModel : public dmlc::Serializable {
  public:
   ::admm::AdmmConfig admm_params_;

   void Load(dmlc::Stream *fi) {
     fi->Read(&admm_params_.step_size, sizeof(admm_params_.step_size));
     fi->Read(&admm_params_.global_var, sizeof(admm_params_.global_var));
     fi->Read(&admm_params_.bias_var, sizeof(admm_params_.bias_var));
     fi->Read(&admm_params_.dim, sizeof(admm_params_.dim));
     fi->Read(&admm_params_.global_weights);
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
   void InitModel(float step_size,
                  float global_var,
                  float bias_var,
                  std::size_t dim) {
     admm_params_.Init(step_size, global_var, bias_var, dim);
   }
};
int main(int argc, char* argv[]) {
  InitLogging(argv[0]);
  rabit::Init(argc, argv);
  
  LocalModel local_model;
  GlobalModel global_model;
  ::admm::SampleSet sample_set;
  CHECK(sample_set.Initialize(argv[1], rabit::GetRank(), rabit::GetWorldSize()));

  int max_iter = 1;
  int iter = rabit::LoadCheckPoint(&global_model);
  if (iter == 0) {
    global_model.InitModel(atof(argv[2]),
                           atof(argv[3]),
                           atof(argv[4]),
                           atoi(argv[5]));
    local_model.InitModel(atoi(argv[5]));
  }

  rabit::TrackerPrintf("Initialization finished");

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

    rabit::TrackerPrintf("allreduce begined");

    LOG(INFO) << "the " << rabit::GetRank() << " reduce begins " << rabit::VersionNumber() << "\n";
    rabit::Allreduce<op::Sum>(&tmp[0], tmp.size(), lazy_ftrl);
    LOG(INFO) << "the " << rabit::GetRank() << " reduce ends " << rabit::VersionNumber() << "\n";

    rabit::TrackerPrintf("allreduce finished");

    if (rabit::GetRank() == 0) {
        master_processor.GlobalUpdate(tmp, global_model.admm_params_, rabit::GetWorldSize());
    }
    //rabit::Broadcast(&(global_model.admm_params_.global_weights), 0);

    local_model.worker_processor_.LangrangeUpdate(sample_set, global_model.admm_params_);
    rabit::LazyCheckPoint(&global_model);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }

  rabit::TrackerPrintf("len of base weight is %f \n", local_model.worker_processor_.base_vec_[0]);

  std::string path = "hdfs://ns1/user/yunhao1/admm/";
  std::string filename = "processor_" + std::to_string(rabit::GetRank());
  auto *stream = dmlc::Stream::Create(&filename[0], "w");
  local_model.Save(stream);
  delete stream;
  if (rabit::GetRank() == 0) {
    std::string result = "allprocesses.txt";
    auto *streama(dmlc::Stream::Create(&result[0], "w"));
    global_model.Save(streama);
    delete streama;

    rabit::TrackerPrintf("len of global weight is %f \n", global_model.admm_params_.global_weights[0]);
  }

  rabit::Finalize();
}
