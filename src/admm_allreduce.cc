#include <memory>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "worker.h"
#include "master.h"
#include "config.h"

using namespace dmlc;
using namespace rabit;
class LocalModel : public dmlc::Serializable {
  public:
   ::admm::Worker worker_processor_;

   virtual void Load(dmlc::Stream *fi) {
   }
   virtual void Save(dmlc::Stream *fo) {
   }
   virtual void InitModel(std::size_t fdim) {
     worker_pprocessor_.InitWorker(fdim);
   }
};

class GlobalModel : public dmlc::Serializable {
  public:
   ::admm::AdmmConfig admm_params_;

   virtual void Load(dmlc::Stream *fi) {
     fi->Read(&admm_params_.step_size, sizeof(admm_params_.step_size));
     fi->Read(&admm_params_.global_var, sizeof(admm_params_.global_var));
     fi->Read(&admm_params_.bias_var, sizeof(admm_params_.bias_var));
     fi->Read(&admm_params_.dim, sizeof(admm_params_.dim));
     fi->Read(&admm_params_.global_weights);
   }
   virtual void Save(dmlc::Stream *fo) {
     fo->Read(&admm_params_.step_size, sizeof(admm_params_.step_size));
     fo->Read(&admm_params_.global_var, sizeof(admm_params_.global_var));
     fo->Read(&admm_params_.bias_var, sizeof(admm_params_.bias_var));
     fo->Read(&admm_params_.dim, sizeof(admm_params_.dim));
     fo->Write(admm_paramms_.global_weights);
   }
   virtual void InitModel(float step_size,
                          float global_var,
                          float bias_var,
                          std::size_t dim) {
     admm_paramms_.Init(step_size, global_var, bias_var, dim);
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
  int iter = rabit::LoadCheckPoint(&global_model, &local_model);
  if (iter == 0) {
    global_model.InitModel(argv[2],
                           argv[3],
                           argv[4],
                           argv[5]);
    local_model.InitModel(argv[5]);
  }

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
    rabit::Allreduce<op::Sum>(&tmp[0], tmp.size(), lazy_ftrl);
    if (rabit::GetRank() == 0) {
        master_processor.GlobalUpdate(tmp, global_model.admm_params_, ranks);
    }
    rabit::Broadcast(&(global_model.admm_params_.global_weights), 0);
    local_model.worker_processor_.LangrangeUpdate(sample_set, global_model.admm_params_);
    rabit::CheckPoint(&global_model, &local_model);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }

  rabit::Finalize();
}
