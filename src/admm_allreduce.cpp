#include "admm.h"

using namespace dmlc;
using namespace rabit;

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
    if (arg_parser.load_path.size() != 0) {
      std::string local_load = arg_parser.load_path + "admm_weight_" + local_model.worker_processor_.psid_ + 
                               std::to_string(local_model.worker_processor_.partid_);
      auto *local_stream = dmlc::Stream::Create(&local_load[0], "r");
      local_model.Load(local_stream);
      delete local_stream;

      std::string global_load = arg_parser.load_path + "global_params_" + local_model.worker_processor_.psid_ + 
                               std::to_string(local_model.worker_processor_.partid_);
      auto *global_stream = dmlc::Stream::Create(&global_load[0], "r");
      global_model.Load(global_stream);
      delete global_stream;
    }
  }

  std::string test_name = global_model.admm_params_.test_path + local_model.worker_processor_.psid_ + "_aggregated/part-00000"; 

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
      local_model.worker_processor_.BiasUpdate(*train_set, *test_set, global_model.admm_params_);
      local_model.worker_processor_.BaseUpdate(*train_set, *test_set, global_model.admm_params_);
      local_model.worker_processor_.GetWeights(global_model.admm_params_, tmp);
    };
    rabit::Allreduce<op::Sum>(&tmp[0], dim, lazy_ftrl);

    std::vector<std::vector<float>> group_w;
    group_w.push_back(local_model.worker_processor_.base_vec_);
    group_w.push_back(local_model.worker_processor_.bias_vec_);

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
  local_model.SaveState(stream);
  delete stream;

  //if (rabit::GetRank() == 0) {
  std::string global_file = global_model.admm_params_.output_path + "global_params_" + local_model.worker_processor_.psid_ + 
                             std::to_string(local_model.worker_processor_.partid_);
  auto *streama(dmlc::Stream::Create(&global_file[0], "w"));
  global_model.Save(streama);
  delete streama;
  //}

  rabit::Finalize();
  return 0;
}
